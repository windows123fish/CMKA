"""检测引擎模块"""

import math
import os
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .config import CAMERA_MAX_FAIL, COCO_CLASSES, DEFAULT_CONFIG
from .stats import StatsAggregator
from .tracker import ObjectTracker, TrackedObject
from .utils import (
 build_log_path,
 load_settings,
 logger,
 put_chinese_text,
 save_log_row,
 save_settings,
)


class DetectionEngine:
 """封装模型加载、推理、追踪和日志导出的核心引擎。"""

 def __init__(self, base_path: str) -> None:
  self.base_path: str = base_path
  self.pt_model_path: str = os.path.join(base_path, "yolo26n.pt")
  self.settings: Dict[str, Any] = load_settings(base_path)
  self.state_lock = threading.Lock()

  self.use_ultralytics: bool = False
  self.model: Any = None
  self.classes: Dict[int, str] = dict(COCO_CLASSES)

  self._load_state()
  self._load_model()

  self.tracker = ObjectTracker(
   iou_threshold=self.iou_threshold,
   tracker_mode=self.tracker_mode,
   max_missing=self.max_missing,
  )
  self.stats_aggregator = StatsAggregator()

  self.log_path: Optional[str] = None
  self.exporting: bool = False

  self._stop_event = threading.Event()
  self._camera_thread: Optional[threading.Thread] = None
  self._on_frame: Optional[Callable] = None

 def _load_state(self) -> None:
  s = self.settings
  self.disabled_classes: Set[str] = set(s.get("disabled_classes", []))
  self.show_trajectory: bool = bool(s.get("show_trajectory", True))
  self.show_prediction: bool = bool(s.get("show_prediction", True))
  self.show_boxes: bool = bool(s.get("show_boxes", True))
  tc = s.get("trajectory_color")
  self.trajectory_color: Tuple[int, int, int] = (
   tuple(int(x) for x in tc) if isinstance(tc, (list, tuple)) and len(tc) == 3 else (0, 0, 255)
  )
  pc = s.get("prediction_color")
  self.prediction_color: Tuple[int, int, int] = (
   tuple(int(x) for x in pc) if isinstance(pc, (list, tuple)) and len(pc) == 3 else (0, 255, 255)
  )
  self.tracker_mode: str = str(s.get("tracker_mode", "classic") or "classic")
  self.max_missing: int = int(s.get("max_missing", 10) or 10)
  self.iou_threshold: float = float(s.get("iou_threshold", 0.25) or 0.25)
  self.conf_threshold: float = float(s.get("conf_threshold", 0.5) or 0.5)

 def get_classes(self) -> Dict[int, str]:
  return dict(self.classes)

 def get_settings(self) -> Dict[str, Any]:
  with self.state_lock:
   return {
    "disabled_classes": sorted(self.disabled_classes),
    "show_trajectory": self.show_trajectory,
    "show_prediction": self.show_prediction,
    "show_boxes": self.show_boxes,
    "trajectory_color": list(self.trajectory_color),
    "prediction_color": list(self.prediction_color),
    "tracker_mode": self.tracker_mode,
    "max_missing": self.max_missing,
    "iou_threshold": self.iou_threshold,
    "conf_threshold": self.conf_threshold,
   }

 def get_model_path(self) -> str:
  return self.pt_model_path

 def set_setting(self, key: str, value: Any) -> None:
  with self.state_lock:
   if key == "disabled_classes":
    self.disabled_classes = set(value)
   elif key == "show_trajectory":
    self.show_trajectory = bool(value)
   elif key == "show_prediction":
    self.show_prediction = bool(value)
   elif key == "show_boxes":
    self.show_boxes = bool(value)
   elif key == "trajectory_color" and isinstance(value, (list, tuple)) and len(value) == 3:
    self.trajectory_color = tuple(int(x) for x in value)
   elif key == "prediction_color" and isinstance(value, (list, tuple)) and len(value) == 3:
    self.prediction_color = tuple(int(x) for x in value)
   elif key == "tracker_mode":
    self.tracker_mode = str(value)
   elif key == "max_missing":
    self.max_missing = int(value)
   elif key == "iou_threshold":
    self.iou_threshold = float(value)
   elif key == "conf_threshold":
    self.conf_threshold = float(value)

 def save_settings(self) -> bool:
  settings = {
   "disabled_classes": sorted(self.disabled_classes),
   "show_trajectory": self.show_trajectory,
   "show_prediction": self.show_prediction,
   "show_boxes": self.show_boxes,
   "trajectory_color": list(self.trajectory_color),
   "prediction_color": list(self.prediction_color),
   "tracker_mode": self.tracker_mode,
   "max_missing": self.max_missing,
   "iou_threshold": self.iou_threshold,
   "conf_threshold": self.conf_threshold,
   "camera_id": int(self.settings.get("camera_id", 0)),
   "model_url": self.settings.get("model_url", DEFAULT_CONFIG["model_url"]),
  }
  return save_settings(self.base_path, settings)

 def _load_model(self) -> None:
  self._lazy_load_model()

 def _lazy_load_model(self) -> None:
  if self.model is not None and self.use_ultralytics:
   return
  try:
   from ultralytics import YOLO
   if not os.path.exists(self.pt_model_path):
    raise FileNotFoundError(f"未找到模型文件: {self.pt_model_path}")
   self.model = YOLO(self.pt_model_path)
   self.use_ultralytics = True
   with self.state_lock:
    self.classes = self.model.names
   logger.info("YOLO模型加载成功: %s", self.pt_model_path)
  except ImportError as exc:
   logger.error("未安装 ultralytics 库，请运行: pip install ultralytics")
   raise RuntimeError("缺少 ultralytics 依赖") from exc
  except Exception as exc:
   logger.error("模型加载失败: %s", exc)
   self.use_ultralytics = False
   self.model = None
   raise

 def reload_model(self) -> bool:
  try:
   from ultralytics import YOLO
   candidates = [self.pt_model_path, os.path.join(tempfile.gettempdir(), "cmka_model.pt")]
   for path in candidates:
    if os.path.exists(path):
     try:
      loaded = YOLO(path)
      with self.state_lock:
       self.model = loaded
       self.use_ultralytics = True
       self.classes = loaded.names
       self.pt_model_path = path
      logger.info("模型重载成功: %s", path)
      return True
     except Exception:
      continue
  except Exception:
   pass
  with self.state_lock:
   self.model = None
   self.use_ultralytics = False
  return False

 def download_model(self, url: str) -> Optional[str]:
  try:
   import requests
   from tqdm import tqdm
   from .utils import _model_name_from_url

   dst = os.path.join(self.base_path, _model_name_from_url(url))
   temp_dst = dst + ".tmp"
   resp = requests.get(url, stream=True, timeout=30)
   resp.raise_for_status()
   total = int(resp.headers.get("content-length", 0))
   with open(temp_dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
    for chunk in resp.iter_content(chunk_size=256 * 1024):
     if chunk:
      f.write(chunk)
      pbar.update(len(chunk))
   if total > 0 and os.path.getsize(temp_dst) != total:
    raise RuntimeError("下载大小不匹配")
   os.replace(temp_dst, dst)
   self.pt_model_path = dst
   return dst
  except Exception as exc:
   logger.error("下载失败: %s", exc)
   return None

 @staticmethod
 def _draw_arrow(frame: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
  sx, sy = start
  ex, ey = end
  dx, dy = ex - sx, ey - sy
  dist = math.hypot(dx, dy)
  if dist < 5:
   return frame
  angle = math.atan2(dy, dx)
  arrow_len = min(dist, 30)
  ex = sx + int(arrow_len * math.cos(angle))
  ey = sy + int(arrow_len * math.sin(angle))
  cv2.line(frame, (sx, sy), (ex, ey), color, 3)
  ah = 10
  cv2.line(frame, (ex, ey), (ex - int(ah * math.cos(angle + math.pi / 6)), ey - int(ah * math.sin(angle + math.pi / 6))), color, 3)
  cv2.line(frame, (ex, ey), (ex - int(ah * math.cos(angle - math.pi / 6)), ey - int(ah * math.sin(angle - math.pi / 6))), color, 3)
  return frame

 def _run_detection(self, frame: np.ndarray, model: Any, classes: Dict[int, str], disabled: Set[str]) -> List[Tuple]:
  detections: List[Tuple] = []
  try:
   results = model(frame, conf=self.conf_threshold, iou=0.45, verbose=False)
   for result in results:
    for box in result.boxes:
     x1, y1, x2, y2 = map(int, box.xyxy[0])
     w, h = x2 - x1, y2 - y1
     conf = float(box.conf[0])
     cid = int(box.cls[0])
     if 0 <= cid < len(classes):
      name = classes[cid]
      if name not in disabled:
       detections.append((x1, y1, w, h, cid, conf))
  except Exception as exc:
   logger.error("检测错误: %s", exc)
  return detections

 def _draw_tracked_objects(
  self,
  frame: np.ndarray,
  tracked: List[TrackedObject],
  classes: Dict[int, str],
  show_traj: bool,
  show_pred: bool,
  traj_color: Tuple[int, int, int],
  pred_color: Tuple[int, int, int],
  stats: Dict[str, Any],
  show_boxes: bool = True,
 ) -> np.ndarray:
  for obj in tracked:
   if obj.class_id < 0 or obj.class_id >= len(classes):
    continue
   name = classes[obj.class_id]
   x, y, w, h = obj.x, obj.y, obj.w, obj.h
   if show_boxes:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    frame = put_chinese_text(frame, f"ID:{obj.track_id} {name}: {obj.confidence:.2f}", (x, y - 10), 14, (0, 255, 0))
   if show_traj and len(obj.trajectory) > 1 and name != "person":
    for i in range(1, len(obj.trajectory)):
     cv2.line(frame, obj.trajectory[i - 1], obj.trajectory[i], traj_color, 2)
   if show_pred and name != "person":
    pred = obj.predict_next_position()
    if pred:
     frame = self._draw_arrow(frame, (x + w // 2, y + h // 2), (int(pred[0]), int(pred[1])), pred_color)
   if self.exporting and self.log_path:
    try:
     save_log_row(
      self.log_path,
      时间=time.strftime("%Y-%m-%d %H:%M:%S"),
      摄像头编号=0,
      追踪ID=obj.track_id,
      类别名称=name,
      置信度=f"{obj.confidence:.3f}",
      坐标X=x,
      坐标Y=y,
      框宽W=w,
      框高H=h,
      帧率=stats.get("帧率", ""),
      目标总数=stats.get("目标总数", ""),
     )
    except Exception:
     pass
  return frame

 def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
  """处理单帧图像：检测 → 追踪 → 绘制 → 返回结果。"""
  empty_stats: Dict[str, Any] = {"帧率": 0.0, "平均置信度": 0.0, "目标总数": 0, "类别统计": {}}

  with self.state_lock:
   if self.model is None:
    try:
     self._lazy_load_model()
    except Exception:
     return frame, empty_stats
   if not self.use_ultralytics or self.model is None:
    return frame, empty_stats
   model = self.model
   classes = dict(self.classes)
   disabled = set(self.disabled_classes)
   show_traj = self.show_trajectory
   show_pred = self.show_prediction
   show_boxes = self.show_boxes
   traj_color = self.trajectory_color
   pred_color = self.prediction_color
   mode = self.tracker_mode
   max_missing = self.max_missing
   iou = self.iou_threshold

  detections = self._run_detection(frame, model, classes, disabled)
  if not detections and not self.tracker.tracked_objects:
   return frame, empty_stats

  self.tracker.mode = mode
  self.tracker.max_missing = max_missing
  self.tracker.max_iou_distance = iou
  tracked = self.tracker.update(detections)
  stats = self.stats_aggregator.update(tracked)

  class_counts: Dict[str, int] = {}
  for obj in tracked:
   if obj.class_id < 0 or obj.class_id >= len(classes):
    continue
   name = classes[obj.class_id]
   class_counts[name] = class_counts.get(name, 0) + 1

  stats["类别统计"] = class_counts

  frame = self._draw_tracked_objects(
   frame, tracked, classes, show_traj, show_pred, traj_color, pred_color, stats, show_boxes
  )
  return frame, stats

 def run(self, camera_id: int = 0, on_frame: Optional[Callable] = None) -> None:
  self._on_frame = on_frame
  self._stop_event.clear()
  self._camera_thread = threading.Thread(target=self._camera_loop, args=(camera_id,), daemon=True)
  self._camera_thread.start()

 def stop(self) -> None:
  self._stop_event.set()
  if self._camera_thread is not None:
   self._camera_thread.join(timeout=2)
   self._camera_thread = None

 def _camera_loop(self, camera_id: int) -> None:
  cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
  if not cap.isOpened():
   logger.error("无法打开摄像头 %d", camera_id)
   return
  try:
   fail_count = 0
   while not self._stop_event.is_set():
    ret, frame = cap.read()
    if not ret:
     fail_count += 1
     if fail_count >= CAMERA_MAX_FAIL:
      logger.error("摄像头连续失败，停止检测")
      break
     time.sleep(0.1)
     continue
    fail_count = 0
    frame_out, stats = self.process_frame(frame)
    if callable(self._on_frame):
     self._on_frame(frame_out, stats)
  except Exception as exc:
   logger.error("视频处理错误: %s", exc)
  finally:
   cap.release()
   logger.info("摄像头已释放")

 def start_log_export(self) -> str:
  self.log_path = build_log_path(self.base_path)
  self.exporting = True
  return self.log_path

 def stop_log_export(self) -> None:
  self.exporting = False
  self.log_path = None

import os
import sys
import cv2
import numpy as np
import json
import time
import re
import tempfile
import shutil
import hashlib
import math
import threading
import argparse
from collections import deque
from urllib.parse import urlparse

try:
    from PIL import Image, ImageDraw, ImageFont
    pil_available = True
except ImportError:
    print("未找到PIL库，请安装: pillow")
    sys.exit(1)


def put_chinese_text(img, text, position, font_size=20, color=(0, 0, 0)):
    if not pil_available:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/20, color, 2)
        return img
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = None
    if os.name == 'nt':
        for font in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\microsoftyahei.ttf']:
            if os.path.exists(font):
                font_path = font
                break
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/20, color, 2)
    return img


DEFAULT_CONFIG = {
    "camera_id": 0,
    "disabled_classes": [],
    "show_trajectory": True,
    "show_prediction": True,
    "trajectory_color": [0, 0, 255],
    "prediction_color": [0, 255, 255],
    "model_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "tracker_mode": "classic",
    "max_missing": 10,
    "iou_threshold": 0.25,
}

MODEL_LIST = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt",
]


def _model_name_from_url(url):
    try:
        return os.path.basename(urlparse(url).path)
    except:
        return "model.pt"


def resolve_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def get_logs_dir(base_path):
    logs_dir = os.path.join(base_path, "logs")
    ensure_dir(logs_dir)
    return logs_dir


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"创建目录失败: {path}, 错误: {e}")


def load_settings(base_path):
    path = os.path.join(base_path, "config.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {**DEFAULT_CONFIG, **{k: v for k, v in data.items() if k in DEFAULT_CONFIG}}
    except Exception as e:
        print(f"加载配置失败: {e}")
    return dict(DEFAULT_CONFIG)


def save_settings(base_path, settings):
    path = os.path.join(base_path, "config.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: settings.get(k, v) for k, v in DEFAULT_CONFIG.items()}, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存配置失败: {e}")
    return False


def save_log_row(file_path, row):
    try:
        write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            if write_header:
                f.write("time,camera_id,track_id,class_name,confidence,x,y,w,h,fps,total_objects\n")
            f.write(f"{row.get('time','')},{row.get('camera_id','')},{row.get('track_id','')},{row.get('class_name','')},{row.get('confidence','')},{row.get('x','')},{row.get('y','')},{row.get('w','')},{row.get('h','')},{row.get('fps','')},{row.get('total_objects','')}\n")
        return True
    except Exception as e:
        print(f"日志写入失败: {file_path}, 错误: {e}")
    return False


def build_log_path(base_path):
    safe_base = re.sub(r"[^0-9A-Za-z_-]+", "_", os.path.basename(base_path)).strip("_") or "app"
    filename = f"detection_log_{safe_base}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    logs_dir = os.path.join(base_path, "logs")
    ensure_dir(logs_dir)
    return os.path.join(logs_dir, filename)


class StatsAggregator:
    def __init__(self):
        self.fps_values = deque(maxlen=30)
        self.conf_values = deque(maxlen=30)
        self.last_time = time.time()

    def update(self, detections):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_values.append(1 / dt)
        for det in detections:
            try:
                self.conf_values.append(float(det[5]))
            except:
                pass
        fps = sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0.0
        avg_conf = sum(self.conf_values) / len(self.conf_values) if self.conf_values else 0.0
        return {"fps": round(fps, 1), "avg_conf": round(avg_conf, 3), "total_objects": len(detections)}


class TrackedObject:
    def __init__(self, track_id, x, y, w, h, class_id, confidence):
        self.track_id, self.x, self.y, self.w, self.h = track_id, x, y, w, h
        self.class_id, self.confidence = class_id, confidence
        self.trajectory = [(x + w//2, y + h//2)]
        self.missing_frames, self.max_missing = 0, 10
        self.hits = 0

    def update(self, x, y, w, h, confidence, max_missing=10):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.confidence = confidence
        self.trajectory.append((x + w//2, y + h//2))
        self.missing_frames, self.max_missing = 0, max_missing
        self.hits += 1
        if len(self.trajectory) > 50:
            self.trajectory = self.trajectory[-50:]

    def predict_next_position(self):
        if len(self.trajectory) >= 2:
            x1, y1 = self.trajectory[-1]
            x2, y2 = self.trajectory[-2]
            return (x1 + (x1 - x2), y1 + (y1 - y2))
        return None

    def is_lost(self):
        return self.missing_frames >= self.max_missing


class ObjectTracker:
    def __init__(self, iou_threshold=0.25, tracker_mode="classic", max_missing=10):
        self.tracked_objects = []
        self.next_track_id = 1
        self.max_iou_distance = float(iou_threshold)
        self.mode = str(tracker_mode)
        self.max_missing = max_missing

    def _iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area, box2_area = max(0.0, w1 * h1), max(0.0, w2 * h2)
        denom = box1_area + box2_area - inter
        return inter / denom if denom > 0 else 0.0

    def update(self, detections):
        dets = sorted(detections, key=lambda d: float(d[5]), reverse=True) if self.mode == "bytetrack" else detections
        used_det, new_tracked = set(), []

        for obj in self.tracked_objects:
            best_iou, best_det = self.max_iou_distance, None
            for idx, det in enumerate(dets):
                if idx in used_det or obj.class_id != det[4]:
                    continue
                iou_val = self._iou((obj.x, obj.y, obj.w, obj.h), (det[0], det[1], det[2], det[3]))
                if iou_val > best_iou:
                    best_iou, best_det = iou_val, idx
            if best_det is not None:
                x, y, w, h, class_id, confidence = dets[best_det]
                obj.update(x, y, w, h, confidence, self.max_missing)
                used_det.add(best_det)
                new_tracked.append(obj)

        for idx, det in enumerate(dets):
            if idx not in used_det:
                x, y, w, h, class_id, confidence = det
                new_obj = TrackedObject(self.next_track_id, x, y, w, h, class_id, confidence)
                new_obj.max_missing = self.max_missing
                self.next_track_id += 1
                new_tracked.append(new_obj)

        for obj in self.tracked_objects:
            if obj not in new_tracked:
                obj.missing_frames += 1

        self.tracked_objects = [obj for obj in self.tracked_objects if not obj.is_lost()]
        self.tracked_objects.extend([obj for obj in new_tracked if obj not in self.tracked_objects])
        return self.tracked_objects


class DetectionEngine:
    def __init__(self, base_path):
        self.base_path = base_path
        self.pt_model_path = os.path.join(base_path, 'yolo26n.pt')
        self.settings = load_settings(base_path)
        self.state_lock = threading.Lock()
        
        self.use_ultralytics, self.model = False, None
        self.classes = {i: name for i, name in enumerate([
            'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ])}
        
        self._load_state()
        self._load_model()
        
        self.tracker = ObjectTracker(iou_threshold=self.iou_threshold, tracker_mode=self.tracker_mode, max_missing=self.max_missing)
        self.stats_aggregator = StatsAggregator()
        self.log_path, self.exporting = None, False

    def _load_state(self):
        s = self.settings
        self.disabled_classes = set(s.get("disabled_classes", []))
        self.show_trajectory = bool(s.get("show_trajectory", True))
        self.show_prediction = bool(s.get("show_prediction", True))
        
        tc = s.get("trajectory_color")
        self.trajectory_color = tuple(int(x) for x in tc) if isinstance(tc, (list, tuple)) and len(tc) == 3 else (0, 0, 255)
        
        pc = s.get("prediction_color")
        self.prediction_color = tuple(int(x) for x in pc) if isinstance(pc, (list, tuple)) and len(pc) == 3 else (0, 255, 255)
        
        self.tracker_mode = str(s.get("tracker_mode", "classic") or "classic")
        self.max_missing = int(s.get("max_missing", 10) or 10)
        self.iou_threshold = float(s.get("iou_threshold", 0.25) or 0.25)

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if os.path.exists(self.pt_model_path):
                self.model = YOLO(self.pt_model_path)
                self.use_ultralytics = True
                self.classes = self.model.names
                print(f"YOLO模型加载成功: {self.pt_model_path}")
            else:
                print(f"警告: 未找到模型文件: {self.pt_model_path}")
        except ImportError:
            print("未安装 ultralytics 库，请运行: pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            print(f"模型加载失败: {e}")

    def get_classes(self):
        return dict(self.classes)

    def get_settings(self):
        with self.state_lock:
            return {
                "disabled_classes": sorted(list(self.disabled_classes)),
                "show_trajectory": self.show_trajectory,
                "show_prediction": self.show_prediction,
                "trajectory_color": list(self.trajectory_color),
                "prediction_color": list(self.prediction_color),
                "tracker_mode": self.tracker_mode,
                "max_missing": self.max_missing,
                "iou_threshold": self.iou_threshold,
            }

    def get_model_path(self):
        return self.pt_model_path

    def reload_model(self):
        try:
            from ultralytics import YOLO
            for path in [self.pt_model_path, os.path.join(tempfile.gettempdir(), "cmka_model.pt")]:
                if os.path.exists(path):
                    try:
                        loaded = YOLO(path)
                        with self.state_lock:
                            self.model, self.use_ultralytics, self.classes = loaded, True, loaded.names
                        print(f"模型重载成功: {path}")
                        return True
                    except:
                        continue
        except:
            pass
        with self.state_lock:
            self.model, self.use_ultralytics = None, False
        return False

    def download_model(self, url):
        try:
            import requests
            from tqdm import tqdm
            dst = os.path.join(self.base_path, _model_name_from_url(url))
            temp_dst = dst + ".tmp"
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(temp_dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            if total > 0 and os.path.getsize(temp_dst) != total:
                raise RuntimeError("下载大小不匹配")
            os.replace(temp_dst, dst)
            self.pt_model_path = dst
            return dst
        except Exception as e:
            print(f"下载失败: {e}")
            return None

    def _draw_arrow(self, frame, start, end, color):
        sx, sy, ex, ey = start[0], start[1], end[0], end[1]
        dx, dy = ex - sx, ey - sy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 5:
            return frame
        angle = math.atan2(dy, dx)
        arrow_len = min(dist, 30)
        ex = sx + int(arrow_len * math.cos(angle))
        ey = sy + int(arrow_len * math.sin(angle))
        cv2.line(frame, (sx, sy), (ex, ey), color, 3)
        ah = 10
        cv2.line(frame, (ex, ey), (ex - int(ah * math.cos(angle + math.pi/6)), ey - int(ah * math.sin(angle + math.pi/6))), color, 3)
        cv2.line(frame, (ex, ey), (ex - int(ah * math.cos(angle - math.pi/6)), ey - int(ah * math.sin(angle - math.pi/6))), color, 3)
        return frame

    def process_frame(self, frame):
        detections = []
        with self.state_lock:
            _use = self.use_ultralytics
            _model = self.model
            _disabled = set(self.disabled_classes)
            _show_traj = self.show_trajectory
            _show_pred = self.show_prediction
            _tc = self.trajectory_color
            _pc = self.prediction_color
            _classes = dict(self.classes)
            _mode = self.tracker_mode
            _max_missing = self.max_missing
            _iou = self.iou_threshold

        if _use and _model is not None:
            try:
                results = _model(frame, conf=0.5, iou=0.45, verbose=False)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        conf, cid = float(box.conf[0]), int(box.cls[0])
                        name = _classes[cid]
                        if name not in _disabled:
                            detections.append((x1, y1, w, h, cid, conf))
            except Exception as e:
                print(f"检测错误: {e}")

        self.tracker.mode, self.tracker.max_missing, self.tracker.max_iou_distance = _mode, _max_missing, _iou
        tracked = self.tracker.update(detections)
        stats = self.stats_aggregator.update(detections)

        for obj in tracked:
            x, y, w, h = obj.x, obj.y, obj.w, obj.h
            if 0 <= obj.class_id < len(_classes):
                name = _classes[obj.class_id]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = put_chinese_text(frame, f"ID:{obj.track_id} {name}: {obj.confidence:.2f}", (x, y - 10), 14, (0, 255, 0))
                
                if _show_traj and len(obj.trajectory) > 1 and name != 'person':
                    for i in range(1, len(obj.trajectory)):
                        cv2.line(frame, obj.trajectory[i-1], obj.trajectory[i], _tc, 2)
                
                if _show_pred and name != 'person':
                    pred = obj.predict_next_position()
                    if pred:
                        frame = self._draw_arrow(frame, (x + w//2, y + h//2), (int(pred[0]), int(pred[1])), _pc)

        if self.exporting and self.log_path:
            try:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                for det in detections:
                    x, y, w, h, cid, conf = det
                    if 0 <= cid < len(_classes):
                        save_log_row(self.log_path, {
                            "time": ts, "camera_id": 0, "track_id": "", "class_name": _classes[cid],
                            "confidence": f"{conf:.3f}", "x": x, "y": y, "w": w, "h": h,
                            "fps": stats.get("fps", ""), "total_objects": stats.get("total_objects", "")
                        })
            except:
                pass

        return frame, stats

    def start_log_export(self):
        self.log_path = build_log_path(self.base_path)
        self.exporting = True
        return self.log_path

    def stop_log_export(self):
        self.exporting = False

    def set_setting(self, key, value):
        with self.state_lock:
            if key == "disabled_classes":
                self.disabled_classes = set(value)
            elif key == "show_trajectory":
                self.show_trajectory = bool(value)
            elif key == "show_prediction":
                self.show_prediction = bool(value)
            elif key == "trajectory_color":
                if isinstance(value, (list, tuple)) and len(value) == 3:
                    self.trajectory_color = tuple(int(x) for x in value)
            elif key == "prediction_color":
                if isinstance(value, (list, tuple)) and len(value) == 3:
                    self.prediction_color = tuple(int(x) for x in value)
            elif key == "tracker_mode":
                self.tracker_mode = str(value)
            elif key == "max_missing":
                self.max_missing = int(value)
            elif key == "iou_threshold":
                self.iou_threshold = float(value)

    def save_settings(self):
        settings = {
            "disabled_classes": sorted(list(self.disabled_classes)),
            "show_trajectory": self.show_trajectory,
            "show_prediction": self.show_prediction,
            "trajectory_color": list(self.trajectory_color),
            "prediction_color": list(self.prediction_color),
            "tracker_mode": self.tracker_mode,
            "max_missing": self.max_missing,
            "iou_threshold": self.iou_threshold,
            "camera_id": int(self.settings.get("camera_id", 0)),
            "model_url": self.settings.get("model_url", DEFAULT_CONFIG.get("model_url"))
        }
        return save_settings(self.base_path, settings)


# ===== WEB UI =====
def run_web(host="0.0.0.0", port=8000):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, HTMLResponse

    app = FastAPI(title="CMKA")
    engine = DetectionEngine(resolve_base_path())
    camera_running = False
    current_frame = None
    current_stats = {"fps": 0.0, "avg_conf": 0.0, "total_objects": 0}
    frame_lock = threading.Lock()

    def camera_loop(camera_id=0):
        nonlocal camera_running, current_frame, current_stats
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("摄像头打开失败")
            return
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            frame, stats = engine.process_frame(frame)
            with frame_lock:
                current_frame, current_stats = frame, stats
        cap.release()
        camera_running = False

    def gen_frames():
        while True:
            with frame_lock:
                f = current_frame
            if f is not None:
                ret, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(0.033)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CMKA 目标检测</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: sans-serif; background: #1a1a2e; color: white; padding: 20px; }
.container { max-width: 1000px; margin: 0 auto; }
h1 { text-align: center; margin-bottom: 20px; color: #00d4ff; }
.main { display: grid; grid-template-columns: 1fr 280px; gap: 20px; }
.video-box { background: #16213e; border-radius: 10px; overflow: hidden; }
.video-box img { width: 100%; display: block; }
.stats { display: flex; gap: 15px; padding: 15px; background: #0f3460; margin-top: 10px; border-radius: 10px; }
.stat { flex: 1; text-align: center; }
.stat label { display: block; font-size: 12px; color: #aaa; }
.stat value { font-size: 20px; font-weight: bold; color: #00d4ff; }
.panel { background: #16213e; border-radius: 10px; padding: 15px; }
.panel h3 { color: #00d4ff; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #333; }
.btn { width: 100%; padding: 10px; border: none; border-radius: 6px; font-size: 14px; font-weight: bold; cursor: pointer; margin-bottom: 8px; opacity: 1; }
.btn:hover { opacity: 0.8; }
.btn-start { background: #00d4ff; color: #000; }
.btn-stop { background: #e94560; color: white; }
.btn-info { background: #533483; color: white; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.toggle { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333; }
.toggle label { font-size: 13px; }
.toggle input { width: 18px; height: 18px; }
select { width: 100%; padding: 8px; border: 1px solid #333; border-radius: 4px; background: #0f3460; color: white; margin-bottom: 8px; }
.status { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-bottom: 10px; }
.status.running { background: #2ecc71; color: #000; }
.status.stopped { background: #e94560; color: white; }
</style>
</head>
<body>
<div class="container">
  <h1>🔍 CMKA 实时目标检测</h1>
  <div class="main">
    <div>
      <div class="video-box">
        <span class="status stopped" id="status">未运行</span>
        <img id="feed" src="/video_feed">
      </div>
      <div class="stats">
        <div class="stat"><label>FPS</label><value id="fps">0</value></div>
        <div class="stat"><label>目标</label><value id="obj">0</value></div>
        <div class="stat"><label>置信度</label><value id="conf">0</value></div>
      </div>
    </div>
    <div class="panel">
      <h3>控制</h3>
      <button class="btn btn-start" id="start" onclick="start()">▶ 开始检测</button>
      <button class="btn btn-stop" id="stop" onclick="stop()" disabled>⏹ 停止检测</button>
      
      <h3>设置</h3>
      <div class="toggle"><label>轨迹线</label><input type="checkbox" checked onchange="set('show_trajectory', this.checked)"></div>
      <div class="toggle"><label>预测方向</label><input type="checkbox" checked onchange="set('show_prediction', this.checked)"></div>
      <div class="toggle"><label>增强匹配</label><input type="checkbox" onchange="set('tracker_mode', this.checked?'bytetrack':'classic')"></div>
      
      <h3>模型</h3>
      <select id="model" onchange="download()">
        <option value="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt">YOLOv8n (小)</option>
        <option value="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt">YOLOv8s (标准)</option>
        <option value="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt">YOLOv8m (中)</option>
      </select>
      <button class="btn btn-info" onclick="download()">⬇ 下载模型</button>
      <button class="btn btn-info" onclick="reload()">🔄 重载模型</button>
      
      <h3>日志</h3>
      <button class="btn btn-info" id="logStart" onclick="logStart()">📝 开始导出</button>
      <button class="btn btn-info" id="logStop" onclick="logStop()" disabled>⏹ 停止导出</button>
      
      <h3>配置</h3>
      <button class="btn btn-info" onclick="save()">💾 保存配置</button>
    </div>
  </div>
</div>
<script>
let running = false, exporting = false;
async function req(url, method='POST', data=null) {
  const opts = { method, headers: data ? {'Content-Type':'application/json'} : {} };
  if (data) opts.body = JSON.stringify(data);
  return (await fetch(url, opts)).json();
}
async function start() { const res = await req('/start'); if (res.ok) { running = true; updateUI(); } }
async function stop() { const res = await req('/stop'); if (res.ok) { running = false; updateUI(); } }
async function set(key, val) { await req('/settings', 'PUT', { [key]: val }); }
async function download() { const res = await req('/download', 'POST', { url: document.getElementById('model').value }); alert(res.ok ? '下载成功' : '下载失败'); }
async function reload() { const res = await req('/reload'); alert(res.ok ? '重载成功' : '重载失败'); }
async function logStart() { const res = await req('/log_start'); if (res.ok) { exporting = true; updateUI(); } }
async function logStop() { const res = await req('/log_stop'); if (res.ok) { exporting = false; updateUI(); } }
async function save() { const res = await req('/save'); alert(res.ok ? '保存成功' : '保存失败'); }
function updateUI() {
  const s = document.getElementById('status');
  s.className = 'status ' + (running ? 'running' : 'stopped');
  s.textContent = running ? '检测中' : '未运行';
  document.getElementById('start').disabled = running;
  document.getElementById('stop').disabled = !running;
  document.getElementById('logStart').disabled = exporting;
  document.getElementById('logStop').disabled = !exporting;
}
setInterval(async () => {
  const stats = await (await fetch('/stats')).json();
  document.getElementById('fps').textContent = stats.fps;
  document.getElementById('obj').textContent = stats.total_objects;
  document.getElementById('conf').textContent = stats.avg_conf.toFixed(3);
}, 500);
</script>
</body>
</html>
    """)

    @app.get("/video_feed")
    async def video_feed():
        return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/start")
    async def start(camera_id: int = 0):
        nonlocal camera_running
        if camera_running:
            return {"ok": False}
        camera_running = True
        threading.Thread(target=camera_loop, args=(camera_id,), daemon=True).start()
        return {"ok": True}

    @app.post("/stop")
    async def stop():
        nonlocal camera_running
        camera_running = False
        return {"ok": True}

    @app.get("/stats")
    async def stats():
        return current_stats

    @app.put("/settings")
    async def settings_update(settings: dict):
        for k, v in settings.items():
            engine.set_setting(k, v)
        return {"ok": True}

    @app.post("/download")
    async def download_model(req: dict):
        return {"ok": engine.download_model(req.get("url", "")) is not None}

    @app.post("/reload")
    async def reload_model():
        return {"ok": engine.reload_model()}

    @app.post("/log_start")
    async def log_start():
        engine.start_log_export()
        return {"ok": True}

    @app.post("/log_stop")
    async def log_stop():
        engine.stop_log_export()
        return {"ok": True}

    @app.post("/save")
    async def save_settings():
        return {"ok": engine.save_settings()}

    import uvicorn
    print(f"🌐 Web UI: http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


# ===== QT UI =====
def run_qt():
    try:
        import PyQt5
        if PyQt5.__file__:
            pyqt5_path = os.path.dirname(PyQt5.__file__)
            for qt_folder in ['Qt', 'Qt5']:
                qt_plugins_path = os.path.join(pyqt5_path, qt_folder, 'plugins', 'platforms')
                if os.path.exists(qt_plugins_path):
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins_path
                    print(f"设置Qt平台插件路径: {qt_plugins_path}")
                    break
            else:
                print(f"Qt平台插件路径不存在")
        else:
            print("PyQt5路径为空")
    except ImportError:
        print("错误: 未找到PyQt5模块")
        print("请运行: pip install PyQt5")
        sys.exit(1)

    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
                                QMessageBox, QDialog, QCheckBox, QScrollArea, QGroupBox, QComboBox)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
    from PyQt5.QtGui import QImage, QPixmap, QFont

    class CameraSelectDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("摄像头选择")
            self.setFixedSize(500, 400)
            self.setStyleSheet("background-color: white;")
            
            self.available_cameras = []
            self.selected_camera = None
            
            layout = QVBoxLayout()
            
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_widget.setStyleSheet("background-color: #FFB6C1; border-radius: 15px;")
            
            title_label = QLabel("摄像头选择")
            title_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
            title_label.setStyleSheet("color: white;")
            header_layout.addWidget(title_label, Qt.AlignLeft)
            
            close_button = QPushButton("×")
            close_button.setFixedSize(40, 40)
            close_button.setStyleSheet("background-color: #FFC0CB; border-radius: 20px; color: white; font-size: 24px;")
            close_button.clicked.connect(self.reject)
            header_layout.addWidget(close_button, Qt.AlignRight)
            
            layout.addWidget(header_widget)
            
            self.camera_list = QListWidget()
            self.camera_list.setFont(QFont("Microsoft YaHei", 16))
            self.camera_list.setStyleSheet("background-color: white; border: 2px solid #B0E0E6; border-radius: 10px; margin: 20px;")
            self.camera_list.itemClicked.connect(self.on_camera_selected)
            layout.addWidget(self.camera_list)
            
            confirm_button = QPushButton("确认选择")
            confirm_button.setFont(QFont("Microsoft YaHei", 14))
            confirm_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 10px; padding: 10px; margin: 0 20px 20px 20px;")
            confirm_button.clicked.connect(self.accept)
            layout.addWidget(confirm_button)
            
            self.setLayout(layout)
            self.detect_cameras()
        
        def detect_cameras(self):
            max_cameras = 3
            self.available_cameras = []
            
            print("正在检测可用摄像头...")
            for i in range(max_cameras):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        self.available_cameras.append(i)
                        
                        display_name = f"摄像头 {i}"
                        try:
                            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            res_info = f" ({int(width)}x{int(height)})" if width > 0 and height > 0 else ""
                        except:
                            res_info = ""
                        
                        camera_info_str = f"{i}: {display_name}{res_info}"
                        item = QListWidgetItem(camera_info_str)
                        self.camera_list.addItem(item)
                        
                        cap.release()
                        print(f"发现摄像头 {i}: {display_name}{res_info}")
                except Exception as e:
                    print(f"检测摄像头 {i} 时出错: {e}")
                    continue
            
            if not self.available_cameras:
                QMessageBox.critical(self, "错误", "未找到可用摄像头")
                self.reject()
            else:
                self.camera_list.setCurrentRow(0)
                self.selected_camera = self.available_cameras[0]
        
        def on_camera_selected(self, item):
            index = self.camera_list.row(item)
            if index < len(self.available_cameras):
                self.selected_camera = self.available_cameras[index]
        
        def accept(self):
            if self.selected_camera is not None:
                super().accept()
            else:
                QMessageBox.warning(self, "警告", "请选择一个摄像头")

    class DisableClassDialog(QDialog):
        def __init__(self, engine, parent=None):
            super().__init__(parent)
            self.setWindowTitle("禁用识别类别")
            self.setFixedSize(550, 600)
            self.setStyleSheet("background-color: white;")
            self.engine = engine
            
            layout = QVBoxLayout()
            
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_widget.setStyleSheet("background-color: #FFB6C1; border-radius: 15px;")
            
            title_label = QLabel("禁用识别类别")
            title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
            title_label.setStyleSheet("color: white;")
            header_layout.addWidget(title_label, Qt.AlignLeft)
            
            close_button = QPushButton("×")
            close_button.setFixedSize(40, 40)
            close_button.setStyleSheet("background-color: #FFC0CB; border-radius: 20px; color: white; font-size: 24px;")
            close_button.clicked.connect(self.reject)
            header_layout.addWidget(close_button, Qt.AlignRight)
            
            layout.addWidget(header_widget)
            
            info_label = QLabel("勾选的类别将不会被识别显示")
            info_label.setFont(QFont("Microsoft YaHei", 12))
            info_label.setStyleSheet("color: #9370DB; padding: 15px;")
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)
            
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("""
                QScrollArea { 
                    border: 2px solid #FFB6C1; 
                    border-radius: 10px;
                    background-color: #FFF0F5;
                }
            """)
            
            scroll_content = QWidget()
            self.scroll_layout = QVBoxLayout(scroll_content)
            
            self.checkboxes = []
            classes = self.engine.get_classes()
            
            for class_id, class_name in sorted(classes.items()):
                checkbox = QCheckBox(f"{class_id}. {class_name}")
                checkbox.setFont(QFont("Microsoft YaHei", 11))
                checkbox.setStyleSheet("padding: 8px; color: #696969;")
                if class_name in self.engine.get_settings()["disabled_classes"]:
                    checkbox.setChecked(True)
                self.checkboxes.append((class_name, checkbox))
                self.scroll_layout.addWidget(checkbox)
            
            scroll.setWidget(scroll_content)
            layout.addWidget(scroll)
            
            button_layout = QHBoxLayout()
            
            clear_button = QPushButton("清除全部")
            clear_button.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            clear_button.setStyleSheet("background-color: #87CEEB; color: white; border-radius: 10px; padding: 12px;")
            clear_button.clicked.connect(self.clear_all)
            button_layout.addWidget(clear_button)
            
            save_button = QPushButton("保存")
            save_button.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            save_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 10px; padding: 12px 30px;")
            save_button.clicked.connect(self.save_and_close)
            button_layout.addWidget(save_button)
            
            layout.addLayout(button_layout)
            
            self.setLayout(layout)
        
        def clear_all(self):
            for class_name, checkbox in self.checkboxes:
                checkbox.setChecked(False)
        
        def save_and_close(self):
            disabled = []
            for class_name, checkbox in self.checkboxes:
                if checkbox.isChecked():
                    disabled.append(class_name)
            
            self.engine.set_setting("disabled_classes", disabled)
            QMessageBox.information(self, "成功", f"已禁用 {len(disabled)} 个类别")
            self.accept()

    class TrackSettingsDialog(QDialog):
        def __init__(self, engine, parent=None):
            super().__init__(parent)
            self.setWindowTitle("轨迹设置")
            self.setMinimumSize(520, 520)
            self.setStyleSheet("background-color: white;")
            self.engine = engine
            
            layout = QVBoxLayout()
            
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_widget.setStyleSheet("background-color: #20B2AA; border-radius: 15px;")
            
            title_label = QLabel("轨迹设置")
            title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
            title_label.setStyleSheet("color: white;")
            header_layout.addWidget(title_label, Qt.AlignLeft)
            
            close_button = QPushButton("×")
            close_button.setFixedSize(40, 40)
            close_button.setStyleSheet("background-color: #48D1CC; color: white; border-radius: 15px; font-size: 24px;")
            close_button.clicked.connect(self.reject)
            header_layout.addWidget(close_button, Qt.AlignRight)
            
            layout.addWidget(header_widget)
            
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("""
                QScrollArea { 
                    border: 2px solid #20B2AA; 
                    border-radius: 10px;
                    background-color: #E0FFFF;
                }
            """)
            
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            
            settings = self.engine.get_settings()
            
            self.trajectory_checkbox = QCheckBox("显示轨迹线")
            self.trajectory_checkbox.setFont(QFont("Microsoft YaHei", 14))
            self.trajectory_checkbox.setStyleSheet("padding: 15px;")
            self.trajectory_checkbox.setChecked(settings["show_trajectory"])
            self.trajectory_checkbox.stateChanged.connect(self.on_trajectory_toggle)
            scroll_layout.addWidget(self.trajectory_checkbox)
            
            self.prediction_checkbox = QCheckBox("显示预测方向")
            self.prediction_checkbox.setFont(QFont("Microsoft YaHei", 14))
            self.prediction_checkbox.setStyleSheet("padding: 15px;")
            self.prediction_checkbox.setChecked(settings["show_prediction"])
            self.prediction_checkbox.stateChanged.connect(self.on_prediction_toggle)
            scroll_layout.addWidget(self.prediction_checkbox)
            
            self.tracker_mode_checkbox = QCheckBox("使用增强匹配模式")
            self.tracker_mode_checkbox.setFont(QFont("Microsoft YaHei", 14))
            self.tracker_mode_checkbox.setStyleSheet("padding: 15px;")
            self.tracker_mode_checkbox.setChecked(settings["tracker_mode"] == "bytetrack")
            self.tracker_mode_checkbox.stateChanged.connect(self.on_tracker_mode_toggle)
            scroll_layout.addWidget(self.tracker_mode_checkbox)
            
            trajectory_color_group = QGroupBox("轨迹线颜色")
            trajectory_color_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            trajectory_color_group.setStyleSheet("margin: 10px; padding: 15px;")
            color_layout1 = QVBoxLayout(trajectory_color_group)
            
            preset_layout1 = QHBoxLayout()
            preset_layout1.setSpacing(10)
            
            self.trajectory_color_buttons = []
            trajectory_colors = [
                ("红色", (0, 0, 255)),
                ("蓝色", (255, 0, 0)),
                ("绿色", (0, 255, 0)),
                ("黄色", (0, 255, 255)),
                ("紫色", (128, 0, 128)),
                ("白色", (255, 255, 255)),
            ]
            
            for name, color in trajectory_colors:
                btn = QPushButton(name)
                btn.setFixedSize(70, 40)
                btn.setFont(QFont("Microsoft YaHei", 11))
                btn.setStyleSheet(f"background-color: rgb({color[2]}, {color[1]}, {color[0]}); color: {'black' if sum(color) > 380 else 'white'}; border-radius: 5px; border: 2px solid #ccc;")
                btn.clicked.connect(lambda checked, c=color: self.set_trajectory_color(c))
                self.trajectory_color_buttons.append((btn, color))
                preset_layout1.addWidget(btn)
            
            color_layout1.addLayout(preset_layout1)
            
            custom_color_layout1 = QHBoxLayout()
            custom_color_label1 = QLabel("自定义颜色:")
            custom_color_label1.setFont(QFont("Microsoft YaHei", 11))
            custom_color_layout1.addWidget(custom_color_label1)
            
            self.custom_trajectory_button = QPushButton()
            self.custom_trajectory_button.setFixedSize(40, 40)
            self.update_custom_button_color(self.custom_trajectory_button, tuple(settings["trajectory_color"]))
            self.custom_trajectory_button.clicked.connect(self.choose_custom_trajectory_color)
            custom_color_layout1.addWidget(self.custom_trajectory_button)
            
            custom_color_layout1.addStretch(1)
            color_layout1.addLayout(custom_color_layout1)
            
            scroll_layout.addWidget(trajectory_color_group)
            
            prediction_color_group = QGroupBox("预测方向颜色")
            prediction_color_group.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            prediction_color_group.setStyleSheet("margin: 10px; padding: 15px;")
            color_layout2 = QVBoxLayout(prediction_color_group)
            
            preset_layout2 = QHBoxLayout()
            preset_layout2.setSpacing(10)
            
            self.prediction_color_buttons = []
            prediction_colors = [
                ("黄色", (0, 255, 255)),
                ("红色", (0, 0, 255)),
                ("蓝色", (255, 0, 0)),
                ("绿色", (0, 255, 0)),
                ("紫色", (128, 0, 128)),
                ("白色", (255, 255, 255)),
            ]
            
            for name, color in prediction_colors:
                btn = QPushButton(name)
                btn.setFixedSize(70, 40)
                btn.setFont(QFont("Microsoft YaHei", 11))
                btn.setStyleSheet(f"background-color: rgb({color[2]}, {color[1]}, {color[0]}); color: {'black' if sum(color) > 380 else 'white'}; border-radius: 5px; border: 2px solid #ccc;")
                btn.clicked.connect(lambda checked, c=color: self.set_prediction_color(c))
                self.prediction_color_buttons.append((btn, color))
                preset_layout2.addWidget(btn)
            
            color_layout2.addLayout(preset_layout2)
            
            custom_color_layout2 = QHBoxLayout()
            custom_color_label2 = QLabel("自定义颜色:")
            custom_color_label2.setFont(QFont("Microsoft YaHei", 11))
            custom_color_layout2.addWidget(custom_color_label2)
            
            self.custom_prediction_button = QPushButton()
            self.custom_prediction_button.setFixedSize(40, 40)
            self.update_custom_button_color(self.custom_prediction_button, tuple(settings["prediction_color"]))
            self.custom_prediction_button.clicked.connect(self.choose_custom_prediction_color)
            custom_color_layout2.addWidget(self.custom_prediction_button)
            
            custom_color_layout2.addStretch(1)
            color_layout2.addLayout(custom_color_layout2)
            
            scroll_layout.addWidget(prediction_color_group)
            
            scroll.setWidget(scroll_content)
            layout.addWidget(scroll)
            
            info_label = QLabel("提示：点击颜色按钮后立即生效，无需保存")
            info_label.setFont(QFont("Microsoft YaHei", 10))
            info_label.setStyleSheet("color: #666; text-align: center; padding: 10px;")
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)
            
            self.setLayout(layout)
        
        def update_custom_button_color(self, button, color):
            r, g, b = color
            button.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border-radius: 5px; border: 2px solid #20B2AA;")
        
        def on_trajectory_toggle(self, state):
            self.engine.set_setting("show_trajectory", state == Qt.Checked)
        
        def on_prediction_toggle(self, state):
            self.engine.set_setting("show_prediction", state == Qt.Checked)

        def on_tracker_mode_toggle(self, state):
            self.engine.set_setting("tracker_mode", "bytetrack" if state == Qt.Checked else "classic")
        
        def set_trajectory_color(self, color):
            self.engine.set_setting("trajectory_color", color)
            self.update_custom_button_color(self.custom_trajectory_button, color)
        
        def set_prediction_color(self, color):
            self.engine.set_setting("prediction_color", color)
            self.update_custom_button_color(self.custom_prediction_button, color)
        
        def choose_custom_trajectory_color(self):
            try:
                from PyQt5.QtWidgets import QColorDialog
                from PyQt5.QtGui import QColor
                
                settings = self.engine.get_settings()
                r, g, b = settings["trajectory_color"]
                qcolor = QColor(r, g, b)
                color = QColorDialog.getColor(qcolor, self, "选择轨迹线颜色")
                
                if color.isValid():
                    new_color = (color.red(), color.green(), color.blue())
                    self.set_trajectory_color(new_color)
            except Exception as e:
                print(f"选择轨迹线颜色错误: {e}")
        
        def choose_custom_prediction_color(self):
            try:
                from PyQt5.QtWidgets import QColorDialog
                from PyQt5.QtGui import QColor
                
                settings = self.engine.get_settings()
                r, g, b = settings["prediction_color"]
                qcolor = QColor(r, g, b)
                color = QColorDialog.getColor(qcolor, self, "选择预测方向颜色")
                
                if color.isValid():
                    new_color = (color.red(), color.green(), color.blue())
                    self.set_prediction_color(new_color)
            except Exception as e:
                print(f"选择预测方向颜色错误: {e}")

    class ModelManageDialog(QDialog):
        def __init__(self, engine, parent=None):
            super().__init__(parent)
            self.setWindowTitle("模型管理")
            self.setFixedSize(620, 360)
            self.setStyleSheet("background-color: white;")
            self.engine = engine

            layout = QVBoxLayout()

            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_widget.setStyleSheet("background-color: #20B2AA; border-radius: 15px;")

            title_label = QLabel("模型管理")
            title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
            title_label.setStyleSheet("color: white;")
            header_layout.addWidget(title_label, Qt.AlignLeft)

            close_button = QPushButton("×")
            close_button.setFixedSize(40, 40)
            close_button.setStyleSheet("background-color: #48D1CC; color: white; border-radius: 15px; font-size: 24px;")
            close_button.clicked.connect(self.reject)
            header_layout.addWidget(close_button, Qt.AlignRight)

            layout.addWidget(header_widget)

            info_label = QLabel("选择模型后可直接下载，也可删除当前模型后重载其他模型。")
            info_label.setFont(QFont("Microsoft YaHei", 12))
            info_label.setStyleSheet("color: #404040; padding: 15px;")
            info_label.setWordWrap(True)
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)

            select_layout = QHBoxLayout()
            select_label = QLabel("选择模型：")
            select_label.setFont(QFont("Microsoft YaHei", 12))
            select_layout.addWidget(select_label)

            self.model_combo = QComboBox()
            self.model_combo.setFont(QFont("Microsoft YaHei", 12))
            self.model_combo.setMinimumWidth(340)
            for url in MODEL_LIST:
                self.model_combo.addItem(_model_name_from_url(url), url)
            
            current_url = self.engine.settings.get("model_url", "")
            index = self.model_combo.findData(current_url)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            select_layout.addWidget(self.model_combo)
            layout.addLayout(select_layout)

            self.status_label = QLabel("")
            self.status_label.setFont(QFont("Microsoft YaHei", 11))
            self.status_label.setStyleSheet("color: #666; padding: 0 15px 15px 15px;")
            self.status_label.setWordWrap(True)
            layout.addWidget(self.status_label)

            button_layout = QHBoxLayout()
            button_layout.setContentsMargins(15, 0, 15, 15)
            button_layout.setSpacing(12)

            delete_button = QPushButton("删除旧模型")
            delete_button.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            delete_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 10px; padding: 10px;")
            delete_button.clicked.connect(self.delete_model)
            button_layout.addWidget(delete_button)

            download_button = QPushButton("下载模型")
            download_button.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            download_button.setStyleSheet("background-color: #4682B4; color: white; border-radius: 10px; padding: 10px;")
            download_button.clicked.connect(self.download_model_action)
            button_layout.addWidget(download_button)

            reload_button = QPushButton("重载模型")
            reload_button.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            reload_button.setStyleSheet("background-color: #9370DB; color: white; border-radius: 10px; padding: 10px;")
            reload_button.clicked.connect(self.reload_model_action)
            button_layout.addWidget(reload_button)

            layout.addLayout(button_layout)
            self.setLayout(layout)
            self.update_status()

        def update_status(self):
            exists = os.path.exists(self.engine.get_model_path())
            text = f"模型路径：{self.engine.get_model_path()}\n状态：{'存在' if exists else '未找到'}"
            if exists:
                try:
                    size = os.path.getsize(self.engine.get_model_path())
                    text += f"，大小：{size/1024/1024:.2f} MB"
                except Exception:
                    pass
            self.status_label.setText(text)

        def delete_model(self):
            path = self.engine.get_model_path()
            if os.path.exists(path):
                try:
                    os.remove(path)
                    QMessageBox.information(self, "成功", "已删除旧模型文件。")
                    self.update_status()
                except Exception as e:
                    QMessageBox.critical(self, "错误", f"删除失败：{e}")
            else:
                QMessageBox.information(self, "提示", "当前没有可删除的模型文件。")

        def download_model_action(self):
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                url = self.model_combo.currentData()
                path = self.engine.download_model(url)
                if path:
                    QMessageBox.information(self, "成功", f"模型已下载：{path}")
                    self.update_status()
                else:
                    QMessageBox.warning(self, "失败", "模型下载失败，未保存文件。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"下载异常：{e}")
            finally:
                QApplication.restoreOverrideCursor()

        def reload_model_action(self):
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                ok = self.engine.reload_model()
                if ok:
                    QMessageBox.information(self, "成功", "模型已重载。")
                else:
                    QMessageBox.warning(self, "失败", "模型重载失败，仍使用旧状态。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"重载异常：{e}")
            finally:
                QApplication.restoreOverrideCursor()

    class LogExportDialog(QDialog):
        def __init__(self, engine, parent=None):
            super().__init__(parent)
            self.setWindowTitle("日志导出")
            self.setFixedSize(520, 260)
            self.setStyleSheet("background-color: white;")
            self.engine = engine
            self.exporting = False

            layout = QVBoxLayout()

            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_widget.setStyleSheet("background-color: #20B2AA; border-radius: 15px;")

            title_label = QLabel("日志导出")
            title_label.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
            title_label.setStyleSheet("color: white;")
            header_layout.addWidget(title_label, Qt.AlignLeft)

            close_button = QPushButton("×")
            close_button.setFixedSize(40, 40)
            close_button.setStyleSheet("background-color: #48D1CC; color: white; border-radius: 15px; font-size: 24px;")
            close_button.clicked.connect(self.reject)
            header_layout.addWidget(close_button, Qt.AlignRight)

            layout.addWidget(header_widget)

            self.path_label = QLabel(f"日志文件：{self.engine.get_model_path()}")
            self.path_label.setFont(QFont("Microsoft YaHei", 11))
            self.path_label.setStyleSheet("color: #404040; padding: 15px;")
            self.path_label.setWordWrap(True)
            layout.addWidget(self.path_label)

            self.status_label = QLabel("状态：未开始")
            self.status_label.setFont(QFont("Microsoft YaHei", 11))
            self.status_label.setStyleSheet("color: #666; padding: 0 15px 15px 15px;")
            layout.addWidget(self.status_label)

            button_layout = QHBoxLayout()
            button_layout.setContentsMargins(15, 0, 15, 15)
            button_layout.setSpacing(12)

            self.start_button = QPushButton("开始导出")
            self.start_button.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            self.start_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 10px; padding: 10px;")
            self.start_button.clicked.connect(self.start_export)
            button_layout.addWidget(self.start_button)

            self.stop_button = QPushButton("停止导出")
            self.stop_button.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            self.stop_button.setStyleSheet("background-color: #4682B4; color: white; border-radius: 10px; padding: 10px;")
            self.stop_button.clicked.connect(self.stop_export)
            self.stop_button.setEnabled(False)
            button_layout.addWidget(self.stop_button)

            open_button = QPushButton("打开日志目录")
            open_button.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            open_button.setStyleSheet("background-color: #9370DB; color: white; border-radius: 10px; padding: 10px;")
            open_button.clicked.connect(self.open_logs_dir)
            button_layout.addWidget(open_button)

            layout.addLayout(button_layout)
            self.setLayout(layout)

        def start_export(self):
            self.exporting = True
            log_path = self.engine.start_log_export()
            self.path_label.setText(f"日志文件：{log_path}")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("状态：导出中...")

        def stop_export(self):
            self.exporting = False
            self.engine.stop_log_export()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("状态：已暂停")

        def open_logs_dir(self):
            try:
                path = get_logs_dir(self.engine.base_path)
                if sys.platform == "win32":
                    os.startfile(path)
                else:
                    QMessageBox.information(self, "路径", path)
            except Exception as e:
                QMessageBox.warning(self, "失败", str(e))

    class VideoThread(QThread):
        change_pixmap_signal = pyqtSignal(QImage)
        stats_signal = pyqtSignal(dict)
        
        def __init__(self, camera_id, engine):
            super().__init__()
            self.camera_id = camera_id
            self.running = True
            self.engine = engine
        
        def run(self):
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(f"无法打开摄像头 {self.camera_id}")
                return
                
            fail_count = 0
            max_fail = 30
            
            while self.running:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        fail_count += 1
                        print(f"摄像头读取失败 ({fail_count}/{max_fail})")
                        if fail_count >= max_fail:
                            print("摄像头连续失败，停止检测")
                            break
                        time.sleep(0.1)
                        continue
                    fail_count = 0
                    
                    frame, stats = self.engine.process_frame(frame)
                    self.stats_signal.emit(stats)
                    
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(convert_to_Qt_format.copy())
                except Exception as e:
                    print(f"视频处理错误: {e}")
                    continue
            
            cap.release()
            print("摄像头已释放")
        
        def stop(self):
            self.running = False
            self.wait()

    class MainWindow(QMainWindow):
        def __init__(self, camera_id, engine):
            super().__init__()
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setGeometry(100, 100, 980, 740)
            
            self.engine = engine
            
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            
            main_layout = QVBoxLayout(main_widget)
            main_layout.setContentsMargins(0, 0, 0, 0)
            
            title_bar = QWidget()
            title_bar.setFixedHeight(60)
            title_bar.setStyleSheet("background-color: #FFB6C1; border-top-left-radius: 15px; border-top-right-radius: 15px;")
            title_layout = QHBoxLayout(title_bar)
            title_layout.setContentsMargins(20, 0, 20, 0)
            
            title_label = QLabel("实时目标检测")
            title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
            title_label.setStyleSheet("color: white;")
            title_layout.addWidget(title_label, 1, Qt.AlignLeft | Qt.AlignVCenter)
            
            min_button = QPushButton("_")
            min_button.setFixedSize(30, 30)
            min_button.setStyleSheet("background-color: #FFC0CB; color: white; border-radius: 15px; font-size: 16px;")
            min_button.clicked.connect(self.showMinimized)
            title_layout.addWidget(min_button, Qt.AlignRight | Qt.AlignVCenter)
            
            self.max_button = QPushButton("□")
            self.max_button.setFixedSize(30, 30)
            self.max_button.setStyleSheet("background-color: #FFC0CB; color: white; border-radius: 15px; font-size: 16px;")
            self.max_button.clicked.connect(self.toggle_maximize)
            title_layout.addWidget(self.max_button, Qt.AlignRight | Qt.AlignVCenter)
            
            close_button = QPushButton("×")
            close_button.setFixedSize(30, 30)
            close_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 15px; font-size: 16px;")
            close_button.clicked.connect(self.close)
            title_layout.addWidget(close_button, Qt.AlignRight | Qt.AlignVCenter)
            
            main_layout.addWidget(title_bar)
            
            content_widget = QWidget()
            content_widget.setStyleSheet("background-color: white; border-bottom-left-radius: 15px; border-bottom-right-radius: 15px; padding: 20px;")
            content_layout = QVBoxLayout(content_widget)
            
            self.video_label = QLabel()
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setStyleSheet("border: 2px solid #B0E0E6; border-radius: 10px; min-height: 520px;")
            self.video_label.setScaledContents(True)
            content_layout.addWidget(self.video_label)
            
            self.stats_label = QLabel("FPS: 0.0 | 目标: 0 | 平均置信度: 0.000")
            self.stats_label.setFont(QFont("Microsoft YaHei", 12))
            self.stats_label.setStyleSheet("color: #404040; padding: 4px 8px;")
            self.stats_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            content_layout.addWidget(self.stats_label)
            
            control_layout = QHBoxLayout()
            control_layout.setContentsMargins(0, 10, 0, 0)
            control_layout.setSpacing(12)
            
            self.start_button = QPushButton("开始检测")
            self.start_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            self.start_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 10px; padding: 10px 18px;")
            self.start_button.clicked.connect(self.start_detection)
            control_layout.addWidget(self.start_button)
            
            self.stop_button = QPushButton("停止检测")
            self.stop_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            self.stop_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 10px; padding: 10px 18px;")
            self.stop_button.clicked.connect(self.stop_detection)
            self.stop_button.setEnabled(False)
            control_layout.addWidget(self.stop_button)
            
            self.switch_button = QPushButton("切换摄像头")
            self.switch_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            self.switch_button.setStyleSheet("background-color: #4682B4; color: white; border-radius: 10px; padding: 10px 18px;")
            self.switch_button.clicked.connect(self.switch_camera)
            control_layout.addWidget(self.switch_button)
            
            self.disable_button = QPushButton("禁用类别")
            self.disable_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            self.disable_button.setStyleSheet("background-color: #9370DB; color: white; border-radius: 10px; padding: 10px 18px;")
            self.disable_button.clicked.connect(self.open_disable_dialog)
            control_layout.addWidget(self.disable_button)
            
            self.track_button = QPushButton("轨迹设置")
            self.track_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            self.track_button.setStyleSheet("background-color: #20B2AA; color: white; border-radius: 10px; padding: 10px 18px;")
            self.track_button.clicked.connect(self.open_track_settings)
            control_layout.addWidget(self.track_button)
            
            self.model_button = QPushButton("模型管理")
            self.model_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            self.model_button.setStyleSheet("background-color: #4682B4; color: white; border-radius: 10px; padding: 10px 18px;")
            self.model_button.clicked.connect(self.open_model_manage)
            control_layout.addWidget(self.model_button)
            
            content_layout.addLayout(control_layout)
            
            control_layout2 = QHBoxLayout()
            control_layout2.setContentsMargins(0, 8, 0, 0)
            control_layout2.setSpacing(12)
            
            self.log_button = QPushButton("日志导出")
            self.log_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            self.log_button.setStyleSheet("background-color: #9370DB; color: white; border-radius: 10px; padding: 10px 18px;")
            self.log_button.clicked.connect(self.open_log_export)
            control_layout2.addWidget(self.log_button)
            
            self.save_config_button = QPushButton("保存配置")
            self.save_config_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            self.save_config_button.setStyleSheet("background-color: #20B2AA; color: white; border-radius: 10px; padding: 10px 18px;")
            self.save_config_button.clicked.connect(self.save_current_config)
            control_layout2.addWidget(self.save_config_button)
            
            content_layout.addLayout(control_layout2)
            main_layout.addWidget(content_widget)
            
            self.thread = None
            self.camera_id = camera_id
            self.log_export_dialog = None
            
            self.dragging = False
            self.drag_start_pos = None
            
            title_bar.installEventFilter(self)
        
        def eventFilter(self, obj, event):
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.dragging = True
                    self.drag_start_pos = event.globalPos() - self.frameGeometry().topLeft()
                    return True
            elif event.type() == QEvent.MouseMove:
                if self.dragging:
                    self.move(event.globalPos() - self.drag_start_pos)
                    return True
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.dragging = False
                    return True
            return super().eventFilter(obj, event)
        
        def toggle_maximize(self):
            if self.isMaximized():
                self.showNormal()
                self.max_button.setText("□")
            else:
                self.showMaximized()
                self.max_button.setText("▢")
        
        def start_detection(self):
            if self.log_export_dialog is None:
                self.log_export_dialog = LogExportDialog(self.engine, self)
            self.thread = VideoThread(self.camera_id, self.engine)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.stats_signal.connect(self.update_stats)
            self.thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        
        def stop_detection(self):
            if self.thread:
                self.thread.stop()
                self.thread = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.stats_label.setText("FPS: 0.0 | 目标: 0 | 平均置信度: 0.000")
        
        def update_image(self, qt_image):
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
        def update_stats(self, stats):
            try:
                self.stats_label.setText(
                    f"FPS: {stats.get('fps', 0.0)} | 目标: {stats.get('total_objects', 0)} | 平均置信度: {stats.get('avg_conf', 0.0):.3f}"
                )
            except Exception:
                pass
        
        def switch_camera(self):
            self.stop_detection()
            
            camera_dialog = CameraSelectDialog(self)
            if camera_dialog.exec_():
                new_camera_id = camera_dialog.selected_camera
                if new_camera_id != self.camera_id:
                    self.camera_id = new_camera_id
                    QMessageBox.information(self, "切换成功", f"已切换到摄像头 {self.camera_id}")
                else:
                    QMessageBox.information(self, "提示", "当前已选择该摄像头")
        
        def open_disable_dialog(self):
            disable_dialog = DisableClassDialog(self.engine, self)
            disable_dialog.exec_()
        
        def open_track_settings(self):
            track_dialog = TrackSettingsDialog(self.engine, self)
            track_dialog.exec_()
        
        def open_model_manage(self):
            model_dialog = ModelManageDialog(self.engine, self)
            model_dialog.exec_()
        
        def open_log_export(self):
            if self.log_export_dialog is None:
                self.log_export_dialog = LogExportDialog(self.engine, self)
            self.log_export_dialog.show()
        
        def save_current_config(self):
            ok = self.engine.save_settings()
            if ok:
                QMessageBox.information(self, "成功", "当前配置已保存。")
            else:
                QMessageBox.warning(self, "失败", "保存配置失败，请检查目录权限。")
        
        def closeEvent(self, event):
            self.stop_detection()
            event.accept()

    base_path = resolve_base_path()
    engine = DetectionEngine(base_path)
    
    app = QApplication(sys.argv)
    
    camera_dialog = CameraSelectDialog()
    selected_camera = int(engine.settings.get("camera_id", 0))
    if camera_dialog.exec_():
        selected_camera = camera_dialog.selected_camera
    else:
        print("未选择摄像头，程序退出")
        return
    
    window = MainWindow(selected_camera, engine)
    window.show()
    
    sys.exit(app.exec_())


def show_mode_selection():
    try:
        import PyQt5
        if PyQt5.__file__:
            pyqt5_path = os.path.dirname(PyQt5.__file__)
            for qt_folder in ['Qt', 'Qt5']:
                qt_plugins_path = os.path.join(pyqt5_path, qt_folder, 'plugins', 'platforms')
                if os.path.exists(qt_plugins_path):
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins_path
                    break
    except ImportError:
        pass

    from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
                                QLabel, QPushButton, QWidget)
    from PyQt5.QtGui import QFont
    from PyQt5.QtCore import Qt

    class ModeSelectDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("CMKA 目标检测 - 模式选择")
            self.setFixedSize(520, 420)
            self.setStyleSheet("background-color: white;")
            self.selected_mode = None

            layout = QVBoxLayout()
            layout.setSpacing(15)
            layout.setContentsMargins(20, 20, 20, 20)

            title = QLabel("🔍 CMKA 实时目标检测")
            title.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
            title.setStyleSheet("color: white; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;")
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)

            info_label = QLabel("请选择运行模式：")
            info_label.setFont(QFont("Microsoft YaHei", 16))
            info_label.setStyleSheet("color: #404040;")
            info_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(info_label)

            button_layout = QHBoxLayout()
            button_layout.setSpacing(25)
            button_layout.setContentsMargins(50, 0, 50, 0)

            qt_button = QPushButton("🖥️\n桌面模式")
            qt_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            qt_button.setStyleSheet("""
                QPushButton {
                    background: linear-gradient(135deg, #FF69B4 0%, #FFB6C1 100%);
                    color: white;
                    border-radius: 12px;
                    padding: 25px 20px;
                    border: none;
                    min-width: 160px;
                    min-height: 100px;
                }
                QPushButton:hover {
                    background: linear-gradient(135deg, #FF1493 0%, #FF69B4 100%);
                }
            """)
            qt_button.clicked.connect(lambda: self.select_mode("qt"))
            button_layout.addWidget(qt_button)

            web_button = QPushButton("🌐\n网页模式")
            web_button.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
            web_button.setStyleSheet("""
                QPushButton {
                    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
                    color: white;
                    border-radius: 12px;
                    padding: 25px 20px;
                    border: none;
                    min-width: 160px;
                    min-height: 100px;
                }
                QPushButton:hover {
                    background: linear-gradient(135deg, #00b8e6 0%, #0080aa 100%);
                }
            """)
            web_button.clicked.connect(lambda: self.select_mode("web"))
            button_layout.addWidget(web_button)

            layout.addLayout(button_layout)

            desc_layout = QHBoxLayout()
            desc_layout.setSpacing(25)
            desc_layout.setContentsMargins(50, 10, 50, 10)

            qt_desc = QLabel("• 本地桌面应用<br>• 界面美观流畅<br>• 需要安装 PyQt5")
            qt_desc.setFont(QFont("Microsoft YaHei", 11))
            qt_desc.setStyleSheet("color: #666;")
            qt_desc.setAlignment(Qt.AlignCenter)
            qt_desc.setWordWrap(True)
            qt_desc.setTextFormat(Qt.RichText)
            desc_layout.addWidget(qt_desc)

            web_desc = QLabel("• 浏览器访问<br>• 支持远程访问<br>• 需要安装 fastapi")
            web_desc.setFont(QFont("Microsoft YaHei", 11))
            web_desc.setStyleSheet("color: #666;")
            web_desc.setAlignment(Qt.AlignCenter)
            web_desc.setWordWrap(True)
            web_desc.setTextFormat(Qt.RichText)
            desc_layout.addWidget(web_desc)

            layout.addLayout(desc_layout)

            self.setLayout(layout)

        def select_mode(self, mode):
            self.selected_mode = mode
            self.accept()

    app = QApplication(sys.argv)
    dialog = ModeSelectDialog()
    if dialog.exec_():
        return dialog.selected_mode
    return None


_web_server_thread = None
_web_server_process = None


def run_web_in_thread(host, port):
    global _web_server_thread
    import threading
    _web_server_thread = threading.Thread(target=run_web, args=(host, port), daemon=True)
    _web_server_thread.start()


def stop_web_server():
    global _web_server_process
    if _web_server_process:
        pid = _web_server_process.pid
        print(f"正在停止Web服务器进程 (PID: {pid})...")
        
        try:
            _web_server_process.terminate()
            _web_server_process.wait(timeout=3)
            print(f"进程 {pid} 已终止")
        except:
            try:
                _web_server_process.kill()
                _web_server_process.wait(timeout=3)
                print(f"进程 {pid} 已强制杀死")
            except:
                print(f"进程 {pid} 终止失败")
        
        import subprocess as sp
        try:
            result = sp.run(["taskkill", "/F", "/T", "/PID", str(pid)], 
                            capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"taskkill 成功终止进程 {pid}")
            else:
                print(f"taskkill 失败: {result.stderr.decode('gbk', errors='ignore')}")
        except Exception as e:
            print(f"taskkill 执行异常: {e}")
        
        import os
        try:
            os.kill(pid, 9)
            print(f"os.kill 成功终止进程 {pid}")
        except:
            pass
        
        _web_server_process = None
        print("Web服务器已完全停止")


def main():
    global _web_server_process
    parser = argparse.ArgumentParser(description="CMKA 目标检测")
    parser.add_argument("--mode", choices=["qt", "web"], help="运行模式")
    parser.add_argument("--port", type=int, default=8000, help="Web端口")
    parser.add_argument("--host", default="0.0.0.0", help="Web监听地址")
    args = parser.parse_args()

    mode = args.mode
    is_cli = mode is not None

    if mode is None:
        mode = show_mode_selection()
        if mode is None:
            print("用户取消选择，程序退出")
            return

    if mode == "web":
        if is_cli:
            run_web(host=args.host, port=args.port)
        else:
            import subprocess
            _web_server_process = subprocess.Popen([sys.executable, __file__, "--mode", "web", "--port", str(args.port), "--host", args.host])
            print(f"Web模式已启动，访问: http://localhost:{args.port}")
            print("按 Enter 键停止Web服务器并返回模式选择...")
            input()
            stop_web_server()
            print("Web服务器已停止")
            main()
    else:
        run_qt()


if __name__ == "__main__":
    main()
"""目标追踪模块"""

from typing import List, Optional, Set, Tuple


class TrackedObject:
    """单个被追踪目标的运行时状态。"""

    MAX_TRAJECTORY_LENGTH: int = 50

    def __init__(
        self,
        track_id: int,
        x: int, y: int, w: int, h: int,
        class_id: int,
        confidence: float,
        max_missing: int = 10,
    ) -> None:
        self.track_id = track_id
        self.x, self.y, self.w, self.h = x, y, w, h
        self.class_id = class_id
        self.confidence = confidence
        self.trajectory: List[Tuple[int, int]] = [(x + w // 2, y + h // 2)]
        self.missing_frames: int = 0
        self.max_missing: int = max_missing
        self.hits: int = 0

    def update(self, x: int, y: int, w: int, h: int, confidence: float, max_missing: int = 10) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h
        self.confidence = confidence
        self.trajectory.append((x + w // 2, y + h // 2))
        self.missing_frames = 0
        self.max_missing = max_missing
        self.hits += 1
        if len(self.trajectory) > self.MAX_TRAJECTORY_LENGTH:
            self.trajectory = self.trajectory[-self.MAX_TRAJECTORY_LENGTH:]

    def predict_next_position(self) -> Optional[Tuple[int, int]]:
        if len(self.trajectory) >= 2:
            x1, y1 = self.trajectory[-1]
            x2, y2 = self.trajectory[-2]
            return (x1 + (x1 - x2), y1 + (y1 - y2))
        return None

    def is_lost(self) -> bool:
        return self.missing_frames >= self.max_missing


class ObjectTracker:
    """基于 IoU 的简单多目标追踪器。"""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        tracker_mode: str = "classic",
        max_missing: int = 10,
    ) -> None:
        self.tracked_objects: List[TrackedObject] = []
        self.next_track_id: int = 1
        self.max_iou_distance: float = float(iou_threshold)
        self.mode: str = str(tracker_mode)
        self.max_missing: int = max_missing

    @staticmethod
    def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        denom = max(0.0, w1 * h1) + max(0.0, w2 * h2) - inter
        return inter / denom if denom > 0 else 0.0

    def update(self, detections: List[Tuple]) -> List[TrackedObject]:
        dets = (
            sorted(detections, key=lambda d: float(d[5]), reverse=True)
            if self.mode == "bytetrack"
            else detections
        )

        used_det: Set[int] = set()
        new_tracked: List[TrackedObject] = []

        # 匹配已有目标
        for obj in self.tracked_objects:
            best_iou, best_det = self.max_iou_distance, None
            for idx, det in enumerate(dets):
                if idx in used_det or obj.class_id != det[4]:
                    continue
                iou_val = self._iou(
                    (obj.x, obj.y, obj.w, obj.h),
                    (det[0], det[1], det[2], det[3]),
                )
                if iou_val > best_iou:
                    best_iou, best_det = iou_val, idx
            if best_det is not None:
                x, y, w, h, class_id, confidence = dets[best_det]
                obj.update(x, y, w, h, confidence, self.max_missing)
                used_det.add(best_det)
                new_tracked.append(obj)

        # 未匹配检测 → 新目标
        for idx, det in enumerate(dets):
            if idx not in used_det:
                x, y, w, h, class_id, confidence = det
                new_obj = TrackedObject(self.next_track_id, x, y, w, h, class_id, confidence)
                new_obj.max_missing = self.max_missing
                self.next_track_id += 1
                new_tracked.append(new_obj)

        # 未匹配旧目标 → missing++
        for obj in self.tracked_objects:
            if obj not in new_tracked:
                obj.missing_frames += 1

        # 清理丢失目标，合并新追踪结果
        surviving = [obj for obj in self.tracked_objects if not obj.is_lost()]
        new_ids = {obj.track_id for obj in surviving}
        for obj in new_tracked:
            if obj.track_id not in new_ids:
                surviving.append(obj)
                new_ids.add(obj.track_id)
        self.tracked_objects = surviving
        return self.tracked_objects

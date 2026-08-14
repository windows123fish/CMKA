"""统计聚合模块"""

import time
from collections import deque
from typing import Any, Deque, Dict, List, Tuple


class StatsAggregator:
    """滑动窗口统计 FPS 和平均置信度。"""

    def __init__(self, window_size: int = 30) -> None:
        self.fps_values: Deque[float] = deque(maxlen=window_size)
        self.conf_values: Deque[float] = deque(maxlen=window_size)
        self.last_time: float = time.time()

    def update(self, detections: List[Tuple]) -> Dict[str, Any]:
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_values.append(1.0 / dt)
        for det in detections:
            try:
                self.conf_values.append(float(det[5]))
            except (IndexError, ValueError):
                pass

        fps = sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0.0
        avg_conf = sum(self.conf_values) / len(self.conf_values) if self.conf_values else 0.0
        return {"帧率": round(fps, 1), "平均置信度": round(avg_conf, 3), "目标总数": len(detections)}

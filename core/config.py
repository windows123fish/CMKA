"""配置和常量定义"""

import os
from typing import Any, Dict, List

# COCO 类别映射
COCO_CLASSES: Dict[int, str] = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'sofa', 58: 'pottedplant', 59: 'bed',
    60: 'diningtable', 61: 'toilet', 62: 'tvmonitor', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
}

# 默认配置
DEFAULT_CONFIG: Dict[str, Any] = {
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
    "conf_threshold": 0.5,
    "show_boxes": True,
}

# 可用模型列表
MODEL_LIST: List[str] = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt",
]

# UI 常量
UI_FONT_FAMILY: str = "Microsoft YaHei"

UI_COLORS: Dict[str, str] = {
    "primary": "#FF69B4",
    "primary_hover": "#FF1493",
    "secondary": "#4682B4",
    "accent": "#9370DB",
    "teal": "#20B2AA",
    "teal_light": "#48D1CC",
    "header_bg": "#FFB6C1",
    "header_close": "#FFC0CB",
    "border_light": "#B0E0E6",
    "scroll_bg": "#FFF0F5",
    "scroll_border": "#FFB6C1",
    "info_text": "#666",
    "body_text": "#404040",
    "label_text": "#696969",
    "white": "#FFFFFF",
    "black": "#000000",
}

UI_SIZES: Dict[str, int] = {
    "dialog_title_font": 20,
    "dialog_title_bold_font": 24,
    "body_font": 12,
    "body_bold_font": 14,
    "small_font": 11,
    "close_btn": 40,
    "header_radius": 15,
    "btn_radius": 10,
    "window_width": 980,
    "window_height": 740,
    "title_bar_height": 60,
}

# 中文字体路径
_CHINESE_FONT_PATHS: List[str] = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\microsoftyahei.ttf",
] if os.name == "nt" else []

# 摄像头检测上限
MAX_CAMERA_SCAN: int = 3
# 摄像头连续读取失败阈值
CAMERA_MAX_FAIL: int = 30
# 日志 CSV 列定义
LOG_COLUMNS: List[str] = [
    "时间", "摄像头编号", "追踪ID", "类别名称", "置信度",
    "坐标X", "坐标Y", "框宽W", "框高H", "帧率", "目标总数",
]

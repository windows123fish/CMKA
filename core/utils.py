"""工具函数：日志、图像处理、配置读写"""

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _pil_available = True
except ImportError:
    _pil_available = False

from .config import (
    _CHINESE_FONT_PATHS,
    DEFAULT_CONFIG,
    LOG_COLUMNS,
)

# 日志配置
logger = logging.getLogger("CMKA")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def put_chinese_text(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 20,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """在图像上绘制中文文本，PIL 不可用时回退到 cv2。"""
    if not _pil_available:
        import cv2
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 20, color, 2)
        return img

    import cv2
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path: Optional[str] = None
    for fp in _CHINESE_FONT_PATHS:
        if os.path.exists(fp):
            font_path = fp
            break

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 20, color, 2)
    return img


def _model_name_from_url(url: str) -> str:
    """从 URL 中提取模型文件名。"""
    try:
        from urllib.parse import urlparse
        return os.path.basename(urlparse(url).path)
    except Exception:
        return "model.pt"


def resolve_base_path() -> str:
    """获取程序所在目录（兼容 PyInstaller 打包）。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # 从 core/ 子目录回到项目根目录
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> None:
    """确保目录存在，不存在则创建。"""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        logger.error("创建目录失败: %s, 错误: %s", path, exc)


def get_logs_dir(base_path: str) -> str:
    """获取日志目录路径，不存在时自动创建。"""
    logs_dir = os.path.join(base_path, "logs")
    ensure_dir(logs_dir)
    return logs_dir


def load_settings(base_path: str) -> Dict[str, Any]:
    """从 config.json 加载配置，缺失字段用默认值填充。"""
    path = os.path.join(base_path, "config.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {**DEFAULT_CONFIG, **{k: v for k, v in data.items() if k in DEFAULT_CONFIG}}
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("加载配置失败: %s", exc)
    return dict(DEFAULT_CONFIG)


def save_settings(base_path: str, settings: Dict[str, Any]) -> bool:
    """将配置写入 config.json，返回是否成功。"""
    path = os.path.join(base_path, "config.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: settings.get(k, v) for k, v in DEFAULT_CONFIG.items()}, f, ensure_ascii=False, indent=2)
        return True
    except OSError as exc:
        logger.error("保存配置失败: %s", exc)
        return False


def save_log_row(file_path: str, **row_kwargs: Any) -> bool:
    """向 CSV 日志文件追加一行数据。"""
    try:
        write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            if write_header:
                f.write(",".join(LOG_COLUMNS) + "\n")
            f.write(",".join(str(row_kwargs.get(c, "")) for c in LOG_COLUMNS) + "\n")
        return True
    except OSError as exc:
        logger.error("日志写入失败: %s, 错误: %s", file_path, exc)
        return False


def get_available_cameras(max_index: int = 10) -> List[int]:
    """Scan for available cameras up to max_index and return their IDs."""
    available: List[int] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


def build_log_path(base_path: str) -> str:
    """根据时间戳生成日志文件路径。"""
    safe_base = re.sub(r"[^0-9A-Za-z_-]+", "_", os.path.basename(base_path)).strip("_") or "app"
    filename = f"检测日志_{safe_base}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    logs_dir = os.path.join(base_path, "logs")
    ensure_dir(logs_dir)
    return os.path.join(logs_dir, filename)

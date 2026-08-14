"""CMKA 实时目标检测系统 - 主入口

提供桌面(PyQt5)和Web(FastAPI)两种运行模式。
"""

import argparse
import sys

from core.detector import DetectionEngine
from core.utils import load_settings, logger, resolve_base_path, save_settings


def _configure_qt_platform_plugins() -> None:
    """配置 Qt 平台插件路径"""
    try:
        import PyQt5
        if PyQt5.__file__:
            import os
            pyqt5_path = os.path.dirname(PyQt5.__file__)
            for qt_folder in ("Qt", "Qt5"):
                qt_plugins_path = os.path.join(pyqt5_path, qt_folder, "plugins", "platforms")
                if os.path.exists(qt_plugins_path):
                    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugins_path
                    break
    except ImportError:
        pass


def run_qt() -> None:
    """启动桌面模式"""
    from PyQt5.QtWidgets import QApplication, QMessageBox

    from ui.dialogs import CameraSelectDialog
    from ui.main_window import MainWindow

    _configure_qt_platform_plugins()

    base_path = resolve_base_path()
    engine = DetectionEngine(base_path)

    app = QApplication(sys.argv)
    if not engine.use_ultralytics:
        QMessageBox.critical(None, "模型加载失败", "YOLO模型加载失败，请检查模型文件是否存在。")
        sys.exit(1)

    camera_dialog = CameraSelectDialog()
    selected_camera = int(engine.settings.get("camera_id", 0))
    if camera_dialog.exec_():
        selected_camera = camera_dialog.selected_camera
    else:
        logger.info("未选择摄像头，程序退出")
        sys.exit(0)

    window = MainWindow(selected_camera, engine)
    window.show()
    sys.exit(app.exec_())


def main() -> None:
    """主入口"""
    parser = argparse.ArgumentParser(description="CMKA 目标检测")
    parser.add_argument("--mode", choices=["qt", "web"], nargs="?", const="qt", help="运行模式（已废弃，仅保留向后兼容）")
    parser.add_argument("--port", type=int, default=8000, help="已废弃：Web 端口")
    parser.add_argument("--host", default="127.0.0.1", help="已废弃：Web 监听地址")
    parser.add_argument("--camera", type=int, default=None, help="默认摄像头编号")
    args = parser.parse_args()

    if args.camera is not None:
        base_path = resolve_base_path()
        settings = load_settings(base_path)
        settings["camera_id"] = args.camera
        save_settings(base_path, settings)

    logger.info("已切换到桌面模式。")
    run_qt()


if __name__ == "__main__":
    main()

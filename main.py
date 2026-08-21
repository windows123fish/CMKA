import argparse
import os
import sys
from core.detector import DetectionEngine
from core.utils import load_settings, logger, resolve_base_path, save_settings


def _configure_qt_platform_plugins() -> None:
    try:
        import PyQt5
        if PyQt5.__file__:
            pyqt5_path = os.path.dirname(PyQt5.__file__)
            for qt_folder in ("Qt", "Qt5"):
                qt_plugins_path = os.path.join(pyqt5_path, qt_folder, "plugins", "platforms")
                if os.path.exists(qt_plugins_path):
                    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugins_path
                    break
    except ImportError:
        pass


def Windows_123_fish() -> None:
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
    parsed = argparse.Namespace()
    parsed.mode = "desktop"
    parsed.config = None
    parsed.source = None

    base_path = resolve_base_path()
    settings = load_settings(base_path)
    engine = DetectionEngine(base_path)

    Windows_123_fish()


if __name__ == "__main__":
    main()
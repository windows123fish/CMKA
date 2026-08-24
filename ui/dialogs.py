import os
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import UI_COLORS, UI_FONT_FAMILY, UI_SIZES
from core.utils import get_logs_dir, logger


def _update_custom_button(button: QPushButton, color: Tuple[int, int, int]) -> None:
    if len(color) == 3:
        b, g, r = int(color[0]), int(color[1]), int(color[2])
    else:
        r, g, b = 0, 0, 0
    button.setStyleSheet(
        f"background-color: rgb({r}, {g}, {b}); "
        f"border-radius: 5px; border: 2px solid {UI_COLORS['teal']};"
    )


class BaseDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)

    def _setup_dialog(self, title: str, width: int = 400, height: int = 300) -> None:
        container = QWidget(self)
        container.setGeometry(0, 0, width, height)
        container.setStyleSheet(
            f"background-color: {UI_COLORS['panel_bg']}; "
            f"border-radius: {UI_SIZES['dialog_radius']}px;"
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header.setFixedHeight(40)
        header.setStyleSheet(f"background-color: {UI_COLORS['header_bg']}; border-radius: {UI_SIZES['dialog_radius']}px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(15, 0, 10, 0)

        title_label = QLabel(title)
        title_label.setFont(QFont(UI_FONT_FAMILY, 14, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        header_layout.addWidget(title_label, 1, Qt.AlignLeft | Qt.AlignVCenter)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet(f"background-color: {UI_COLORS['primary']}; color: white; border-radius: 15px; font-size: 16px;")
        close_btn.clicked.connect(self.reject)
        header_layout.addWidget(close_btn, Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(header)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(12)
        layout.addWidget(self.content_widget, 1)

        self.dragging = False
        self.drag_start_pos = None
        header.installEventFilter(self)

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if event.type() == 6 and event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.globalPos() - self.frameGeometry().topLeft()
            return True
        if event.type() == 8 and self.dragging:
            self.move(event.globalPos() - self.drag_start_pos)
            return True
        if event.type() == 7:
            self.dragging = False
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event: Any) -> None:
        if hasattr(self, "content_widget"):
            self.content_widget.setGeometry(0, 40, self.width(), self.height() - 40)
        super().resizeEvent(event)


class CameraSelectDialog(BaseDialog):
    def __init__(self, parent: Optional[QWidget] = None, current_camera: int = 0) -> None:
        super().__init__(parent)
        self.selected_camera = current_camera
        self._setup_dialog("选择摄像头", 430, 240)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout()
        layout.setSpacing(12)

        self.camera_box = QComboBox()
        self.camera_box.setFont(QFont(UI_FONT_FAMILY, 12))
        self.camera_box.setStyleSheet(
            f"padding: 6px; border: 1px solid {UI_COLORS['border_light']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px;"
        )
        available_cameras = self._scan_available_cameras()
        if available_cameras:
            for cam_id in available_cameras:
                self.camera_box.addItem(f"摄像头 {cam_id}", cam_id)
            default_index = self.camera_box.findData(self.selected_camera)
            if default_index >= 0:
                self.camera_box.setCurrentIndex(default_index)
        else:
            self.camera_box.addItem("未检测到可用摄像头", -1)
            self.camera_box.setEnabled(False)
        layout.addRow("可用摄像头:", self.camera_box)

        self.content_layout.addLayout(layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch(1)

        ok_btn = QPushButton("确定")
        ok_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        ok_btn.setMinimumWidth(100)
        ok_btn.setStyleSheet(
            f"background-color: {UI_COLORS['primary']}; color: white; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setStyleSheet(
            f"background-color: {UI_COLORS['border_light']}; color: {UI_COLORS['body_text']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.content_layout.addLayout(btn_layout)

    def _scan_available_cameras(self) -> List[int]:
        try:
            import cv2
            available: List[int] = []
            for idx in range(10):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    available.append(idx)
                    cap.release()
            return available
        except Exception as exc:
            logger.error("扫描摄像头失败: %s", exc)
            return []

    def _on_ok(self) -> None:
        cam_data = self.camera_box.currentData()
        if cam_data is not None and cam_data >= 0:
            self.selected_camera = int(cam_data)
        self.accept()


class DisableClassDialog(BaseDialog):
    def __init__(self, engine: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.class_names: Dict[int, str] = engine.get_classes()
        self.disabled_classes = set(engine.settings.get("disabled_classes", []))
        self._setup_dialog("禁用类别", 420, 500)
        self._build_ui()

    def _build_ui(self) -> None:
        self.list_widget = QListWidget()
        self.list_widget.setFont(QFont(UI_FONT_FAMILY, 11))
        self.list_widget.setStyleSheet(
            f"border: 1px solid {UI_COLORS['border_light']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 6px;"
        )

        for class_id, name in sorted(self.class_names.items()):
            item = QListWidgetItem(f"{name} (ID: {class_id})")
            item.setData(Qt.UserRole, class_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if class_id in self.disabled_classes else Qt.Checked)
            self.list_widget.addItem(item)

        self.content_layout.addWidget(self.list_widget)

        self.hint_label = QLabel("取消勾选 = 禁用该类别的检测")
        self.hint_label.setFont(QFont(UI_FONT_FAMILY, 10))
        self.hint_label.setStyleSheet(f"color: {UI_COLORS['label_text']};")
        self.content_layout.addWidget(self.hint_label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch(1)

        ok_btn = QPushButton("保存")
        ok_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        ok_btn.setMinimumWidth(100)
        ok_btn.setStyleSheet(
            f"background-color: {UI_COLORS['primary']}; color: white; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        ok_btn.clicked.connect(self._save)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setStyleSheet(
            f"background-color: {UI_COLORS['border_light']}; color: {UI_COLORS['body_text']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.content_layout.addLayout(btn_layout)

    def _save(self) -> None:
        disabled = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Unchecked:
                disabled.append(item.data(Qt.UserRole))
        self.engine.settings["disabled_classes"] = disabled
        self.engine._apply_disabled_classes()
        self.accept()


class TrackSettingsDialog(BaseDialog):
    def __init__(self, engine: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._setup_dialog("轨迹设置", 420, 280)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout()
        layout.setSpacing(12)

        self.mode_box = QSpinBox()
        self.mode_box.setRange(3, 30)
        self.mode_box.setValue(int(self.engine.settings.get("max_missing", 10) or 10))
        self.mode_box.setFont(QFont(UI_FONT_FAMILY, 12))
        self.mode_box.setStyleSheet(
            f"padding: 6px; border: 1px solid {UI_COLORS['border_light']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px;"
        )
        layout.addRow("最大丢失帧数:", self.mode_box)

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(10, 50)
        self.iou_slider.setValue(int(float(self.engine.settings.get("iou_threshold", 0.25) or 0.25) * 100))
        self.iou_label = QLabel(f"{self.iou_slider.value() / 100:.2f}")
        self.iou_label.setFont(QFont(UI_FONT_FAMILY, 11))
        self.iou_label.setStyleSheet(f"color: {UI_COLORS['body_text']};")
        self.iou_slider.valueChanged.connect(lambda v: self.iou_label.setText(f"{v / 100:.2f}"))
        layout.addRow("追踪 IoU 阈值:", self.iou_slider)
        layout.addRow("", self.iou_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(int(float(self.engine.settings.get("conf_threshold", 0.5) or 0.5) * 100))
        self.conf_label = QLabel(f"{self.conf_slider.value() / 100:.2f}")
        self.conf_label.setFont(QFont(UI_FONT_FAMILY, 11))
        self.conf_label.setStyleSheet(f"color: {UI_COLORS['body_text']};")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v / 100:.2f}"))
        layout.addRow("检测置信度:", self.conf_slider)
        layout.addRow("", self.conf_label)

        self.content_layout.addLayout(layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch(1)

        ok_btn = QPushButton("保存")
        ok_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        ok_btn.setMinimumWidth(100)
        ok_btn.setStyleSheet(
            f"background-color: {UI_COLORS['primary']}; color: white; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        ok_btn.clicked.connect(self._save)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setStyleSheet(
            f"background-color: {UI_COLORS['border_light']}; color: {UI_COLORS['body_text']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.content_layout.addLayout(btn_layout)

    def _save(self) -> None:
        self.engine.set_setting("max_missing", int(self.mode_box.value()))
        self.engine.set_setting("iou_threshold", self.iou_slider.value() / 100)
        self.engine.set_setting("conf_threshold", self.conf_slider.value() / 100)
        self.accept()


class ModelManageDialog(BaseDialog):
    def __init__(self, engine: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._setup_dialog("模型管理", 520, 360)
        self._build_ui()

    def _build_ui(self) -> None:
        info_label = QLabel(f"当前模型：{getattr(self.engine, 'model_name', '未知')}")
        info_label.setFont(QFont(UI_FONT_FAMILY, 11))
        info_label.setStyleSheet(f"color: {UI_COLORS['body_text']};")
        self.content_layout.addWidget(info_label)

        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(self.engine.model_path or "")
        self.path_edit.setFont(QFont(UI_FONT_FAMILY, 11))
        self.path_edit.setReadOnly(True)
        self.path_edit.setStyleSheet(
            f"padding: 6px; border: 1px solid {UI_COLORS['border_light']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px;"
        )
        path_layout.addWidget(self.path_edit, 1)

        browse_btn = QPushButton("浏览")
        browse_btn.setFont(QFont(UI_FONT_FAMILY, 11))
        browse_btn.setStyleSheet(
            f"background-color: {UI_COLORS['secondary']}; color: white; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 6px 12px;"
        )
        browse_btn.clicked.connect(self._browse_model)
        path_layout.addWidget(browse_btn)
        self.content_layout.addLayout(path_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch(1)

        ok_btn = QPushButton("加载")
        ok_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        ok_btn.setMinimumWidth(100)
        ok_btn.setStyleSheet(
            f"background-color: {UI_COLORS['primary']}; color: white; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        ok_btn.clicked.connect(self._load_model)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("关闭")
        cancel_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setStyleSheet(
            f"background-color: {UI_COLORS['border_light']}; color: {UI_COLORS['body_text']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.content_layout.addLayout(btn_layout)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "YOLO 模型 (*.pt *.onnx)")
        if path:
            self.path_edit.setText(path)

    def _load_model(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择模型文件")
            return
        try:
            self.engine.load_model(path)
            self.engine.model_path = path
            QMessageBox.information(self, "成功", "模型加载成功")
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "失败", f"模型加载失败：{exc}")


class LogExportDialog(BaseDialog):
    def __init__(self, engine: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._setup_dialog("日志导出", 520, 380)
        self._build_ui()

    def _build_ui(self) -> None:
        desc = QLabel("导出当前检测日志到 CSV 文件：")
        desc.setFont(QFont(UI_FONT_FAMILY, 11))
        desc.setStyleSheet(f"color: {UI_COLORS['body_text']};")
        self.content_layout.addWidget(desc)

        logs_dir = get_logs_dir(self.engine.base_path)
        files = []
        if os.path.isdir(logs_dir):
            files = [
                f for f in os.listdir(logs_dir) if f.endswith(".csv") or f.endswith(".json")
            ]

        if not files:
            tip = QLabel("暂无日志文件")
            tip.setFont(QFont(UI_FONT_FAMILY, 11))
            tip.setStyleSheet(f"color: {UI_COLORS['label_text']};")
            self.content_layout.addWidget(tip)
        else:
            self.file_list = QListWidget()
            self.file_list.setFont(QFont(UI_FONT_FAMILY, 11))
            self.file_list.setStyleSheet(
                f"border: 1px solid {UI_COLORS['border_light']}; "
                f"border-radius: {UI_SIZES['btn_radius']}px; padding: 6px;"
            )
            for name in sorted(files)[-20:]:
                self.file_list.addItem(name)
            self.content_layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch(1)

        export_btn = QPushButton("导出")
        export_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        export_btn.setMinimumWidth(100)
        export_btn.setStyleSheet(
            f"background-color: {UI_COLORS['primary']}; color: white; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        export_btn.clicked.connect(self._export)
        btn_layout.addWidget(export_btn)

        close_btn = QPushButton("关闭")
        close_btn.setFont(QFont(UI_FONT_FAMILY, 12))
        close_btn.setMinimumWidth(100)
        close_btn.setStyleSheet(
            f"background-color: {UI_COLORS['border_light']}; color: {UI_COLORS['body_text']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 8px;"
        )
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        self.content_layout.addLayout(btn_layout)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", os.path.join(get_logs_dir(self.engine.base_path), "export.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            import shutil
            src = os.path.join(get_logs_dir(self.engine.base_path), "log.csv")
            if os.path.exists(src):
                shutil.copy2(src, path)
                QMessageBox.information(self, "成功", f"已导出到：{path}")
            else:
                QMessageBox.warning(self, "提示", "当前没有可导出的日志")
        except Exception as exc:
            QMessageBox.critical(self, "失败", f"导出失败：{exc}")
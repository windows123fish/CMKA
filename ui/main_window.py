"""主窗口模块"""

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

from PyQt5.QtCore import QEvent, QObject, QThread, Qt, pyqtSignal
 from PyQt5.QtGui import QFont, QImage
 from PyQt5.QtWidgets import (
  QCheckBox, QGridLayout, QHBoxLayout, QLabel,
  QMainWindow, QMessageBox, QPushButton, QScrollArea,
  QSlider, QVBoxLayout, QWidget,
 )

import cv2
import numpy as np

from core.config import UI_COLORS, UI_FONT_FAMILY, UI_SIZES
 from core.detector import DetectionEngine
 from core.utils import ensure_dir, get_logs_dir
 from ui.dialogs import (
  CameraSelectDialog,
  DisableClassDialog,
  LogExportDialog,
  ModelManageDialog,
  TrackSettingsDialog,
 )


class VideoThread(QThread):
    """视频处理线程"""

    change_pixmap_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(dict)

    def __init__(self, camera_id: int, engine: DetectionEngine, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.engine = engine

    def run(self) -> None:
        def on_frame(frame: np.ndarray, stats: Dict[str, Any]) -> None:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image.copy())
            self.stats_signal.emit(stats)

        self.engine.run(camera_id=self.camera_id, on_frame=on_frame)

    def stop(self) -> None:
        self.engine.stop()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self, camera_id: int, engine: DetectionEngine) -> None:
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, UI_SIZES["window_width"] + 220, UI_SIZES["window_height"])
        self.setMinimumSize(1020, 600)
        self.setMaximumSize(1920, 1080)

        self.engine = engine
        self.camera_id = camera_id
        self.thread: Optional[VideoThread] = None
        self.log_export_dialog: Optional[LogExportDialog] = None
        self.dragging: bool = False
        self.drag_start_pos: Any = None

        self.class_counts: Dict[str, int] = defaultdict(int)
        self.total_detections: int = 0
        self.class_names: Dict[int, str] = engine.get_classes()

        self._build_ui()

    def _build_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self._build_title_bar())
        left_layout.addWidget(self._build_content())

        right_widget = self._build_stats_panel()

        main_layout.addWidget(left_widget, 4)
        main_layout.addWidget(right_widget, 1)

    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(UI_SIZES["title_bar_height"])
        bar.setStyleSheet(
            f"background-color: {UI_COLORS['header_bg']}; "
            f"border-top-left-radius: {UI_SIZES['header_radius']}px; "
            f"border-top-right-radius: {UI_SIZES['header_radius']}px;"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("实时目标检测")
        title.setFont(QFont(UI_FONT_FAMILY, 18, QFont.Bold))
        title.setStyleSheet("color: white;")
        layout.addWidget(title, 1, Qt.AlignLeft | Qt.AlignVCenter)

        for text, slot, bg in [
            ("_", self.showMinimized, UI_COLORS["header_close"]),
            ("□", self.toggle_maximize, UI_COLORS["header_close"]),
            ("×", self.close, UI_COLORS["primary"]),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(30, 30)
            btn.setStyleSheet(f"background-color: {bg}; color: white; border-radius: 15px; font-size: 16px;")
            btn.clicked.connect(slot)
            layout.addWidget(btn, Qt.AlignRight | Qt.AlignVCenter)
            if text == "□":
                self.max_button = btn

        bar.installEventFilter(self)
        return bar

    def _build_content(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(
            f"background-color: white; "
            f"border-bottom-left-radius: {UI_SIZES['header_radius']}px; "
            f"border-bottom-right-radius: {UI_SIZES['header_radius']}px; padding: 15px;"
        )
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.addWidget(self._build_video_section(), 1)
        layout.addLayout(self._build_control_row_1())
        layout.addLayout(self._build_control_row_2())
        return widget

    def _build_video_section(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(
            f"border: 2px solid {UI_COLORS['border_light']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; background-color: #f8f8f8;"
        )
        v_layout = QVBoxLayout(container)
        v_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        self.video_label.setScaledContents(True)
        v_layout.addWidget(self.video_label, 1)

        self.stats_label = QLabel("帧率: 0.0 | 目标: 0 | 平均置信度: 0.000")
        self.stats_label.setFont(QFont(UI_FONT_FAMILY, 12, QFont.Bold))
        self.stats_label.setStyleSheet("color: white; background-color: rgba(0,0,0,0.6); padding: 6px 15px;")
        self.stats_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        v_layout.addWidget(self.stats_label)
        return container

    def _build_control_row_1(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(12)

        self.start_button = self._make_ctrl_btn("开始检测", UI_COLORS["primary"], self._start_detection)
        layout.addWidget(self.start_button)

        self.stop_button = self._make_ctrl_btn("停止检测", UI_COLORS["primary"], self._stop_detection)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        self.switch_button = self._make_ctrl_btn("切换摄像头", UI_COLORS["secondary"], self._switch_camera)
        layout.addWidget(self.switch_button)

        self.disable_button = self._make_ctrl_btn("禁用类别", UI_COLORS["accent"], self._open_disable_dialog)
        layout.addWidget(self.disable_button)

        self.track_button = self._make_ctrl_btn("轨迹设置", UI_COLORS["teal"], self._open_track_settings)
        layout.addWidget(self.track_button)

        self.model_button = self._make_ctrl_btn("模型管理", UI_COLORS["secondary"], self._open_model_manage)
        layout.addWidget(self.model_button)

        return layout

    def _build_control_row_2(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)

        self.log_button = self._make_ctrl_btn("日志导出", UI_COLORS["accent"], self._open_log_export)
        layout.addWidget(self.log_button)

        self.save_config_button = self._make_ctrl_btn("保存配置", UI_COLORS["teal"], self._save_current_config)
        layout.addWidget(self.save_config_button)

        return layout

    @staticmethod
    def _make_ctrl_btn(text: str, bg: str, callback: Callable) -> QPushButton:
        btn = QPushButton(text)
        btn.setFont(QFont(UI_FONT_FAMILY, 13, QFont.Bold))
        btn.setStyleSheet(
            f"background-color: {bg}; color: white; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 10px 18px;"
        )
        btn.clicked.connect(callback)
        return btn

    def _build_stats_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(
            f"background-color: {UI_COLORS['scroll_bg']}; "
            f"border-left: 1px solid {UI_COLORS['border_light']};"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("实时统计")
        title.setFont(QFont(UI_FONT_FAMILY, 14, QFont.Bold))
        title.setStyleSheet(f"color: {UI_COLORS['body_text']};")
        layout.addWidget(title)

        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)
        self.show_all_checkbox = QCheckBox("显示全部类别")
        self.show_all_checkbox.setChecked(True)
        self.show_all_checkbox.setFont(QFont(UI_FONT_FAMILY, 11))
        self.show_all_checkbox.stateChanged.connect(self._refresh_stats)
        filter_layout.addWidget(self.show_all_checkbox)
        filter_layout.addStretch(1)
        layout.addLayout(filter_layout)

        self.total_label = QLabel("总检测数：0")
        self.total_label.setFont(QFont(UI_FONT_FAMILY, 12, QFont.Bold))
        self.total_label.setStyleSheet(f"color: {UI_COLORS['body_text']};")
        layout.addWidget(self.total_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: 1px solid {UI_COLORS['border_light']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; background-color: white; }}"
        )
        scroll_content = QWidget()
        self.class_grid = QGridLayout(scroll_content)
        self.class_grid.setContentsMargins(8, 8, 8, 8)
        self.class_grid.setSpacing(6)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        clear_btn = QPushButton("清空统计")
        clear_btn.setFont(QFont(UI_FONT_FAMILY, 11))
        clear_btn.setStyleSheet(
            f"background-color: {UI_COLORS['border_light']}; color: {UI_COLORS['body_text']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 6px;"
        )
        clear_btn.clicked.connect(self._clear_stats)
        btn_layout.addWidget(clear_btn)
        layout.addLayout(btn_layout)

        return panel

    def _update_stats_grid(self, stats: Dict[str, Any]) -> None:
        class_names = list(self.class_counts.keys())
        show_all = self.show_all_checkbox.isChecked()
        display_names = class_names if show_all else class_names[:8]

        while self.class_grid.count():
            item = self.class_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for row, name in enumerate(display_names):
            count = int(self.class_counts.get(name, 0))
            count_label = QLabel(f"{count}")
            count_label.setFont(QFont(UI_FONT_FAMILY, 11, QFont.Bold))
            count_label.setStyleSheet(f"color: {UI_COLORS['body_text']}; min-width: 30px;")
            count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.class_grid.addWidget(count_label, row, 0)

            name_label = QLabel(name)
            name_label.setFont(QFont(UI_FONT_FAMILY, 11))
            name_label.setStyleSheet(f"color: {UI_COLORS['label_text']};")
            self.class_grid.addWidget(name_label, row, 1)

    def _refresh_stats(self) -> None:
        self._update_stats_grid({})

    def _clear_stats(self) -> None:
        self.class_counts.clear()
        self.total_detections = 0
        self._refresh_stats()

    def eventFilter(self, obj: Any, event: QEvent) -> bool:
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.globalPos() - self.frameGeometry().topLeft()
            return True
        if event.type() == QEvent.MouseMove and self.dragging:
            self.move(event.globalPos() - self.drag_start_pos)
            return True
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            self.dragging = False
            return True
        return super().eventFilter(obj, event)

    def toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
            self.max_button.setText("□")
        else:
            self.showMaximized()
            self.max_button.setText("▢")

    def _start_detection(self) -> None:
        if self.thread and self.thread.isRunning():
            return
        if self.log_export_dialog is None:
            self.log_export_dialog = LogExportDialog(self.engine, self)
        self.thread = VideoThread(self.camera_id, self.engine)
        self.thread.change_pixmap_signal.connect(self._update_image)
        self.thread.stats_signal.connect(self._update_stats)
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _stop_detection(self) -> None:
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
        self.thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stats_label.setText("帧率: 0.0 | 目标: 0 | 平均置信度: 0.000")
        self.class_counts.clear()
        self.total_detections = 0
        self.total_label.setText("总检测数：0")
        self._refresh_stats()

    def _update_image(self, qt_image: QImage) -> None:
        from PyQt5.QtGui import QPixmap
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def _update_stats(self, stats: Dict[str, Any]) -> None:
        try:
            self.stats_label.setText(
                f"帧率: {stats.get('帧率', 0.0)} | 目标: {stats.get('目标总数', 0)} "
                f"| 平均置信度: {stats.get('平均置信度', 0.0):.3f}"
            )
        except Exception:
            pass

        class_counts = stats.get("类别统计", {}) if isinstance(stats, dict) else {}
 if isinstance(class_counts, dict):
  self.class_counts = defaultdict(int, {k: int(v) for k, v in class_counts.items()})
 self.total_detections = int(stats.get("目标总数", 0))
 self.total_label.setText(f"总检测数：{self.total_detections}")
 self._update_stats_grid(stats)

    def _switch_camera(self) -> None:
        self._stop_detection()
        dialog = CameraSelectDialog(self)
        if dialog.exec_():
            new_id = dialog.selected_camera
            if new_id != self.camera_id:
                self.camera_id = new_id
                QMessageBox.information(self, "切换成功", f"已切换到摄像头 {self.camera_id}")
            else:
                QMessageBox.information(self, "提示", "当前已选择该摄像头")

    def _open_disable_dialog(self) -> None:
        DisableClassDialog(self.engine, self).exec_()

    def _open_track_settings(self) -> None:
        TrackSettingsDialog(self.engine, self).exec_()

    def _open_model_manage(self) -> None:
        ModelManageDialog(self.engine, self).exec_()

    def _open_log_export(self) -> None:
        if self.log_export_dialog is None:
            self.log_export_dialog = LogExportDialog(self.engine, self)
        self.log_export_dialog.show()

    def _save_current_config(self) -> None:
        if self.engine.save_settings():
            QMessageBox.information(self, "成功", "当前配置已保存。")
        else:
            QMessageBox.warning(self, "失败", "保存配置失败，请检查目录权限。")

    def closeEvent(self, event: QEvent) -> None:
        self._stop_detection()
        event.accept()

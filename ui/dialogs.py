"""UI 对话框模块"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.config import MAX_CAMERA_SCAN, MODEL_LIST, UI_COLORS, UI_FONT_FAMILY, UI_SIZES
from core.detector import DetectionEngine
from core.utils import get_logs_dir, logger


class BaseDialog(QDialog):
    """对话框基类"""

    def __init__(self, title: str, width: int, height: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: white;")

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setSpacing(15)
        self._root_layout.setContentsMargins(20, 20, 20, 20)

        self._add_header(title)

    def _add_header(self, title: str) -> None:
        header = QWidget()
        header.setStyleSheet(f"background-color: {UI_COLORS['header_bg']}; border-radius: {UI_SIZES['header_radius']}px;")
        header.setFixedHeight(UI_SIZES["title_bar_height"])

        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        label = QLabel(title)
        label.setFont(QFont(UI_FONT_FAMILY, UI_SIZES["dialog_title_font"], QFont.Bold))
        label.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(label, 1)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(UI_SIZES["close_btn"], UI_SIZES["close_btn"])
        close_btn.setStyleSheet(
            f"background-color: {UI_COLORS['header_close']}; color: white; "
            f"border-radius: {UI_SIZES['close_btn'] // 2}px; font-size: 24px; border: none;"
        )
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn, 0, Qt.AlignRight)

        self._root_layout.addWidget(header)

    def _make_label(
        self,
        text: str,
        font_size: int = UI_SIZES["body_font"],
        color: str = UI_COLORS["body_text"],
        bold: bool = False,
        align: int = Qt.AlignLeft,
    ) -> QLabel:
        label = QLabel(text)
        weight = QFont.Bold if bold else QFont.Normal
        label.setFont(QFont(UI_FONT_FAMILY, font_size, weight))
        label.setStyleSheet(f"color: {color};")
        label.setAlignment(align)
        return label

    def _make_button(
        self,
        text: str,
        bg_color: str = UI_COLORS["primary"],
        font_size: int = UI_SIZES["body_bold_font"],
        padding: str = "10px",
    ) -> QPushButton:
        btn = QPushButton(text)
        btn.setFont(QFont(UI_FONT_FAMILY, font_size, QFont.Bold))
        btn.setStyleSheet(
            f"background-color: {bg_color}; color: white; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: {padding};"
        )
        return btn

    def _finish_layout(self) -> None:
        self._root_layout.addStretch()


class ModeSelectDialog(BaseDialog):
    """模式选择对话框"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("选择模式", 400, 300, parent)
        self.selected_mode: Optional[str] = None

        self._root_layout.addStretch()

        title = QLabel("CMKA 目标检测系统")
        title.setFont(QFont(UI_FONT_FAMILY, UI_SIZES["dialog_title_bold_font"], QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {UI_COLORS['primary']};")
        self._root_layout.addWidget(title)

        subtitle = QLabel("请选择运行模式")
        subtitle.setFont(QFont(UI_FONT_FAMILY, UI_SIZES["body_font"]))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {UI_COLORS['info_text']};")
        self._root_layout.addWidget(subtitle)

        self._root_layout.addSpacing(20)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        qt_btn = self._make_button("桌面模式", UI_COLORS["primary"])
        qt_btn.clicked.connect(lambda: self._select_mode("qt"))
        btn_layout.addWidget(qt_btn)

        web_btn = self._make_button("网页模式", UI_COLORS["secondary"])
        web_btn.clicked.connect(lambda: self._select_mode("web"))
        btn_layout.addWidget(web_btn)

        self._root_layout.addLayout(btn_layout)
        self._finish_layout()

    def _select_mode(self, mode: str) -> None:
        self.selected_mode = mode
        self.accept()


class CameraSelectDialog(BaseDialog):
    """摄像头选择对话框"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("选择摄像头", 520, 480, parent)
        self.selected_camera: Optional[int] = None
        self._failed_cameras: List[int] = []

        self.camera_list = QListWidget()
        self.camera_list.setFont(QFont(UI_FONT_FAMILY, UI_SIZES["body_font"]))
        self.camera_list.setStyleSheet(
            f"QListWidget {{ border: 1px solid {UI_COLORS['border_light']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; padding: 10px; }}"
        )
        self._root_layout.addWidget(self.camera_list)

        self.info_label = self._make_label("正在扫描摄像头...", 10, UI_COLORS["info_text"], align=Qt.AlignCenter)
        self._root_layout.addWidget(self.info_label)

        btn_layout = QHBoxLayout()
        confirm_btn = self._make_button("确认", UI_COLORS["primary"])
        confirm_btn.clicked.connect(self._on_confirm)
        btn_layout.addWidget(confirm_btn)

        cancel_btn = self._make_button("取消", UI_COLORS["secondary"])
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._root_layout.addLayout(btn_layout)
        self._finish_layout()

        self._scan_cameras()

    def _scan_cameras(self) -> None:
        import cv2
        self.camera_list.clear()
        self._failed_cameras = []

        for i in range(MAX_CAMERA_SCAN):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.camera_list.addItem(f"摄像头 {i}（可用）")
                cap.release()
            else:
                self.camera_list.addItem(f"摄像头 {i}（无法打开）")
                item = self.camera_list.item(self.camera_list.count() - 1)
                item.setForeground(QColor(UI_COLORS["primary"]))
                self._failed_cameras.append(i)

        if self._failed_cameras:
            failed_str = ", ".join(str(i) for i in self._failed_cameras)
            self.info_label.setText(f"以下摄像头无法打开：{failed_str}")
            self.info_label.setStyleSheet(f"color: {UI_COLORS['primary']}; padding: 10px; font-weight: bold;")
        else:
            self.info_label.setText("所有扫描的摄像头均可正常使用")
            self.info_label.setStyleSheet(f"color: {UI_COLORS['body_text']}; padding: 10px;")

    def _on_confirm(self) -> None:
        current = self.camera_list.currentRow()
        if current >= 0:
            self.selected_camera = current
            self.accept()
        else:
            QMessageBox.warning(self, "警告", "请选择一个摄像头")


class DisableClassDialog(BaseDialog):
    """禁用类别对话框"""

    def __init__(self, engine: DetectionEngine, parent: Optional[QWidget] = None) -> None:
        super().__init__("禁用类别", 500, 600, parent)
        self.engine = engine

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: 1px solid {UI_COLORS['scroll_border']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; background-color: {UI_COLORS['scroll_bg']}; }}"
        )

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        self.checkboxes: List[Tuple[str, QCheckBox]] = []
        classes = self.engine.get_classes()
        disabled = set(self.engine.get_settings()["disabled_classes"])

        for class_id, class_name in sorted(classes.items()):
            cb = QCheckBox(f"{class_id}. {class_name}")
            cb.setFont(QFont(UI_FONT_FAMILY, UI_SIZES["small_font"]))
            cb.setStyleSheet(f"padding: 5px; color: {UI_COLORS['label_text']};")
            if class_name in disabled:
                cb.setChecked(True)
            self.checkboxes.append((class_name, cb))
            scroll_layout.addWidget(cb)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        self._root_layout.addWidget(scroll)

        btn_layout = QHBoxLayout()
        clear_btn = self._make_button("清除全部", UI_COLORS["border_light"])
        clear_btn.clicked.connect(self._clear_all)
        btn_layout.addWidget(clear_btn)
        save_btn = self._make_button("保存", UI_COLORS["primary"])
        save_btn.clicked.connect(self._save_and_close)
        btn_layout.addWidget(save_btn)
        self._root_layout.addLayout(btn_layout)
        self._finish_layout()

    def _clear_all(self) -> None:
        for _, cb in self.checkboxes:
            cb.setChecked(False)

    def _save_and_close(self) -> None:
        disabled = [name for name, cb in self.checkboxes if cb.isChecked()]
        self.engine.set_setting("disabled_classes", disabled)
        QMessageBox.information(self, "成功", f"已禁用 {len(disabled)} 个类别")
        self.accept()


class TrackSettingsDialog(BaseDialog):
    """轨迹设置对话框"""

    HEADER_BG = UI_COLORS["teal"]
    CLOSE_BG = UI_COLORS["teal_light"]

    def __init__(self, engine: DetectionEngine, parent: Optional[QWidget] = None) -> None:
        super().__init__("轨迹设置", 520, 520, parent)
        self.engine = engine
        settings = self.engine.get_settings()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: 2px solid {UI_COLORS['teal']}; "
            f"border-radius: {UI_SIZES['btn_radius']}px; background-color: #E0FFFF; }}"
        )
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # 复选框
        self.trajectory_checkbox = self._make_checkbox("显示轨迹线", settings["show_trajectory"], self._on_trajectory_toggle)
        scroll_layout.addWidget(self.trajectory_checkbox)
        self.prediction_checkbox = self._make_checkbox("显示预测方向", settings["show_prediction"], self._on_prediction_toggle)
        scroll_layout.addWidget(self.prediction_checkbox)
        self.tracker_mode_checkbox = self._make_checkbox("使用增强匹配模式", settings["tracker_mode"] == "bytetrack", self._on_tracker_mode_toggle)
        scroll_layout.addWidget(self.tracker_mode_checkbox)

        # 轨迹颜色
        traj_group = self._make_color_group(
            "轨迹线颜色",
            [("红色", (0, 0, 255)), ("蓝色", (255, 0, 0)), ("绿色", (0, 255, 0)),
             ("黄色", (0, 255, 255)), ("紫色", (128, 0, 128)), ("白色", (255, 255, 255))],
            self._set_trajectory_color,
            tuple(settings["trajectory_color"]),
            "trajectory_color",
        )
        scroll_layout.addWidget(traj_group)

        # 预测颜色
        pred_group = self._make_color_group(
            "预测方向颜色",
            [("黄色", (0, 255, 255)), ("红色", (0, 0, 255)), ("蓝色", (255, 0, 0)),
             ("绿色", (0, 255, 0)), ("紫色", (128, 0, 128)), ("白色", (255, 255, 255))],
            self._set_prediction_color,
            tuple(settings["prediction_color"]),
            "prediction_color",
        )
        scroll_layout.addWidget(pred_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        self._root_layout.addWidget(scroll)

        tip = self._make_label("提示：点击颜色按钮后立即生效，无需保存", 10, UI_COLORS["info_text"], align=Qt.AlignCenter)
        self._root_layout.addWidget(tip)

        self._finish_layout()

    @staticmethod
    def _make_checkbox(text: str, checked: bool, callback: Callable) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setFont(QFont(UI_FONT_FAMILY, 14))
        cb.setStyleSheet("padding: 15px;")
        cb.setChecked(checked)
        cb.stateChanged.connect(callback)
        return cb

    def _make_color_group(
        self,
        title: str,
        colors: List[Tuple[str, Tuple[int, int, int]]],
        on_preset: Callable,
        current_color: Tuple[int, int, int],
        setting_key: str,
    ) -> QGroupBox:
        group = QGroupBox(title)
        group.setFont(QFont(UI_FONT_FAMILY, 12, QFont.Bold))
        group.setStyleSheet("margin: 10px; padding: 15px;")
        layout = QVBoxLayout(group)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(10)
        for name, color in colors:
            btn = QPushButton(name)
            btn.setFixedSize(70, 40)
            btn.setFont(QFont(UI_FONT_FAMILY, UI_SIZES["small_font"]))
            text_color = "black" if sum(color) > 380 else "white"
            btn.setStyleSheet(
                f"background-color: rgb({color[2]}, {color[1]}, {color[0]}); "
                f"color: {text_color}; border-radius: 5px; border: 2px solid #ccc;"
            )
            btn.clicked.connect(lambda checked, c=color: on_preset(c))
            preset_row.addWidget(btn)
        layout.addLayout(preset_row)

        custom_row = QHBoxLayout()
        custom_row.addWidget(self._make_label("自定义颜色:", UI_SIZES["small_font"]))
        custom_btn = QPushButton()
        custom_btn.setFixedSize(40, 40)
        self._update_custom_button(custom_btn, current_color)
        custom_row.addWidget(custom_btn)
        custom_row.addStretch(1)
        layout.addLayout(custom_row)

        # 保存引用供颜色选择器使用
        if setting_key == "trajectory_color":
            self.custom_trajectory_button = custom_btn
            self._choose_trajectory_color = lambda: self._choose_color("选择轨迹线颜色", setting_key, custom_btn)
            custom_btn.clicked.connect(self._choose_trajectory_color)
        else:
            self.custom_prediction_button = custom_btn
            self._choose_prediction_color = lambda: self._choose_color("选择预测方向颜色", setting_key, custom_btn)
            custom_btn.clicked.connect(self._choose_prediction_color)

        return group

    @staticmethod
    def _update_custom_button(button: QPushButton, color: Tuple[int, int, int]) -> None:
        r, g, b = (int(color[0]), int(color[1]), int(color[2])) if len(color) == 3 else (0, 0, 0)
        button.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border-radius: 5px; border: 2px solid {UI_COLORS['teal']};"
        )

    def _on_trajectory_toggle(self, state: int) -> None:
        self.engine.set_setting("show_trajectory", state == Qt.Checked)

    def _on_prediction_toggle(self, state: int) -> None:
        self.engine.set_setting("show_prediction", state == Qt.Checked)

    def _on_tracker_mode_toggle(self, state: int) -> None:
        self.engine.set_setting("tracker_mode", "bytetrack" if state == Qt.Checked else "classic")

    def _set_trajectory_color(self, color: Tuple[int, int, int]) -> None:
        self.engine.set_setting("trajectory_color", color)
        self._update_custom_button(self.custom_trajectory_button, color)

    def _set_prediction_color(self, color: Tuple[int, int, int]) -> None:
        self.engine.set_setting("prediction_color", color)
        self._update_custom_button(self.custom_prediction_button, color)

    def _choose_color(self, dialog_title: str, setting_key: str, button: QPushButton) -> None:
        try:
            settings = self.engine.get_settings()
            r, g, b = settings[setting_key]
            qcolor = QColor(r, g, b)
            color = QColorDialog.getColor(qcolor, self, dialog_title)
            if color.isValid():
                new_color = (color.red(), color.green(), color.blue())
                if "trajectory" in setting_key:
                    self._set_trajectory_color(new_color)
                else:
                    self._set_prediction_color(new_color)
        except Exception as exc:
            logger.error("选择颜色错误: %s", exc)


class ModelManageDialog(BaseDialog):
    """模型管理对话框"""

    HEADER_BG = UI_COLORS["teal"]
    CLOSE_BG = UI_COLORS["teal_light"]

    def __init__(self, engine: DetectionEngine, parent: Optional[QWidget] = None) -> None:
        super().__init__("模型管理", 620, 360, parent)
        self.engine = engine

        info = self._make_label("选择模型后可直接下载，也可删除当前模型后重载其他模型。", 12, align=Qt.AlignCenter)
        info.setWordWrap(True)
        self._root_layout.addWidget(info)

        select_layout = QHBoxLayout()
        select_layout.addWidget(self._make_label("选择模型："))
        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont(UI_FONT_FAMILY, 12))
        self.model_combo.setMinimumWidth(340)
        for url in MODEL_LIST:
            from core.utils import _model_name_from_url
            self.model_combo.addItem(_model_name_from_url(url), url)
        current_url = self.engine.settings.get("model_url", "")
        idx = self.model_combo.findData(current_url)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        select_layout.addWidget(self.model_combo)
        self._root_layout.addLayout(select_layout)

        self.status_label = QLabel("")
        self.status_label.setFont(QFont(UI_FONT_FAMILY, UI_SIZES["small_font"]))
        self.status_label.setStyleSheet(f"color: {UI_COLORS['info_text']}; padding: 0 15px 15px 15px;")
        self.status_label.setWordWrap(True)
        self._root_layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(15, 0, 15, 15)
        btn_layout.setSpacing(12)
        btn_layout.addWidget(self._make_button("删除旧模型", UI_COLORS["primary"]))
        btn_layout[-1].clicked.connect(self._delete_model)  # type: ignore[index]
        btn_layout.addWidget(self._make_button("下载模型", UI_COLORS["secondary"]))
        btn_layout[-1].clicked.connect(self._download_model)  # type: ignore[index]
        btn_layout.addWidget(self._make_button("重载模型", UI_COLORS["accent"]))
        btn_layout[-1].clicked.connect(self._reload_model)  # type: ignore[index]
        self._root_layout.addLayout(btn_layout)

        self._finish_layout()
        self._update_status()

    def _update_status(self) -> None:
        path = self.engine.get_model_path()
        exists = os.path.exists(path)
        text = f"模型路径：{path}\n状态：{'存在' if exists else '未找到'}"
        if exists:
            try:
                size_mb = os.path.getsize(path) / 1024 / 1024
                text += f"，大小：{size_mb:.2f} MB"
            except OSError:
                pass
        self.status_label.setText(text)

    def _delete_model(self) -> None:
        path = self.engine.get_model_path()
        if os.path.exists(path):
            try:
                os.remove(path)
                QMessageBox.information(self, "成功", "已删除旧模型文件。")
                self._update_status()
            except OSError as exc:
                QMessageBox.critical(self, "错误", f"删除失败：{exc}")
        else:
            QMessageBox.information(self, "提示", "当前没有可删除的模型文件。")

    def _download_model(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            url = self.model_combo.currentData()
            path = self.engine.download_model(url)
            if path:
                QMessageBox.information(self, "成功", f"模型已下载：{path}")
                self._update_status()
            else:
                QMessageBox.warning(self, "失败", "模型下载失败，未保存文件。")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"下载异常：{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _reload_model(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok = self.engine.reload_model()
            if ok:
                QMessageBox.information(self, "成功", "模型已重载。")
            else:
                QMessageBox.warning(self, "失败", "模型重载失败。")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"重载异常：{exc}")
        finally:
            QApplication.restoreOverrideCursor()


class LogExportDialog(BaseDialog):
    """日志导出对话框"""

    HEADER_BG = UI_COLORS["teal"]
    CLOSE_BG = UI_COLORS["teal_light"]

    def __init__(self, engine: DetectionEngine, parent: Optional[QWidget] = None) -> None:
        super().__init__("日志导出", 520, 260, parent)
        self.engine = engine
        self.exporting: bool = False

        self.path_label = QLabel()
        self.path_label.setFont(QFont(UI_FONT_FAMILY, UI_SIZES["small_font"]))
        self.path_label.setStyleSheet(f"color: {UI_COLORS['body_text']}; padding: 15px;")
        self.path_label.setWordWrap(True)
        self._root_layout.addWidget(self.path_label)

        self.status_label = QLabel("状态：未开始")
        self.status_label.setFont(QFont(UI_FONT_FAMILY, UI_SIZES["small_font"]))
        self.status_label.setStyleSheet(f"color: {UI_COLORS['info_text']}; padding: 0 15px 15px 15px;")
        self._root_layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(15, 0, 15, 15)
        btn_layout.setSpacing(12)

        self.start_button = self._make_button("开始导出", UI_COLORS["primary"])
        self.start_button.clicked.connect(self._start_export)
        btn_layout.addWidget(self.start_button)

        self.stop_button = self._make_button("停止导出", UI_COLORS["secondary"])
        self.stop_button.clicked.connect(self._stop_export)
        self.stop_button.setEnabled(False)
        btn_layout.addWidget(self.stop_button)

        open_btn = self._make_button("打开日志目录", UI_COLORS["accent"])
        open_btn.clicked.connect(self._open_logs_dir)
        btn_layout.addWidget(open_btn)

        self._root_layout.addLayout(btn_layout)
        self._finish_layout()

        # 初始化日志路径
        log_path = self.engine.log_path or self.engine.start_log_export()
        self.path_label.setText(f"日志文件：{log_path}")

    def _start_export(self) -> None:
        self.exporting = True
        log_path = self.engine.start_log_export()
        self.path_label.setText(f"日志文件：{log_path}")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("状态：导出中...")

    def _stop_export(self) -> None:
        self.exporting = False
        self.engine.stop_log_export()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("状态：已暂停")

    def _open_logs_dir(self) -> None:
        try:
            path = get_logs_dir(self.engine.base_path)
            if sys.platform == "win32":
                os.startfile(path)
            else:
                QMessageBox.information(self, "路径", path)
        except Exception as exc:
            QMessageBox.warning(self, "失败", str(exc))

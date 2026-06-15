import os
import sys

import cv2
import numpy as np
import traceback
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
                            QMessageBox, QDialog, QCheckBox, QScrollArea, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont

try:
    from PIL import Image, ImageDraw, ImageFont
    pil_available = True
except ImportError:
    print("未找到PIL库，请安装: pillow")
    pil_available = False
    sys.exit(1)


def put_chinese_text(img, text, position, font_size=20, color=(0, 0, 0)):
    if not pil_available:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/20, color, 2)
        return img
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path = None
    if os.name == 'nt':
        possible_fonts = [
            r'C:\Windows\Fonts\simhei.ttf',
            r'C:\Windows\Fonts\simsun.ttc',
            r'C:\Windows\Fonts\microsoftyahei.ttf',
        ]
        for font in possible_fonts:
            if os.path.exists(font):
                font_path = font
                break
    
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"PIL绘制失败，使用OpenCV: {e}")
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/20, color, 2)
    
    return img


def draw_rounded_rectangle(img, pt1, pt2, color, thickness=-1, radius=20):
    x1, y1 = pt1
    x2, y2 = pt2
    
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    
    return img


print("正在加载YOLO26n模型...")
use_ultralytics = False
model = None

# 获取程序运行目录（兼容打包后的exe）
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
    # 检查模型文件是否在当前目录，如果不在则检查 _internal 目录
    pt_model_path = os.path.join(base_path, 'yolo26n.pt')
    if not os.path.exists(pt_model_path):
        internal_path = os.path.join(base_path, '_internal', 'yolo26n.pt')
        if os.path.exists(internal_path):
            pt_model_path = internal_path
            base_path = os.path.join(base_path, '_internal')
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
    pt_model_path = os.path.join(base_path, 'yolo26n.pt')

# 使用 Ultralytics
try:
    from ultralytics import YOLO
    if os.path.exists(pt_model_path):
        model = YOLO(pt_model_path)
        use_ultralytics = True
        print(f"YOLO26n模型加载成功！(Ultralytics格式: {pt_model_path})")
        classes = model.names
    else:
        print(f"警告: 未找到模型文件: {pt_model_path}")
        print("程序将以仅显示视频模式运行（无检测功能）...")
except ImportError:
    print("未安装 ultralytics 库")
    print("请运行: pip install ultralytics")
    sys.exit(1)
except Exception as e:
    print(f"Ultralytics 加载失败: {e}")
    print("程序将以仅显示视频模式运行（无检测功能）...")

# COCO 类别名称
classes = {i: name for i, name in enumerate([
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
disabled_classes = set()

# 轨迹跟踪设置
show_trajectory = True
show_prediction = True
trajectory_color = (0, 0, 255)  # BGR格式，红色
prediction_color = (0, 255, 255)  # BGR格式，黄色


class TrackedObject:
    def __init__(self, track_id, x, y, w, h, class_id, confidence):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_id = class_id
        self.confidence = confidence
        self.trajectory = [(x + w//2, y + h//2)]
        self.missing_frames = 0
        self.max_missing = 10
    
    def update(self, x, y, w, h, confidence):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.trajectory.append((x + w//2, y + h//2))
        self.missing_frames = 0
        
        if len(self.trajectory) > 50:
            self.trajectory = self.trajectory[-50:]
    
    def predict_next_position(self):
        if len(self.trajectory) >= 2:
            x1, y1 = self.trajectory[-1]
            x2, y2 = self.trajectory[-2]
            dx = x1 - x2
            dy = y1 - y2
            return (x1 + dx, y1 + dy)
        return None
    
    def is_lost(self):
        return self.missing_frames >= self.max_missing


class ObjectTracker:
    def __init__(self):
        self.tracked_objects = []
        self.next_track_id = 1
        self.max_iou_distance = 0.3
    
    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        return inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    
    def update(self, detections):
        new_tracked = []
        
        for det in detections:
            x, y, w, h, class_id, confidence = det
            matched = False
            
            for obj in self.tracked_objects:
                if obj.class_id == class_id:
                    iou_dist = self.iou((x, y, w, h), (obj.x, obj.y, obj.w, obj.h))
                    if iou_dist > self.max_iou_distance:
                        obj.update(x, y, w, h, confidence)
                        new_tracked.append(obj)
                        matched = True
                        break
            
            if not matched:
                new_obj = TrackedObject(self.next_track_id, x, y, w, h, class_id, confidence)
                self.next_track_id += 1
                new_tracked.append(new_obj)
        
        for obj in self.tracked_objects:
            if obj not in new_tracked:
                obj.missing_frames += 1
        
        self.tracked_objects = [obj for obj in self.tracked_objects if not obj.is_lost()]
        self.tracked_objects.extend([obj for obj in new_tracked if obj not in self.tracked_objects])
        
        return self.tracked_objects
    
    def get_all_tracks(self):
        return self.tracked_objects


class LicenseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("使用码验证")
        self.setFixedSize(500, 450)
        self.setStyleSheet("background-color: white;")
        
        self.correct_code = "Windows123fish"
        self.max_attempts = 5
        self.attempts = 0
        
        layout = QVBoxLayout()
        
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_widget.setStyleSheet("background-color: #FFB6C1; border-radius: 15px;")
        
        title_label = QLabel("使用码验证")
        title_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        header_layout.addWidget(title_label, Qt.AlignLeft)
        
        close_button = QPushButton("×")
        close_button.setFixedSize(40, 40)
        close_button.setStyleSheet("background-color: #FFC0CB; border-radius: 20px; color: white; font-size: 24px;")
        close_button.clicked.connect(self.reject)
        header_layout.addWidget(close_button, Qt.AlignRight)
        
        layout.addWidget(header_widget)
        
        info_widget = QWidget()
        info_widget.setStyleSheet("background-color: #E0FFFF; border: 2px solid #B0E0E6; border-radius: 10px; margin: 20px;")
        info_layout = QVBoxLayout(info_widget)
        
        info_label1 = QLabel("重要提示：本软件为免费软件")
        info_label1.setFont(QFont("Microsoft YaHei", 14))
        info_label1.setStyleSheet("color: #0066CC;")
        info_layout.addWidget(info_label1)
        
        layout.addWidget(info_widget)
        
        input_label = QLabel("请输入使用码以继续使用本软件")
        input_label.setFont(QFont("Microsoft YaHei", 14))
        input_label.setStyleSheet("color: #404040; margin: 0 30px 15px 30px;")
        input_label.setWordWrap(True)
        input_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(input_label)
        
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(40, 10, 40, 20)
        
        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("请输入使用码")
        self.code_input.setEchoMode(QLineEdit.Password)
        self.code_input.setFont(QFont("Microsoft YaHei", 16))
        self.code_input.setStyleSheet("background-color: #FFE4E1; border: 2px solid #FFB6C1; border-radius: 10px; padding: 12px;")
        self.code_input.returnPressed.connect(self.verify_code)
        input_layout.addWidget(self.code_input, 4)
        
        verify_button = QPushButton("验证")
        verify_button.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        verify_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 10px; padding: 12px 20px;")
        verify_button.clicked.connect(self.verify_code)
        input_layout.addWidget(verify_button, 1)
        
        layout.addLayout(input_layout)
        
        self.error_label = QLabel()
        self.error_label.setFont(QFont("Microsoft YaHei", 12))
        self.error_label.setStyleSheet("color: red; text-align: center; margin: 0 30px 20px 30px;")
        self.error_label.setWordWrap(True)
        self.error_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.error_label)
        
        bottom_widget = QWidget()
        bottom_widget.setStyleSheet("background-color: #F0F8FF; border-radius: 10px; margin: 0 30px 20px 30px;")
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(20, 15, 20, 15)
        
        hint_label1 = QLabel("请使用enter(回车)键确认")
        hint_label1.setFont(QFont("Microsoft YaHei", 12))
        hint_label1.setStyleSheet("color: #9370DB;")
        hint_label1.setAlignment(Qt.AlignCenter)
        bottom_layout.addWidget(hint_label1)
        
        hint_label2 = QLabel("使用码：Windows123fish")
        hint_label2.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        hint_label2.setStyleSheet("color: #9370DB; margin-top: 5px;")
        hint_label2.setAlignment(Qt.AlignCenter)
        bottom_layout.addWidget(hint_label2)
        
        layout.addWidget(bottom_widget)
        
        self.setLayout(layout)
        self.code_input.setFocus()
    
    def verify_code(self):
        user_input = self.code_input.text()
        if user_input == self.correct_code:
            QMessageBox.information(self, "验证成功", "欢迎使用本软件")
            self.accept()
        else:
            self.attempts += 1
            self.error_label.setText(f"使用码错误，请重试\n剩余尝试次数：{self.max_attempts - self.attempts}")
            self.code_input.clear()
            if self.attempts >= self.max_attempts:
                QMessageBox.critical(self, "验证失败", "尝试次数已用完")
                self.reject()


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
        max_cameras = 3  # 减少检测数量，避免卡死
        self.available_cameras = []
        
        print("正在检测可用摄像头...")
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 使用 DirectShow 加快检测
                if cap.isOpened():
                    self.available_cameras.append(i)
                    
                    # 简化显示信息，避免获取设备名称导致卡死
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
                    
                    # 立即释放摄像头资源
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("禁用识别类别")
        self.setFixedSize(550, 600)
        self.setStyleSheet("background-color: white;")
        
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
        
        for class_id, class_name in classes.items():
            checkbox = QCheckBox(f"{class_id}. {class_name}")
            checkbox.setFont(QFont("Microsoft YaHei", 11))
            checkbox.setStyleSheet("padding: 8px; color: #696969;")
            if class_name in disabled_classes:
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
        global disabled_classes
        disabled_classes.clear()
        for class_name, checkbox in self.checkboxes:
            if checkbox.isChecked():
                disabled_classes.add(class_name)
        
        QMessageBox.information(self, "成功", f"已禁用 {len(disabled_classes)} 个类别")
        self.accept()


class TrackSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("轨迹设置")
        self.setMinimumSize(500, 450)
        self.setStyleSheet("background-color: white;")
        
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
        close_button.setStyleSheet("background-color: #48D1CC; border-radius: 20px; color: white; font-size: 24px;")
        close_button.clicked.connect(self.close)
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
        
        self.trajectory_checkbox = QCheckBox("显示轨迹线")
        self.trajectory_checkbox.setFont(QFont("Microsoft YaHei", 14))
        self.trajectory_checkbox.setStyleSheet("padding: 15px;")
        self.trajectory_checkbox.setChecked(show_trajectory)
        self.trajectory_checkbox.stateChanged.connect(self.on_trajectory_toggle)
        scroll_layout.addWidget(self.trajectory_checkbox)
        
        self.prediction_checkbox = QCheckBox("显示预测方向")
        self.prediction_checkbox.setFont(QFont("Microsoft YaHei", 14))
        self.prediction_checkbox.setStyleSheet("padding: 15px;")
        self.prediction_checkbox.setChecked(show_prediction)
        self.prediction_checkbox.stateChanged.connect(self.on_prediction_toggle)
        scroll_layout.addWidget(self.prediction_checkbox)
        
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
            btn.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); color: {'black' if sum(color) > 380 else 'white'}; border-radius: 5px; border: 2px solid #ccc;")
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
        self.update_custom_button_color(self.custom_trajectory_button, trajectory_color)
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
            btn.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); color: {'black' if sum(color) > 380 else 'white'}; border-radius: 5px; border: 2px solid #ccc;")
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
        self.update_custom_button_color(self.custom_prediction_button, prediction_color)
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
        global show_trajectory
        show_trajectory = (state == Qt.Checked)
    
    def on_prediction_toggle(self, state):
        global show_prediction
        show_prediction = (state == Qt.Checked)
    
    def set_trajectory_color(self, color):
        global trajectory_color
        trajectory_color = color
        self.update_custom_button_color(self.custom_trajectory_button, color)
    
    def set_prediction_color(self, color):
        global prediction_color
        prediction_color = color
        self.update_custom_button_color(self.custom_prediction_button, color)
    
    def choose_custom_trajectory_color(self):
        try:
            from PyQt5.QtWidgets import QColorDialog
            from PyQt5.QtGui import QColor
            
            r, g, b = trajectory_color
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
            
            r, g, b = prediction_color
            qcolor = QColor(r, g, b)
            color = QColorDialog.getColor(qcolor, self, "选择预测方向颜色")
            
            if color.isValid():
                new_color = (color.red(), color.green(), color.blue())
                self.set_prediction_color(new_color)
        except Exception as e:
            print(f"选择预测方向颜色错误: {e}")


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.running = True
        self.tracker = ObjectTracker()
    
    def run(self):
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # 使用 DirectShow
        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_id}")
            return
            
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                detections = []
                tracked_objects = []
                
                if use_ultralytics and model is not None:
                    try:
                        results = model(frame, conf=0.5, iou=0.45, verbose=False)

                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                w = x2 - x1
                                h = y2 - y1
                                confidence = float(box.conf[0])
                                class_id = int(box.cls[0])
                                class_name = classes[class_id]

                                if class_name in disabled_classes:
                                    continue

                                detections.append((x1, y1, w, h, class_id, confidence))
                                
                    except Exception as e:
                        print(f"检测错误: {e}")
                
                tracked_objects = self.tracker.update(detections)
                
                for obj in tracked_objects:
                    x, y, w, h = obj.x, obj.y, obj.w, obj.h
                    class_id = obj.class_id
                    track_id = obj.track_id
                    
                    if 0 <= class_id < len(classes):
                        class_name = classes[class_id]
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        label = f"ID:{track_id} {class_name}: {obj.confidence:.2f}"
                        frame = put_chinese_text(frame, label, (x, y - 10), font_size=14, color=(0, 255, 0))
                        
                        if show_trajectory and len(obj.trajectory) > 1 and class_name != 'person':
                            for i in range(1, len(obj.trajectory)):
                                pt1 = obj.trajectory[i-1]
                                pt2 = obj.trajectory[i]
                                cv2.line(frame, pt1, pt2, trajectory_color, 2)
                        
                        if show_prediction and class_name != 'person':
                            predicted_pos = obj.predict_next_position()
                            if predicted_pos is not None:
                                current_pos = (x + w // 2, y + h // 2)
                                pred_x, pred_y = int(predicted_pos[0]), int(predicted_pos[1])
                                self.draw_direction_arrow(frame, current_pos, (pred_x, pred_y), prediction_color)
                
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(convert_to_Qt_format.copy())  # 使用 copy() 避免内存问题
            except Exception as e:
                print(f"视频处理错误: {e}")
                continue
        
        cap.release()
        print("摄像头已释放")
    
    def draw_direction_arrow(self, frame, start_point, end_point, color):
        import math
        
        start_x, start_y = start_point
        end_x, end_y = end_point
        
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < 5:
            return
        
        arrow_length = min(distance, 30)
        
        angle = math.atan2(dy, dx)
        
        end_x = start_x + int(arrow_length * math.cos(angle))
        end_y = start_y + int(arrow_length * math.sin(angle))
        
        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 3)
        
        arrow_head_length = 10
        angle1 = angle + math.pi / 6
        angle2 = angle - math.pi / 6
        
        x1 = end_x - int(arrow_head_length * math.cos(angle1))
        y1 = end_y - int(arrow_head_length * math.sin(angle1))
        
        x2 = end_x - int(arrow_head_length * math.cos(angle2))
        y2 = end_y - int(arrow_head_length * math.sin(angle2))
        
        cv2.line(frame, (end_x, end_y), (x1, y1), color, 3)
        cv2.line(frame, (end_x, end_y), (x2, y2), color, 3)
    
    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self, camera_id):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(100, 100, 900, 700)
        
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
        self.video_label.setStyleSheet("border: 2px solid #B0E0E6; border-radius: 10px; min-height: 480px;")
        self.video_label.setScaledContents(True)
        content_layout.addWidget(self.video_label)
        
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 20, 0, 0)
        control_layout.setSpacing(15)
        
        self.start_button = QPushButton("开始检测")
        self.start_button.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.start_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 10px; padding: 12px 24px;")
        self.start_button.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止检测")
        self.stop_button.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.stop_button.setStyleSheet("background-color: #FF69B4; color: white; border-radius: 10px; padding: 12px 24px;")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        self.switch_button = QPushButton("切换摄像头")
        self.switch_button.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.switch_button.setStyleSheet("background-color: #4682B4; color: white; border-radius: 10px; padding: 12px 24px;")
        self.switch_button.clicked.connect(self.switch_camera)
        control_layout.addWidget(self.switch_button)
        
        self.disable_button = QPushButton("禁用类别")
        self.disable_button.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.disable_button.setStyleSheet("background-color: #9370DB; color: white; border-radius: 10px; padding: 12px 24px;")
        self.disable_button.clicked.connect(self.open_disable_dialog)
        control_layout.addWidget(self.disable_button)
        
        self.track_button = QPushButton("轨迹设置")
        self.track_button.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.track_button.setStyleSheet("background-color: #20B2AA; color: white; border-radius: 10px; padding: 12px 24px;")
        self.track_button.clicked.connect(self.open_track_settings)
        control_layout.addWidget(self.track_button)
        
        content_layout.addLayout(control_layout)
        main_layout.addWidget(content_widget)
        
        self.thread = None
        self.camera_id = camera_id
        
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
        self.thread = VideoThread(self.camera_id)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_detection(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def update_image(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
    
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
        disable_dialog = DisableClassDialog(self)
        disable_dialog.exec_()
    
    def open_track_settings(self):
        track_dialog = TrackSettingsDialog(self)
        track_dialog.exec_()
    
    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    license_dialog = LicenseDialog()
    if not license_dialog.exec_():
        print("验证失败，程序退出")
        return
    
    print("验证成功，欢迎使用！")
    
    camera_dialog = CameraSelectDialog()
    if not camera_dialog.exec_():
        print("未选择摄像头，程序退出")
        return
    
    camera_id = camera_dialog.selected_camera
    
    window = MainWindow(camera_id)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
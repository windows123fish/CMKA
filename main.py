import cv2
import os
import urllib.request
import shutil
import numpy as np
import sys
import traceback
import ctypes
import webbrowser  # 添加webbrowser模块用于打开超链接
# 导入pkg_resources.py2_warn以解决PyInstaller打包问题
try:
    import pkg_resources.py2_warn
except ImportError:
    pass
# 移除未使用的tempfile引用

# 尝试导入PIL库
try:
    from PIL import Image, ImageDraw, ImageFont
    pil_available = True
except ImportError:
    print("未找到PIL库，请安装: pip install pillow")
    pil_available = False
    sys.exit(1)

# 使用PIL绘制中文的函数
def put_chinese_text(img, text, position, font_size=20, color=(0, 0, 0)):
    if not pil_available:
        # 如果PIL不可用，使用OpenCV默认绘制（可能无法显示中文）
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/20, color, 2)
        return img
    
    # 转换OpenCV图像到PIL格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体
    font_path = None
    # 尝试Windows系统字体
    if os.name == 'nt':
        possible_fonts = [
            r'C:\Windows\Fonts\simhei.ttf',  # 黑体 (使用原始字符串避免转义问题)
            r'C:\Windows\Fonts\simsun.ttc',  # 宋体
            r'C:\Windows\Fonts\microsoftyahei.ttf',  # 微软雅黑
        ]
        for font in possible_fonts:
            if os.path.exists(font):
                font_path = font
                break
    
    # 如果找不到系统字体，使用默认字体
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# YOLOv3-tiny模型配置和权重文件路径
yolo_config = 'yolov3-tiny.cfg'
yolo_weights = 'yolov3-tiny.weights'
classes_file = 'coco.names'

# 下载函数（添加用户代理头）
def download_file(url, filename, fallback_urls=None, min_expected_size=0):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            # 检查Content-Length是否符合预期
            content_length = response.getheader('Content-Length')
            if content_length and min_expected_size > 0:
                if int(content_length) < min_expected_size:
                    print(f"文件大小不符合预期 {url}: {content_length}字节 < {min_expected_size}字节")
                    return False
            
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            
        print(f"下载成功: {filename}")
        
        # 验证文件是否实际创建且大小符合预期
        if not os.path.exists(filename):
            print(f"警告: 文件下载成功但未找到 {filename}")
            return False
        
        file_size = os.path.getsize(filename)
        if min_expected_size > 0 and file_size < min_expected_size:
            print(f"文件大小不符合预期 {filename}: {file_size}字节 < {min_expected_size}字节")
            os.remove(filename)
            return False
        
        return True
    except (urllib.error.HTTPError, OSError) as e:
        print(f"下载或文件写入失败 {url}: {str(e)}")
        if fallback_urls and len(fallback_urls) > 0:
            next_url = fallback_urls.pop(0)
            print(f"尝试备用链接: {next_url}")
            return download_file(next_url, filename, fallback_urls, min_expected_size)
        return False

# 检查并下载模型文件
if not os.path.exists(yolo_config):
    print(f"正在下载{yolo_config}...")
    # 配置文件多URL fallback机制
    config_success = download_file(
        'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',
        yolo_config,
        fallback_urls=[
            'https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov3-tiny.cfg',
            'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/yolov3-tiny.cfg'
        ]
    )
    if not config_success:
        print("所有配置文件下载链接均失败，请手动下载配置文件并放置到项目目录")
        print("推荐下载地址: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg")
        sys.exit(1)

# 检查并下载权重文件
# 检查权重文件是否存在，如不存在则引导手动下载
if not os.path.exists(yolo_weights):
    print("=============================================")
    print("权重文件下载失败: 所有自动下载链接均不可用")
    print("请手动下载权重文件并放置到项目目录:")
    print("1. 访问: https://pjreddie.com/media/files/yolov3-tiny.weights")
    print("2. 将文件保存为: yolov3-tiny.weights")
    print("3. 确保文件大小约为34MB")
    print("=============================================")
    print("所有下载链接均失败，请手动下载权重文件并放置到项目目录")  # 移除潜在 unreachable 代码问题，需确保代码上下文逻辑正确以避免该提示
    print("推荐下载地址1: https://pjreddie.com/media/files/yolov3-tiny.weights")
    print("推荐下载地址2: https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov3-tiny.pt")
    print("下载后请重命名为'yolov3-tiny.weights'并放在当前目录")
    sys.exit(1)

# 验证文件是否存在
if not os.path.exists(yolo_weights):
    print(f"下载错误: {yolo_weights}文件不存在")
    sys.exit(1)

# 验证文件完整性并处理潜在的文件删除竞态条件
try:
    weights_size = os.path.getsize(yolo_weights)
except FileNotFoundError:
    print(f"致命错误: {yolo_weights}文件在验证过程中丢失")
    sys.exit(1)

if weights_size < 1*1024*1024:  # 降低至1MB阈值
    print(f"权重文件{yolo_weights}损坏或不完整 (大小: {weights_size}字节)")
    print("请尝试手动下载: https://pjreddie.com/media/files/yolov3-tiny.weights")
    os.remove(yolo_weights)
    sys.exit(1)

if os.path.getsize(yolo_config) < 1024:  # 小于1KB视为配置文件异常
    print(f"配置文件{yolo_config}损坏或不完整")
    os.remove(yolo_config)
    sys.exit(1)

if not os.path.exists(classes_file):
    print(f"正在下载{classes_file}...")
    try:
        download_file('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names', classes_file)
    except:
        print("下载失败，使用内置COCO类别列表")
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        with open(classes_file, 'w') as f:
            f.write('\n'.join(coco_classes))

# 加载YOLO模型
# 加载模型并添加错误处理
try:
    net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
except cv2.error as e:
    print(f"模型加载失败: {str(e)}")
    print("详细错误信息:\n", traceback.format_exc())
    print("可能原因: 权重文件与配置文件不匹配或OpenCV版本不兼容")
    sys.exit(1)

# 设置OpenCV DNN后端优化和多线程
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cv2.setNumThreads(4)  # 减少线程数以降低资源占用

# 免费软件声明画面
def show_free_software_notice():
    cv2.namedWindow('软件声明', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('软件声明', 600, 400)
    
    clicked = False
    
    def on_mouse_click(event, x, y, flags, param):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True
    
    cv2.setMouseCallback('软件声明', on_mouse_click)
    
    while True:
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 绘制标题
        title_color = (148, 163, 252)
        frame = put_chinese_text(frame, "软件声明", (250, 50), font_size=32, color=title_color)
        
        # 绘制主要内容
        text_color = (200, 200, 200)
        frame = put_chinese_text(frame, "本软件为免费软件", (120, 120), font_size=20, color=text_color)
        frame = put_chinese_text(frame, "如果您是购买的", (120, 160), font_size=20, color=text_color)
        frame = put_chinese_text(frame, "请尽快联系退款与举报", (120, 200), font_size=20, color=text_color)
        
        # 绘制提示文字
        hint_color = (148, 163, 252)
        frame = put_chinese_text(frame, "点击任意处继续", (220, 320), font_size=16, color=hint_color)
        
        # 绘制边框
        cv2.rectangle(frame, (20, 20), (580, 380), (148, 163, 252), 2)
        
        cv2.imshow('软件声明', frame)
        
        key = cv2.waitKey(25) & 0xFF
        if clicked or key == ord('q') or key == 13 or cv2.getWindowProperty('软件声明', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()

# 显示软件声明
show_free_software_notice()


# 加载类别名称
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 获取输出层名称（兼容不同OpenCV版本）
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 摄像头选择GUI函数
def select_camera_gui():
    # 尝试检测可用摄像头数量
    max_cameras = 10  # 最大尝试摄像头数量
    available_cameras = []
    camera_info = []
    
    print("正在检测可用摄像头...")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            
            # 尝试获取摄像头信息
            try:
                # 获取摄像头名称
                name = cap.get(cv2.CAP_PROP_DEVICE_NAME)
                if name is None or name == 0:
                    name = f"摄像头 {i}"
                else:
                    # 确保名称是字符串并处理编码
                    name = str(name)
                    # 尝试解码可能的中文编码
                    try:
                        name = name.encode('latin1').decode('gbk')
                    except:
                        pass
            except:
                name = f"摄像头 {i}"
            
            # 获取分辨率信息
            try:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                res_info = f" ({int(width)}x{int(height)})"
            except:
                res_info = ""
            
            camera_info.append(f"{i}: {name}{res_info}")
            cap.release()
            print(f"发现摄像头 {i}: {name}")
    
    if not available_cameras:
        print("未找到可用摄像头")
        sys.exit(1)
    
    # 创建选择窗口
    cv2.namedWindow("摄像头选择")
    selected_camera = [available_cameras[0]]  # 默认选中第一个摄像头
    
    # 绘制窗口内容的函数
    def draw_window():
        # 创建空白图像作为窗口背景
        height = 50 + len(available_cameras) * 30
        width = 600
        img = np.ones((height, width, 3), dtype=np.uint8) * 240  # 浅灰色背景
        
        # 绘制标题 - 使用PIL绘制中文
        img = put_chinese_text(img, "请选择摄像头", (20, 10), font_size=24, color=(0, 0, 0))
        
        # 绘制作者主页超链接 - 淡蓝色文字 (BGR格式)
        img = put_chinese_text(img, "作者主页", (20, 40), font_size=14, color=(230, 216, 173))
        
        # 绘制摄像头列表
        for i, info in enumerate(camera_info):
            y_pos = 60 + i * 30
            color = (0, 0, 255) if available_cameras[i] == selected_camera[0] else (0, 0, 0)
            img = put_chinese_text(img, info, (20, y_pos-10), font_size=16, color=color)
            
            # 绘制选择框
            if available_cameras[i] == selected_camera[0]:
                cv2.rectangle(img, (10, y_pos-20), (width-10, y_pos+10), (0, 255, 0), 2)
        
        # 绘制确认提示
        img = put_chinese_text(img, "按Enter键确认选择", (width-200, height-20), font_size=16, color=(0, 0, 255))
        
        cv2.imshow("摄像头选择", img)
    
    # 鼠标事件处理函数
    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击作者主页链接
            if y > 30 and y < 50 and x > 20 and x < 100:
                webbrowser.open("https://space.bilibili.com/3493134080149590")
                return
            
            # 检查点击位置是否在摄像头列表项上
            for i in range(len(available_cameras)):
                y_pos = 60 + i * 30
                if y > y_pos-20 and y < y_pos+10 and x > 10 and x < 600-10:
                    selected_camera[0] = available_cameras[i]
                    draw_window()
                    break
    
    # 注册鼠标事件
    cv2.setMouseCallback("摄像头选择", mouse_event)
    
    # 显示窗口
    draw_window()
    
    # 等待用户选择
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter键
            break
        elif key == ord('q'):
            print("用户取消选择")
            sys.exit(0)
    
    # 关闭选择窗口
    cv2.destroyWindow("摄像头选择")
    
    # 初始化选定的摄像头
    cap = cv2.VideoCapture(selected_camera[0])
    if not cap.isOpened():
        print(f"无法打开摄像头 {selected_camera[0]}")
        sys.exit(1)
    
    print(f"已成功打开摄像头 {selected_camera[0]}")
    return cap

# 初始化摄像头
cap = select_camera_gui()

print("摄像头已打开，按 'q' 键退出...")

# 获取屏幕分辨率
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# 创建窗口（非全屏）
cv2.namedWindow('YOLO实时检测', cv2.WINDOW_NORMAL)
# 设置窗口大小为1280x720，更大的显示窗口
cv2.resizeWindow('YOLO实时检测', 1280, 720)

# 帧率控制
import time
last_time = time.time()
fps = 30  # 目标帧率
frame_delay = 1.0 / fps

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频流，程序即将退出...")
        break
    
    # 预处理：调整尺寸以适应窗口
    frame = cv2.resize(frame, (1280, 720))
    
    height, width, channels = frame.shape

    # 降低输入分辨率以提高速度并减少资源占用
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (128, 128), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            if len(detection) < 6:
                continue
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 提高置信度阈值减少计算量
            if confidence > 0.7:
                try:
                    center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                    w, h = int(detection[2]*width), int(detection[3]*height)
                    x, y = int(center_x - w/2), int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                except (IndexError, ValueError) as e:
                    continue

    # 提高NMS阈值减少重叠框计算
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f'{classes[class_ids[i]]} {confidences[i]:.2f}'
            
            # 模拟距离计算 (这里使用物体宽度的倒数作为距离近似值)
            # 在实际应用中，需要根据相机参数和物体实际大小进行计算
            distance_cm = int(1000 / (w + 1))  # 简单模拟，实际应用需要更复杂的计算
            
            # 绘制边框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 绘制标签
            cv2.putText(frame, label, (x, y-10), font, 0.9, (0, 255, 0), 2)
            
            # 在边框右侧显示距离
            distance_text = f'{distance_cm} cm'
            text_size = cv2.getTextSize(distance_text, font, 0.9, 2)[0]
            text_x = x + w + 10
            text_y = y + h // 2 + text_size[1] // 2
            cv2.putText(frame, distance_text, (text_x, text_y), font, 0.9, (0, 255, 0), 2)

    cv2.imshow('YOLO实时检测', frame)
    
    # 帧率控制
    current_time = time.time()
    elapsed = current_time - last_time
    if elapsed < frame_delay:
        time.sleep(frame_delay - elapsed)
    last_time = time.time()
    
    # 检测窗口关闭事件或按'q'键退出
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('YOLO实时检测', cv2.WND_PROP_VISIBLE) < 1):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("程序已正常关闭")
print("程序已退出")


input("请输入任意字符结束")


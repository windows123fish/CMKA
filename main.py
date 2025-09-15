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
    # 尝试使用不同的YOLO模型配置
    yolo_config = 'yolov3.cfg'
    yolo_weights = 'yolov3.weights'
    net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
except cv2.error as e:
    print(f"模型加载失败: {str(e)}")
    print("详细错误信息:\n", traceback.format_exc())
    print("可能原因: 权重文件与配置文件不匹配或OpenCV版本不兼容")
    sys.exit(1)

# 设置OpenCV DNN后端优化和多线程
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cv2.setNumThreads(8)  # 使用8个CPU线程加速

# 原神风格的加载动画函数 - 黑色背景点击进入版本
def loading_animation():
    cv2.namedWindow('加载中', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('加载中', 800, 500)
    
    # 设置鼠标点击回调函数
    clicked = False
    is_loading_complete = False
    
    def on_mouse_click(event, x, y, flags, param):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN and is_loading_complete:
            clicked = True
    
    cv2.setMouseCallback('加载中', on_mouse_click)
    
    # 加载动画参数
    frame_count = 0
    total_frames = 180  # 总帧数，减少使动画更快
    
    # 加载状态列表 - 类似原神的加载提示
    loading_states = [
        "正在初始化系统...",
        "正在加载着色器资源...",
        "正在编译模型参数...",
        "正在优化渲染管线...",
        "正在配置检测算法...",
        "正在准备摄像头资源..."
    ]
    
    # 进度条参数 - 两头尖中间白色的风格
    bar_width = 500
    bar_height = 10
    bar_x = (800 - bar_width) // 2
    bar_y = 350
    
    # 定义停顿点（百分比）
    pause_points = [20, 40, 70, 90]
    pause_duration = 9  # 每个停顿点持续的帧数，减少使停顿更短
    current_pause = None
    pause_timer = 0
    
    # 网络梗列表 - 防止用户无聊
    fun_facts = [
        "程序员常说：这段代码不会有问题的...",
        "加载中...你知道吗？这个软件由1000+行代码构成",
        "检测算法正在训练，比你减肥还努力",
        "进度条：我动得很慢，但我真的在努力",
        "据说耐心等待的人运气都不会太差",
        "系统提示：给程序员加个鸡腿，加载速度+10%",
        "你知道吗？TAWSCamDetect的TAWS是\"智能分析\"的意思",
        "正在优化性能，让你的电脑跑得比兔子还快",
        "摄像头准备中...不要紧张，它不会拍照的",
        "我：老妈，我要吃豆角！老妈：我看你像豆角",
        "加载完成后，你将获得超能力：识别物体！"
    ]
    current_fun_fact_index = 0
    fun_fact_change_interval = 20  # 每20帧切换一个梗
    
    # 递减数字动画参数
    number_animation_values = ["正在加载：yolov3-tiny.cfg", "正在加载：coco.names", "", "正在加载：yolov3-tiny.weights", "正在加载：yolov3.weights", ""]  # 连贯的输出样式
    current_number_index = 0
    number_change_interval = 25  # 每25帧切换一个数字
    number_change_timer = 0
    
    while True:
        # 创建黑色背景
        frame = np.zeros((500, 800, 3), dtype=np.uint8)
        
        # 绘制标题文字
        title_color = (148, 163, 252)  # 原神风格的蓝色，适应黑色背景
        frame = put_chinese_text(frame, "TAWSCamDetect", (300, 60), font_size=32, color=title_color)
        
        # 根据当前进度选择加载状态文本
        progress = frame_count / total_frames
        state_index = min(int(progress * len(loading_states)), len(loading_states) - 1)
        loading_text = loading_states[state_index]
        
        # 绘制加载提示文字
        text_color = (200, 200, 200)  # 浅灰色文字，适应黑色背景
        frame = put_chinese_text(frame, loading_text, (240, 280), font_size=20, color=text_color)
        
        # 绘制提示点击的文字 - 只有加载完成后才显示
        if is_loading_complete:
            click_text_color = (148, 163, 252)  # 蓝色文字
            frame = put_chinese_text(frame, "点击屏幕进入", (320, 420), font_size=16, color=click_text_color)
        
        # 绘制两头尖中间白色的进度条
        # 进度条背景
        bg_color = (50, 50, 50)  # 深灰色背景，适应黑色背景
        # 绘制进度条轮廓（两头尖的形状）
        bar_points = [
            (bar_x, bar_y + bar_height // 2),
            (bar_x + bar_width, bar_y + bar_height // 2),
            (bar_x + bar_width - bar_height // 2, bar_y + bar_height),
            (bar_x + bar_height // 2, bar_y + bar_height),
            (bar_x, bar_y + bar_height // 2),
            (bar_x + bar_height // 2, bar_y),
            (bar_x + bar_width - bar_height // 2, bar_y),
            (bar_x + bar_width, bar_y + bar_height // 2)
        ]
        cv2.fillPoly(frame, [np.array(bar_points)], bg_color)
        
        # 绘制进度条填充（白色）
        filled_width = int(bar_width * progress)
        if filled_width > 0:
            # 计算填充区域的点
            fill_points = [
                (bar_x, bar_y + bar_height // 2),
                (bar_x + filled_width, bar_y + bar_height // 2),
                (min(bar_x + filled_width, bar_x + bar_width - bar_height // 2), bar_y + bar_height),
                (bar_x + bar_height // 2, bar_y + bar_height),
                (bar_x, bar_y + bar_height // 2),
                (bar_x + bar_height // 2, bar_y),
                (min(bar_x + filled_width, bar_x + bar_width - bar_height // 2), bar_y),
                (bar_x + filled_width, bar_y + bar_height // 2)
            ]
            cv2.fillPoly(frame, [np.array(fill_points)], (255, 255, 255))  # 白色填充
        
        # 进度条停顿逻辑（不显示具体百分比数字）
        if current_pause is None:
            # 计算当前进度对应的百分比值用于停顿判断
            pause_check_percent = int(progress * 100)
            
            # 检查是否需要停顿
            for pause_point in pause_points:
                if pause_check_percent >= pause_point and frame_count < total_frames - pause_duration:
                    current_pause = pause_point
                    pause_timer = 0
                    break
        else:
            # 处于停顿状态
            pause_timer += 1
            if pause_timer >= pause_duration:
                current_pause = None
        
        # 更新和显示网络梗文字
        if frame_count % fun_fact_change_interval == 0:
            current_fun_fact_index = (current_fun_fact_index + 1) % len(fun_facts)
        
        # 绘制网络梗文字 - 在进度条下方
        fun_fact_color = (180, 180, 180)  # 浅灰色文字
        fun_fact_y = bar_y + 40  # 在进度条下方
        frame = put_chinese_text(frame, fun_facts[current_fun_fact_index], (bar_x, fun_fact_y), font_size=14, color=fun_fact_color)
        
        # 绘制递减数字动画 - 在梗文字下方
        number_color = (150, 150, 150)  # 比梗文字稍暗的灰色
        number_y = fun_fact_y + 25  # 在梗文字下方
        
        # 更新数字动画
        number_change_timer += 1
        if number_change_timer >= number_change_interval:
            current_number_index = (current_number_index + 1) % len(number_animation_values)
            number_change_timer = 0
        
        # 显示当前数字
        frame = put_chinese_text(frame, str(number_animation_values[current_number_index]), (bar_x, number_y), font_size=14, color=number_color)
        
        # 底部版本信息
        frame = put_chinese_text(frame, "v1.0.0", (680, 465), font_size=12, color=(120, 120, 120))
        
        # 检查加载是否完成
        if frame_count >= total_frames:
            is_loading_complete = True
        
        # 显示动画
        cv2.imshow('加载中', frame)
        
        # 增加帧计数（如果不在停顿时）
        if current_pause is None and frame_count < total_frames:
            frame_count += 1
        
        # 检查点击事件、按键事件或窗口关闭
        key = cv2.waitKey(25) & 0xFF
        if clicked or key == ord('q') or cv2.getWindowProperty('加载中', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # 确保窗口正确关闭
    cv2.destroyAllWindows()

# 显示加载动画并标记已运行
loading_animation_ran = False
if not loading_animation_ran:
    loading_animation()
    loading_animation_ran = True

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
# 设置窗口大小为800x600
cv2.resizeWindow('YOLO实时检测', 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频流，程序即将退出...")
        break
    
    # 预处理：调整尺寸以适应窗口
    frame = cv2.resize(frame, (800, 600))
    
    height, width, channels = frame.shape

    # 进一步降低输入分辨率以提高速度
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (160, 160), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 提高置信度阈值减少计算量
            if confidence > 0.7:
                center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                w, h = int(detection[2]*width), int(detection[3]*height)
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

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
    
    # 检测窗口关闭事件或按'q'键退出
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('YOLO实时检测', cv2.WND_PROP_VISIBLE) < 1):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("程序已正常关闭")
print("程序已退出")


input("请输入任意字符结束")


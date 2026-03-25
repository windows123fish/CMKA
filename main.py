import cv2
import os
import urllib.request
import shutil
import numpy as np
import sys
import traceback
import ctypes
import webbrowser
import pyperclip
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

try:
    import pkg_resources.py2_warn
except ImportError:
    pass

try:
    from PIL import Image, ImageDraw, ImageFont
    pil_available = True
except ImportError:
    print("未找到PIL库，请安装: pip install pillow")
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


yolo_config = 'yolov3-tiny.cfg'
yolo_weights = 'yolov3-tiny.weights'
classes_file = 'coco.names'


def download_file(url, filename, fallback_urls=None, min_expected_size=0):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    req = urllib.请求.Request(url, headers=headers)
    try:
        with urllib.请求.urlopen(req) as response:
            content_length = response.getheader('Content-Length')
            if content_length and min_expected_size > 0:
                if int(content_length) < min_expected_size:
                    print(f"文件大小不符合预期 {url}: {content_length}字节 < {min_expected_size}字节")
                    return False
            
            total_size = int(content_length) if content_length else 0
            downloaded = 0
            chunk_size = 8192
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.percentage:>3.0f}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                expand=True,
            ) as progress:
                task = progress.add_task(f"[cyan]正在下载: {filename}", total=total_size)
                
                with open(filename, 'wb') as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        
                        out_file.write(chunk)
                        chunk_size_bytes = len(chunk)
                        downloaded += chunk_size_bytes
                        progress.update(task, advance=chunk_size_bytes)
            
            print(f"下载成功: {filename}")
            
            if not os.path.exists(filename):
                print(f"警告: 文件下载成功但未找到 {filename}!")
                return False
            
            file_size = os.path.getsize(filename)
            if min_expected_size > 0 and file_size < min_expected_size:
                print(f"文件大小不符合预期 {filename}: {file_size}字节 < {min_expected_size}字节")
                os.remove(filename)
                return False
            
            return True
    except (urllib.error.HTTPError, OSError) as e:
        print(f"\n下载或文件写入失败 {url}: {str(e)}")
        if fallback_urls and len(fallback_urls) > 0:
            next_url = fallback_urls.pop(0)
            print(f"尝试备用链接: {next_url}")
            return download_file(next_url, filename, fallback_urls, min_expected_size)
        return False


if not os.path.exists(yolo_config):
    print(f"正在下载{yolo_config}...")
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

if not os.path.exists(yolo_weights):
    print("==============================================")
    print("权重文件下载失败: 所有自动下载链接均不可用")
    print("请手动下载权重文件并放置到项目目录:")
    print("1. 访问: https://pjreddie.com/media/files/yolov3-tiny.weights")
    print("2. 将文件保存为: yolov3-tiny.weights")
    print("3. 确保文件大小约为34MB")
    print("===============================================")
    print("所有下载链接均失败，请手动下载权重文件并放置到项目目录")
    print("推荐下载地址1: https://pjreddie.com/media/files/yolov3-tiny.weights")
    print("推荐下载地址2: https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov3-tiny.pt")
    print("下载后请重命名为'yolov3-tiny.weights'并放在当前目录")
    sys.exit(1)

if not os.path.exists(yolo_weights):
    print(f"下载错误: {yolo_weights}文件不存在")
    sys.exit(1)

try:
    weights_size = os.path.getsize(yolo_weights)
except FileNotFoundError:
    print(f"致命错误: {yolo_weights}文件在验证过程中丢失")
    sys.exit(1)

if weights_size < 1*1024*1024:
    print(f"权重文件{yolo_weights}损坏或不完整 (大小: {weights_size}字节)")
    print("请尝试手动下载: https://pjreddie.com/media/files/yolov3-tiny.weights")
    os.remove(yolo_weights)
    sys.exit(1)

if os.path.getsize(yolo_config) < 1024:
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

try:
    net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
except cv2.error as e:
    print(f"模型加载失败: {str(e)}")
    print("详细错误信息:\n", traceback.format_exc())
    print("可能原因: 权重文件与配置文件不匹配或OpenCV版本不兼容")
    sys.exit(1)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cv2.setNumThreads(4)


def verify_trial_code():
    cv2.namedWindow('使用码验证', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('使用码验证', 500, 450)
    
    correct_code = "Windows123fish"
    max_attempts = 5
    attempts = 0
    user_input = ""
    show_error = False
    cursor_visible = True
    cursor_blink_timer = 0
    
    button_clicked = False
    
    def on_mouse_click(event, x, y, flags, param):
        nonlocal button_clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if 410 <= x <= 470 and 230 <= y <= 280:
                button_clicked = True
    
    cv2.setMouseCallback('使用码验证', on_mouse_click)
    # 主循环
    while attempts < max_attempts:
        frame = np.zeros((450, 500, 3), dtype=np.uint8)
        
        draw_rounded_rectangle(frame, (0, 0), (500, 80), (255, 182, 193), radius=15)
        draw_rounded_rectangle(frame, (0, 80), (500, 450), (255, 255, 255), radius=15)
        
        frame = put_chinese_text(frame, "使用码验证", (30, 25), font_size=24, color=(255, 255, 255))
        
        cv2.circle(frame, (460, 30), 20, (255, 192, 203), -1)
        frame = put_chinese_text(frame, "×", (452, 15), font_size=24, color=(255, 255, 255))
        
        draw_rounded_rectangle(frame, (40, 100), (460, 170), (224, 255, 255), radius=10)
        cv2.rectangle(frame, (40, 100), (460, 170), (176, 224, 230), 2)
        
        frame = put_chinese_text(frame, "重要提示：本软件为免费软件，如果您是购买的，请", (50, 120), font_size=14, color=(0, 102, 204))
        frame = put_chinese_text(frame, "尽快联系退款与举报！", (50, 145), font_size=14, color=(0, 102, 204))
        
        frame = put_chinese_text(frame, "请输入使用码以继续使用本软件:", (30, 200), font_size=16, color=(64, 64, 64))
        
        draw_rounded_rectangle(frame, (50, 230), (400, 280), (255, 182, 193), radius=10)
        cv2.rectangle(frame, (50, 230), (400, 280), (255, 240, 245), -1)
        
        masked_input = "*" * len(user_input)
        frame = put_chinese_text(frame, masked_input if masked_input else "请输入使用码", (60, 255), font_size=16, color=(128, 128, 128))
        
        cursor_blink_timer += 1
        if cursor_visible and cursor_blink_timer % 20 < 10:
            cursor_x = 60 + len(user_input) * 12
            cv2.line(frame, (cursor_x, 255), (cursor_x, 275), (0, 0, 0), 2)
        
        draw_rounded_rectangle(frame, (410, 230), (470, 280), (255, 105, 180), radius=10)
        cv2.rectangle(frame, (410, 230), (470, 280), (255, 69, 0), 2)
        frame = put_chinese_text(frame, "验证", (425, 255), font_size=14, color=(255, 255, 255))
        
        if show_error:
            frame = put_chinese_text(frame, "使用码错误，请重试", (180, 310), font_size=14, color=(255, 0, 0))
            frame = put_chinese_text(frame, f"剩余尝试次数：{max_attempts - attempts}", (180, 335), font_size=12, color=(255, 0, 0))
        
        frame = put_chinese_text(frame, "请使用enter(回车)键确认", (30, 380), font_size=14, color=(147, 112, 219))
        frame = put_chinese_text(frame, "使用码：Windows123fish", (30, 410), font_size=14, color=(147, 112, 219))
        cv2.imshow('使用码验证', frame)
        
        key = cv2.waitKey(25) & 0xFF
        
        if key == 13:
            if user_input == correct_code:
                success_frame = np.zeros((450, 500, 3), dtype=np.uint8)
                cv2.rectangle(success_frame, (0, 0), (500, 80), (255, 182, 193), -1)
                cv2.rectangle(success_frame, (0, 80), (500, 450), (255, 255, 255), -1)
                frame = put_chinese_text(success_frame, "使用码验证", (30, 25), font_size=24, color=(255, 255, 255))
                success_color = (0, 255, 0)
                success_frame = put_chinese_text(success_frame, "验证成功！", (175, 200), font_size=32, color=success_color)
                success_frame = put_chinese_text(success_frame, "欢迎使用本软件", (175, 250), font_size=20, color=(64, 64, 64))
                cv2.imshow('使用码验证', success_frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                return True
            else:
                attempts += 1
                show_error = True
                user_input = ""
                button_clicked = False
                if attempts >= max_attempts:
                    fail_frame = np.zeros((450, 500, 3), dtype=np.uint8)
                    cv2.rectangle(fail_frame, (0, 0), (500, 80), (255, 182, 193), -1)
                    cv2.rectangle(fail_frame, (0, 80), (500, 450), (255, 255, 255), -1)
                    frame = put_chinese_text(fail_frame, "使用码验证", (30, 25), font_size=24, color=(255, 255, 255))
                    fail_color = (255, 0, 0)
                    fail_frame = put_chinese_text(fail_frame, "验证失败", (200, 200), font_size=32, color=fail_color)
                    fail_frame = put_chinese_text(fail_frame, "尝试次数已用完", (170, 260), font_size=20, color=(64, 64, 64))
                    cv2.imshow('使用码验证', fail_frame)
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
                    return False
        elif key == 8:
            user_input = user_input[:-1]
            show_error = False
        elif key == 22:
            try:
                pasted_text = pyperclip.paste()
                user_input += pasted_text.replace('\n', '').replace('\r', '')
                show_error = False
            except:
                pass
        elif key == 27:
            cv2.destroyAllWindows()
            return False
        elif 32 <= key <= 126:
            user_input += chr(key)
            show_error = False
        elif key == ord('q') or cv2.getWindowProperty('使用码验证', cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return False
    
    cv2.destroyAllWindows()
    return False


if not verify_trial_code():
    print("验证失败，程序退出")
    sys.exit(1)

print("验证成功，欢迎使用！")


with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def get_camera_name_windows():
    camera_names = []
    
    try:
        import win32com.client
        import pythoncom
        
        pythoncom.CoInitialize()
        devicelist = win32com.client.Dispatch("WIA.DeviceManager").DeviceInfos
        
        for device_info in devicelist:
            if device_info.类型 == 3:  # VideoDeviceType
                camera_names.append(device_info.Properties("Name").Value)
        
        pythoncom.CoUninitialize()
        
        if camera_names:
            return camera_names
    except:
        pass
    
    try:
        import dshowcapture
        cameras = dshowcapture.get_video_devices()
        camera_names = [camera.name for camera in cameras]
        
        if camera_names:
            return camera_names
    except:
        pass
    
    try:
        import pywintypes
        import win32com.client as win32
        
        wmi = win32.Dispatch("WbemScripting.SWbemLocator")
        service = wmi.ConnectServer(".", "root\\cimv2")
        devices = service.ExecQuery("SELECT * FROM Win32_PnPEntity WHERE Name LIKE '%Camera%' OR Name LIKE '%Webcam%' OR Name LIKE '%Video%'")
        
        camera_names = []
        for device in devices:
            if device.名字:
                camera_names.append(device.名字)
        
        if camera_names:
            return camera_names
    except:
        pass
    
    try:
        import subprocess
        result = subprocess.run(['powershell', '-Command', 'Get-PnpDevice -Class Camera | Select-Object FriendlyName'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            camera_names = [line.strip() for line in lines if line.strip() and 'FriendlyName' not in line]
            
            if camera_names:
                return camera_names
    except:
        pass
    
    return None


def select_camera_gui():
    camera_names = get_camera_name_windows()
    
    max_cameras = 10
    available_cameras = []
    camera_info = []
    
    print("正在检测可用摄像头...")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            
            try:
                name = cap.get(cv2.CAP_PROP_DEVICE_NAME)
                if name is None or name == 0 or name == "":
                    name = None
                else:
                    name = str(name)
                    try:
                        name = name.encode('latin1').decode('gbk')
                    except:
                        try:
                            name = name.encode('latin1').decode('utf-8')
                        except:
                            pass
            except:
                name = None
            
            if name is None and camera_names and i < len(camera_names):
                name = camera_names[i]
            
            try:
                backend = cap.get(cv2.CAP_PROP_BACKEND)
                backend_name = {
                    cv2.CAP_ANY: "Any",
                    cv2.CAP_VFW: "VFW",
                    cv2.CAP_V4L: "V4L",
                    cv2.CAP_V4L2: "V4L2",
                    cv2.CAP_FIREWIRE: "FireWire",
                    cv2.CAP_IEEE1394: "IEEE1394",
                    cv2.CAP_QT: "QT",
                    cv2.CAP_GSTREAMER: "GStreamer",
                    cv2.CAP_FFMPEG: "FFMPEG",
                    cv2.CAP_DSHOW: "DirectShow",
                    cv2.CAP_MSMF: "Media Foundation",
                    cv2.CAP_WINRT: "Windows Runtime"
                }.get(backend, f"Backend {backend}")
                
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                res_info = f" ({int(width)}x{int(height)}"
                if fps > 0:
                    res_info += f" @ {int(fps)}fps"
                res_info += ")"
                
                if name:
                    display_name = f"{name}"
                else:
                    display_name = f"{backend_name}"
            except:
                display_name = f"摄像头 {i}"
                res_info = ""
            
            camera_info.append(f"{i}: {display_name}{res_info}")
            cap.release()
            print(f"发现摄像头 {i}: {display_name}{res_info}")
    
    if not available_cameras:
        print("未找到可用摄像头")
        sys.exit(1)
    
    height = 100 + len(available_cameras) * 40
    width = 500
    
    cv2.namedWindow("摄像头选择")
    cv2.resizeWindow("摄像头选择", width, height)
    selected_camera = [available_cameras[0]]
    
    def draw_window():
        height = 100 + len(available_cameras) * 40
        width = 500
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        draw_rounded_rectangle(img, (0, 0), (width, 80), (255, 182, 193), radius=15)
        img = put_chinese_text(img, "摄像头选择", (30, 25), font_size=24, color=(255, 255, 255))
        
        cv2.circle(img, (460, 40), 20, (255, 192, 203), -1)
        img = put_chinese_text(img, "×", (452, 25), font_size=24, color=(255, 255, 255))
        
        for i, info in enumerate(camera_info):
            y_pos = 100 + i * 40
            color = (224, 255, 255) if i == selected_camera[0] else (255, 255, 255)
            draw_rounded_rectangle(img, (50, y_pos), (450, y_pos + 35), color, radius=8)
            
            if i == selected_camera[0]:
                cv2.rectangle(img, (50, y_pos), (450, y_pos + 35), (176, 224, 230), 2)
            
            img = put_chinese_text(img, info, (70, y_pos + 10), font_size=16, color=(64, 64, 64))
        
        return img
    
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(len(available_cameras)):
                y_pos = 100 + i * 40
                if 50 <= x <= 450 and y_pos <= y <= y_pos + 35:
                    selected_camera[0] = i
                    break
    
    cv2.setMouseCallback("摄像头选择", on_mouse_click)
    
    while True:
        frame = draw_window()
        cv2.imshow("摄像头选择", frame)
        cv2.resizeWindow("摄像头选择", 500, frame.shape[0])
        
        if cv2.getWindowProperty("摄像头选择", cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            sys.exit(1)
        
        key = cv2.waitKey(25) & 0xFF
        
        if key == 13:
            cv2.destroyAllWindows()
            return available_cameras[selected_camera[0]]
        elif key == 27:
            cv2.destroyAllWindows()
            return None
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(1)


camera_id = select_camera_gui()
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"无法打开摄像头 {camera_id}")
    sys.exit(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                frame = put_chinese_text(frame, label, (x, y - 10), font_size=14, color=(0, 255, 0))
        
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

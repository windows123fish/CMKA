# CMKA - Real-time Object Detection Application

<div align="center">

![软件图标](软件图标.png)

<br>
<a href="README.md">简体中文</a> ｜
<a href="README_en.md">English</a> ｜
<a href="README_zh-TW.md">繁體中文</a> ｜
<a href="README_ja.md">日本語</a> ｜
<a href="README_ru.md">Русский</a>

<br>
<br>

<img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python Version">
<img src="https://img.shields.io/badge/PyQt5-5.15+-green.svg" alt="PyQt5">
<img src="https://img.shields.io/badge/OpenCV-4.8+-orange.svg" alt="OpenCV">
<img src="https://img.shields.io/badge/YOLO-26n-red.svg" alt="YOLO">

</div>

CMKA is a real-time object detection application based on **YOLO26n** and **OpenCV DNN**, with an elegant GUI built using PyQt5.

## Key Features

- ✅ **Real-time Camera Detection** - Support multiple camera switching
- ✅ **80 Object Categories** - Including person, vehicle, animal, daily items, food, etc.
- ✅ **Chinese Label Display** - Clear Chinese category names using PIL
- ✅ **Category Filtering** - Customizable category exclusion
- ✅ **Elegant UI** - Custom title bar and rounded window design
- ✅ **Window Controls** - Minimize, maximize, and drag operations

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.12+ | Programming Language |
| PyQt5 | 5.15+ | GUI Framework |
| OpenCV | 4.8+ | Computer Vision, DNN Inference |
| PIL/Pillow | 10.0+ | Chinese Text Rendering |
| NumPy | 1.24+ | Numerical Computing |

## Quick Start

```bash
# Install dependencies
pip install PyQt5 opencv-python pillow numpy

# Run application
python main.py
```

## Usage

1. Launch the application
2. Enter verification code `Windows123fish`
3. Select available camera from the dialog
4. Click "Start Detection" to begin real-time detection
5. Click "Disable Categories" to exclude specific objects

## System Requirements

- Windows 10/11 (64-bit)
- Python 3.12+
- Minimum 4GB RAM

---

*Built with PyQt5 & OpenCV*

# CMKA - 实时目标检测应用

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

CMKA 是一款基于 **YOLO26n** 和 **Ultralytics** 的实时目标检测应用程序，使用 PyQt5 构建优雅的图形用户界面。

## 主要特性

- ✅ **实时摄像头检测** - 支持多摄像头切换，实时显示检测结果
- ✅ **80种目标识别** - 涵盖人物、车辆、动物、日常物品、食物等COCO类别
- ✅ **中文标签显示** - 使用 PIL 渲染清晰的中文类别名称
- ✅ **类别过滤** - 可自定义禁用特定检测类别
- ✅ **目标追踪** - 自动追踪检测目标，显示唯一ID
- ✅ **轨迹绘制** - 记录并绘制目标运动轨迹（人脸除外）
- ✅ **位置预测** - 预测目标下一帧位置（人脸除外）
- ✅ **轨迹设置** - 可自定义轨迹线和预测点的显示与颜色
- ✅ **优雅界面** - 自定义标题栏、圆角窗口设计
- ✅ **窗口控制** - 支持最小化、最大化、拖动操作

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 编程语言 |
| PyQt5 | 5.15+ | 图形用户界面 |
| Ultralytics | 8.0+ | YOLO模型推理 |
| OpenCV | 4.8+ | 计算机视觉 |
| PIL/Pillow | 10.0+ | 中文文字渲染 |
| NumPy | 1.24+ | 数值计算 |

## 快速开始

```bash
# 安装依赖
pip install PyQt5 opencv-python pillow numpy ultralytics

# 运行应用
python main.py
```

## 使用说明

1. 启动应用程序
2. 输入验证码 `Windows123fish` 进行验证
3. 在弹出的对话框中选择可用摄像头
4. 点击"开始检测"按钮启动实时检测
5. 点击"禁用类别"可排除特定检测目标
6. 点击"轨迹设置"可配置轨迹显示和颜色

## 轨迹设置

- **显示轨迹线** - 勾选后显示目标运动轨迹
- **显示预测位置** - 勾选后显示预测的下一帧位置（黄色圆点）
- **轨迹线颜色** - 支持红色、蓝色、绿色、黄色、紫色、白色
- **预测位置颜色** - 支持黄色、红色、蓝色、绿色、紫色、白色

> **注意**：人脸（person）检测不显示轨迹和预测位置

## 系统要求

- Windows 10/11 (64-bit)
- Python 3.12+
- 至少 4GB RAM

---

*基于 PyQt5 & YOLO26n 构建*
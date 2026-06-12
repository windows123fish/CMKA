# CMKA - 实时目标检测应用
## 项目简介

CMKA 是一款基于 **YOLO26n** 和 **OpenCV DNN** 的实时目标检测应用程序，使用 PyQt5 构建图形界面，支持对摄像头实时画面中的 80 种 COCO 类别目标进行检测与识别。

## 功能特性

-  **实时摄像头检测** - 支持多摄像头切换
-  **类别过滤** - 可自定义禁用特定检测类别
-  **中文标签显示** - 使用 PIL 渲染中文类别名称
-  **优雅的图形界面** - 自定义标题栏和圆角窗口
-  **窗口控制** - 支持最小化、最大化、拖动操作

## 支持的检测类别

| 类别类型 | 示例 |
|---------|------|
| **人物** | person |
| **交通工具** | bicycle, car, motorcycle, bus, truck |
| **动物** | bird, cat, dog, horse, cow, elephant, bear |
| **日常物品** | backpack, umbrella, handbag, laptop, mouse, keyboard |
| **食物** | banana, apple, sandwich, pizza, cake |
| **家具** | chair, sofa, bed, diningtable, toilet |

共支持 **80 种** COCO 数据集类别。

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.12 | 编程语言 |
| PyQt5 | 5.15+ | 图形用户界面 |
| OpenCV | 4.8+ | 计算机视觉、DNN推理 |
| PIL/Pillow | 10.0+ | 中文文字渲染 |
| NumPy | 1.24+ | 数值计算 |

**注意**：本项目使用 OpenCV DNN 模式运行，**无需安装 PyTorch**。

## 项目结构

```
CMKA/
├── main.py          # 主应用程序
├── main.spec        # PyInstaller 打包配置
├── yolo26.cfg       # YOLO 模型配置文件
├── yolo26n.pt       # YOLO26n 预训练模型
└── README.md        # 项目文档
```

## 快速开始

### 环境要求

- Windows 10/11 (64-bit)
- Python 3.12+
- 至少 4GB RAM

### 安装依赖

```bash
pip install PyQt5 opencv-python pillow numpy
```

### 运行应用

```bash
python main.py
```

### 使用说明

1. **启动程序** - 运行 `main.py` 或 `cmka_app.exe`
2. **验证使用码** - 输入 `Windows123fish` 进行验证
3. **选择摄像头** - 在弹出的对话框中选择可用摄像头
4. **开始检测** - 点击"开始检测"按钮启动实时检测
5. **管理类别** - 点击"禁用类别"可排除特定检测目标

### 打包为可执行文件

```bash
pip install pyinstaller
pyinstaller --noconfirm main.spec
```

打包后的可执行文件位于 `dist/cmka_app/` 目录。

## 操作指南

| 按钮 | 功能 |
|------|------|
| 开始检测 | 启动实时目标检测 |
| 停止检测 | 停止当前检测 |
| 切换摄像头 | 切换到其他可用摄像头 |
| 禁用类别 | 打开类别管理对话框 |

## 常见问题

**Q: 程序启动失败，提示找不到摄像头？**

A: 请确保摄像头已正确连接，并在设备管理器中检查驱动状态。

**Q: 检测画面显示异常？**

A: 尝试切换不同的摄像头或调整摄像头分辨率设置。

**Q: 如何打包应用程序？**

A: 使用 `pyinstaller main.spec` 命令进行打包。

## 许可证
使用MIT许可证
[MIT许可证正文]
MIT License

Copyright (c) 2026 Windows_123_fish

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



[MIT许可证译文]

[*译文部分内容可能为AI或机翻]

MIT 许可证
版权所有 (c) 2026 Windows_123_fish
特此授予任何获得本软件及相关文档文件（以下简称“软件”）副本的人，免费、无限制地处理本软件的权利，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或销售本软件的副本，并允许获得本软件的人员在遵守以下条件的前提下这样做：
所有副本或实质性部分均需包含上述版权声明和本许可声明。
本软件按“原样”提供，不提供任何明示或暗示的担保，包括但不限于适销性、特定用途适用性和非侵权性的担保。在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任负责，无论这些责任是基于合同、侵权或其他原因，也不管是否与本软件有关或由本软件的使用或其他交易引起。 

# CMKA - 即時目標檢測應用

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

CMKA 是一款基於 **YOLO26n** 和 **Ultralytics** 的即時目標檢測應用程式，使用 PyQt5 構建優雅的圖形用戶界面。

## 主要特性

- ✅ **即時攝像頭檢測** - 支持多攝像頭切換，即時顯示檢測結果
- ✅ **80種目標識別** - 涵蓋人物、車輛、動物、日常物品、食物等COCO類別
- ✅ **中文標籤顯示** - 使用 PIL 渲染清晰的中文類別名稱
- ✅ **類別過濾** - 可自定義禁用特定檢測類別
- ✅ **目標追蹤** - 自動追蹤檢測目標，顯示唯一ID
- ✅ **軌跡繪製** - 記錄並繪製目標運動軌跡（人物除外）
- ✅ **位置預測** - 預測目標下一幀位置（人物除外）
- ✅ **軌跡設定** - 可自定義軌跡線和預測點的顯示與顏色
- ✅ **模型管理** - 瀏覽並載入自訂YOLO模型
- ✅ **日誌匯出** - 將檢測日誌匯出為CSV檔案
- ✅ **即時統計** - 右侧面板即時顯示檢測數量與各類別統計
- ✅ **攝像頭切換** - 檢測過程中可切換攝像頭
- ✅ **儲存設定** - 持久化目前設定
- ✅ **優雅界面** - 自定義標題欄、圓角窗口設計
- ✅ **窗口控制** - 支持最小化、最大化、拖動操作

## 技術棧

| 組件 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 編程語言 |
| PyQt5 | 5.15+ | 圖形用戶界面 |
| Ultralytics | 8.0+ | YOLO模型推理 |
| OpenCV | 4.8+ | 計算機視覺 |
| PIL/Pillow | 10.0+ | 中文文字渲染 |
| NumPy | 1.24+ | 數值計算 |

## 快速開始

```bash
# 安裝依賴
pip install PyQt5 opencv-python pillow numpy ultralytics

# 執行應用
python main.py
```

## 使用說明

1. 啟動應用程式
2. 輸入驗證碼 `Windows123fish` 進行驗證
3. 在彈出的對話框中選擇可用攝像頭
4. 點擊"開始檢測"按鈕啟動即時檢測
5. 點擊"禁用類別"可排除特定檢測目標
6. 點擊"軌跡設定"可配置軌跡顯示和顏色
7. 點擊"模型管理"瀏覽並載入自訂YOLO模型
8. 點擊"日誌匯出"將檢測日誌匯出為CSV檔案
9. 點擊"儲存設定"持久化目前設定

## 軌跡設定

- **最大遺失幀數** - 目標連續遺失多少幀後不再追蹤
- **追蹤 IoU 閾值** - 調整目標關聯的IoU閾值（0.10 - 0.50）
- **檢測置信度** - 最低檢測置信度（0.10 - 0.90）

> **注意**：人物（person）檢測不顯示軌跡和預測位置

## 系統要求

- Windows 10/11 (64-bit)
- Python 3.12+
- 至少 4GB RAM

---

*基於 PyQt5 & YOLO26n 構建*
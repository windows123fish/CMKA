# CMKA - リアルタイム物体検出アプリケーション

<div align="center">

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

CMKA は **YOLO26n** と **OpenCV DNN** に基づくリアルタイム物体検出アプリケーションで、PyQt5 を使用してエレガントなグラフィカルユーザーインターフェイスを構築しています。

## 主な機能

- ✅ **リアルタイムカメラ検出** - 複数のカメラ切り替えをサポート
- ✅ **80種類の物体識別** - 人物、車両、動物、日用品、食品などのCOCOカテゴリをカバー
- ✅ **中国語ラベル表示** - PILを使用して明確な中国語カテゴリ名をレンダリング
- ✅ **カテゴリフィルタリング** - 特定の検出カテゴリをカスタマイズして無効化可能
- ✅ **エレガントなUI** - カスタムタイトルバーと角丸ウィンドウデザイン
- ✅ **ウィンドウコントロール** - 最小化、最大化、ドラッグ操作をサポート

## 技術スタック

| コンポーネント | バージョン | 用途 |
|----------------|------------|------|
| Python | 3.12+ | プログラミング言語 |
| PyQt5 | 5.15+ | GUIフレームワーク |
| OpenCV | 4.8+ | コンピュータービジョン、DNN推論 |
| PIL/Pillow | 10.0+ | 中国語テキストレンダリング |
| NumPy | 1.24+ | 数値計算 |

## クイックスタート

```bash
# 依存関係のインストール
pip install PyQt5 opencv-python pillow numpy

# アプリケーションの実行
python main.py
```

## 使用方法

1. アプリケーションを起動
2. 認証コード `Windows123fish` を入力して認証
3. ダイアログから利用可能なカメラを選択
4.「検出開始」ボタンをクリックしてリアルタイム検出を開始
5.「カテゴリを無効化」をクリックして特定の物体を除外

## システム要件

- Windows 10/11 (64-bit)
- Python 3.12+
- 最小4GB RAM

---

*PyQt5 & OpenCV で構築*

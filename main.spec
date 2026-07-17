# -*- mode: python ; coding: utf-8 -*-

# ============================================
# 核心：硬编码纯英文Qt路径，彻底绕过中文用户名探测
# ============================================
QT5_ROOT = r"C:\dev\qt5_py38\Lib\site-packages\PyQt5\Qt5"
QT5_SITE_PACKAGES = r"C:\dev\qt5_py38\Lib\site-packages"

block_cipher = None

a = Analysis(
    ['main.py'],
    # 优先加载纯英文路径下的依赖，解决 Foreign Python environment 警告
    pathex=[QT5_SITE_PACKAGES, r'D:\main\CMKA'],
    binaries=[],
    # 手动收集Qt核心依赖，禁止自动路径探测
    datas=[
        # 【必须】Qt窗口插件，没有这个会报「找不到qwindows.dll」
        (f'{QT5_ROOT}\\plugins\\platforms', 'PyQt5/Qt5/plugins/platforms'),
        # 【可选】图片解码插件，用到png/jpg等图片就留着
        (f'{QT5_ROOT}\\plugins\\imageformats', 'PyQt5/Qt5/plugins/imageformats'),
        # 【可选】样式插件，用到自定义Qt样式就留着
        (f'{QT5_ROOT}\\plugins\\styles', 'PyQt5/Qt5/plugins/styles'),
        # Qt核心DLL，必须收集
        (f'{QT5_ROOT}\\bin\\*.dll', '.'),
        # PyQt5自身DLL
        (f'{QT5_SITE_PACKAGES}\\PyQt5\\*.dll', '.'),
    ],
    # 补全PyQt5隐含依赖，避免运行时缺模块
    hiddenimports=[
        'PyQt5.sip',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 排除用不到的模块，解决Qt3D警告+减小体积
    excludes=[
        # Qt无用模块
        'PyQt5.Qt3DCore',
        'PyQt5.Qt3DRender',
        'PyQt5.Qt3DExtras',
        'PyQt5.QtScxml',
        'PyQt5.QtSerialPort',
        # 深度学习无用模块（你之前日志里的警告来源）
        'tensorboard',
        'torch.distributed',
        'torch._inductor',
        'torch._sharded_tensor',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============================================
# OneFile单exe配置（你要的--onefile效果）
# ============================================
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,  # 打包所有二进制文件到单exe
    a.zipfiles,
    a.datas,
    [],
    name='CMKA',  # 输出的exe名称
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # 开启压缩，减小exe体积
    upx_exclude=[],
    runtime_tmpdir=None, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 要加图标的话填路径，比如 r'D:\main\CMKA\icon.ico'
)

# 【注意】OneFile模式不需要COLLECT块！删掉默认的COLLECT部分
# 如果是要onedir文件夹模式，再保留COLLECT，把上面EXE的exclude_binaries改成True
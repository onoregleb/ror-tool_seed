# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('fav.ico', '.'), ('vdk_logo.png', '.'), ('main_form_lay_seed.ui', '.'), ('new_res.ui', '.'), ('params_seed.ui', '.'), ('yolov5/models/modelsweight.pt', '.'), ('yolov5', '.'), ('C:\\Users\\Gleb Onore\\Documents\\VDK\\ror-tool_seed\\.venv\\Lib\\site-packages\\ultralytics', 'ultralytics'), ('yolov5\\hubconf.py', 'yolov5'), ('yolov5/models/modelsweight.pt', 'yolov5/models'), ('check_list.xlsx', '.')],
    hiddenimports=['utils'],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ROR-tool_seed-1_0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['fav.ico'],
)

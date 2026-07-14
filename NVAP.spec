# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata

datas = [('samples', 'samples')]
binaries = []
hiddenimports = ['skimage._shared.geometry', 'imageio.v3']
datas += copy_metadata('imageio')
datas += copy_metadata('nvap')
datas += copy_metadata('torch-directml')
binaries += collect_dynamic_libs('torch_directml')
hiddenimports += collect_submodules('vtkmodules')
hiddenimports += collect_submodules('PySide6')
hiddenimports += collect_submodules('torch_directml')


a = Analysis(
    ['src\\nvap\\app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='NVAP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

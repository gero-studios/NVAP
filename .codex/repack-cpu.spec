import ast
from pathlib import Path

root = Path('C:/Users/giaco/Documents/NVAP/NVAP')
cached = ast.literal_eval((root / 'build/NVAP/PKG-00.toc').read_text())
# Qt imports Windows' ICU shim. A PDF tool on the build machine's PATH supplied
# an incompatible DLL with the same basename; use the OS version at runtime.
entries = [item for item in cached[2] if Path(item[0]).name.lower() != 'icuuc.dll']
exe = EXE(
    entries,
    name='NVAP-0.1.3-windows-cpu',
    icon=str(root / 'assets/nvap.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

"""Check release identity and compiled feature code in the standalone package."""
import marshal
from pathlib import Path
import subprocess
import sys
import types

from PyInstaller.archive.readers import CArchiveReader

root = Path(__file__).resolve().parents[1]
variant = sys.argv[1]
path = root / 'dist' / f'NVAP-0.1.4-windows-{variant}.exe'
archive = CArchiveReader(str(path))
assert not any(Path(name).name.lower() == 'icuuc.dll' for name in archive.toc)
assert any('vtkFiltersGeneral' in name for name in archive.toc)
pyz = archive.open_embedded_archive(next(name for name in archive.toc if name.lower().endswith('.pyz')))
identity = {}
exec(pyz.extract('nvap._build_metadata'), identity)
commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=root, text=True).strip()
assert identity['BUILD_COMMIT'] == commit, identity
assert identity['BUILD_VARIANT'] == variant, identity
version = {}
exec(pyz.extract('nvap'), version)
assert version['__version__'] == '0.1.4', version
metadata_name = next(
    name for name in archive.toc
    if name.lower().startswith(('nvap-', 'nvap.egg-info'))
    and Path(name).name.lower() in {'metadata', 'pkg-info'}
)
assert 'Version: 0.1.4' in archive.extract(metadata_name).decode()

def normalize(code):
    return code.replace(
        co_filename='<release-source>',
        co_consts=tuple(normalize(value) if isinstance(value, types.CodeType) else value for value in code.co_consts),
    )

for module in ('nvap.render.vtk_scene', 'nvap.ui.main_window', 'nvap.ui.control_panel'):
    source = root / 'src' / Path(*module.split('.')).with_suffix('.py')
    expected = compile(source.read_bytes(), str(source), 'exec', dont_inherit=True, optimize=0)
    actual = pyz.extract(module)
    assert normalize(actual) == normalize(expected), f'Bundled source differs: {module}'
print({'variant': variant, 'commit': identity['BUILD_COMMIT'], 'version': version['__version__'],
       'bytes': path.stat().st_size, 'feature_modules_match_source': True})

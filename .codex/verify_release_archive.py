from pathlib import Path
import sys
from PyInstaller.archive.readers import CArchiveReader

variant = sys.argv[1]
path = Path('dist') / f'NVAP-0.1.3-windows-{variant}.exe'
archive = CArchiveReader(str(path))
assert not any(name.lower().endswith('icuuc.dll') for name in archive.toc)
pyz_name = next(name for name in archive.toc if name.lower().endswith('.pyz'))
pyz = archive.open_embedded_archive(pyz_name)
identity = {}
exec(pyz.extract('nvap._build_metadata'), identity)
assert identity['BUILD_COMMIT'] == '24a5dd2', identity
assert identity['BUILD_VARIANT'] == variant, identity
version = {}
exec(pyz.extract('nvap'), version)
assert version['__version__'] == '0.1.3', version
metadata_name = next(name for name in archive.toc if name.lower().startswith('nvap-') and name.lower().endswith('metadata'))
metadata = archive.extract(metadata_name).decode()
assert 'Version: 0.1.3' in metadata
print({'variant': variant, 'commit': identity['BUILD_COMMIT'], 'version': version['__version__'], 'bytes': path.stat().st_size})

"""Point the reused build environments back at the working checkout."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
tracked = subprocess.check_output(
    ['git', 'ls-files', 'src/nvap.egg-info'], cwd=root, text=True,
).splitlines()
snapshots = {root / name: (root / name).read_bytes() for name in tracked}
try:
    for environment in ('.venv', '.venv-gpu'):
        python = root / environment / 'Scripts/python.exe'
        subprocess.run(
            [str(python), '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', str(root)],
            cwd=root, check=True,
        )
        subprocess.run(
            [str(python), '-B', '-c',
             "import nvap; print(nvap.__file__, nvap.__version__); assert nvap.__version__ == '0.1.4'"],
            cwd=root, check=True,
        )
finally:
    for path, contents in snapshots.items():
        path.write_bytes(contents)

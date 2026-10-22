import ast
from pathlib import Path
import pefile

toc = ast.literal_eval(Path('build/NVAP/Analysis-00.toc').read_text())
binaries = [entry for part in toc if isinstance(part, list) for entry in part
            if isinstance(entry, tuple) and len(entry) == 3 and entry[2] in {'BINARY', 'EXTENSION'}]
by_name = {}
for dest, source, _ in binaries:
    by_name.setdefault(Path(dest).name.lower(), []).append((dest, source))
cache = {}
def pe(path):
    if path not in cache:
        value = pefile.PE(path, fast_load=True, max_symbol_exports=100000)
        value.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'], pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
        cache[path] = value
    return cache[path]

queue = [item for item in binaries if item[0].lower() == r'pyside6\qtcore.pyd']
seen = set()
while queue:
    dest, source, _ = queue.pop(0)
    if source in seen:
        continue
    seen.add(source)
    for group in getattr(pe(source), 'DIRECTORY_ENTRY_IMPORT', []):
        name = group.dll.decode().lower()
        candidates = by_name.get(name, [])
        for candidate_dest, candidate_source in candidates:
            exports = getattr(pe(candidate_source), 'DIRECTORY_ENTRY_EXPORT', None)
            available = {symbol.name for symbol in exports.symbols} if exports else set()
            missing = [symbol.name.decode() for symbol in group.imports if symbol.name and symbol.name not in available]
            if missing:
                print('MISSING EXPORTS', dest, 'requires', name, 'from', candidate_dest, missing[:20], flush=True)
            queue.append((candidate_dest, candidate_source, 'BINARY'))
print('Checked', len(seen), 'packaged Qt dependencies.')
for dest, source, _ in binaries:
    if Path(dest).name.lower().startswith(('vcruntime140', 'msvcp140')):
        print(dest, source)

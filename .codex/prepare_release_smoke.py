from pathlib import Path
import numpy as np
import tifffile

root = Path(__file__).parent / "release-smoke-input"
root.mkdir(exist_ok=True)
green = np.zeros((32, 48, 48), dtype=np.uint8)
red = np.zeros_like(green)
green[12:21, 12:22, 12:22] = 255
green[16, 16:19, 20:36] = 255
red[11:22, 28:34, 10:39] = 255
for name, volume in (("Green", green), ("Red", red)):
    channel = root / name
    channel.mkdir(exist_ok=True)
    tifffile.imwrite(channel / "stack.tif", volume, photometric="minisblack")
print(root.resolve())

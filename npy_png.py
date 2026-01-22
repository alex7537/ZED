import numpy as np
import imageio.v2 as imageio

d = np.load("/home/match/ZED/npy_png/depth_meter.npy").astype(np.float32)
d16 = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)

imageio.imwrite("depth.png", d16)
print("saved depth.png", d16.dtype, d16.shape)

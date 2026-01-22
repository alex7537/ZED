import cv2
import numpy as np

rgb_path   = "/home/match/ZED/对齐检测/left_000004.png"   # 改成你的那一帧
depth_path = "/home/match/ZED/对齐检测/depth.png"         # 你的 16-bit depth(mm) png

# 读 RGB
rgb = cv2.imread(rgb_path)  # BGR
assert rgb is not None, f"Cannot read {rgb_path}"

# 读 16-bit depth
depth16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16
assert depth16 is not None, f"Cannot read {depth_path}"
assert depth16.dtype == np.uint16, f"depth dtype is {depth16.dtype}, expected uint16"

# 尺寸必须一致
assert rgb.shape[:2] == depth16.shape[:2], (rgb.shape, depth16.shape)

# 1) 把 depth 压到 8-bit 仅用于边缘检测（不改变分辨率）
d = depth16.astype(np.float32)
d[d == 0] = np.nan  # 0 视为无效
vmin = np.nanpercentile(d, 5)
vmax = np.nanpercentile(d, 95)
d8 = (np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1) * 255).astype(np.uint8)
d8 = np.nan_to_num(d8).astype(np.uint8)

# 2) 边缘
edges = cv2.Canny(d8, 40, 120)

# 3) 叠加：把边缘画成红色
overlay = rgb.copy()
overlay[edges > 0] = (0, 0, 255)  # BGR 红

cv2.imwrite("align_overlay.png", overlay)
print("Saved align_overlay.png")

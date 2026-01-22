import numpy as np
import cv2
import imageio.v2 as imageio

# depth: H×W, float32, 单位 m
depth = np.load("/home/match/ZED/npy_png/depth_meter.npy").astype(np.float32)

# 去除无效值
depth_vis = depth.copy()
depth_vis[~np.isfinite(depth_vis)] = 0
depth_vis[depth_vis <= 0] = 0

# 归一化到 0–255（仅用于可视化，不改变分辨率）
depth_norm = cv2.normalize(
    depth_vis,
    None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX
).astype(np.uint8)

# 伪彩色映射
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

# ❗不用 cv2.imwrite，改用 imageio
imageio.imwrite("depth_color.png", depth_color)

print("saved depth_color.png", depth_color.shape, depth_color.dtype)


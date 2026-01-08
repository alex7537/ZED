import numpy as np
import cv2

# 读入 FS / ZED 深度（单位：米）
depth_m = np.load("/home/match/ZED/npy_png/depth_meter.npy")   # shape: H x W

# 米 → 毫米
depth_mm = (depth_m * 1000.0).astype(np.uint16)

# 保存为 16-bit PNG
cv2.imwrite("depth.png", depth_mm)


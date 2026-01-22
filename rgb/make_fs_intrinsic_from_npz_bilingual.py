# make_fs_intrinsic_from_npz_bilingual.py
# ------------------------------------------------------------
# 作用：从 stereo_calib_rgb.npz 导出 FoundationStereo 所需的内参文件（K + baseline）
# Zweck: Export der für FoundationStereo benötigten Intrinsics-Datei (K + Baseline) aus stereo_calib_rgb.npz
#
# 关键点 / Wichtig:
# - K 使用 rectified 的 P1[:,:3]
# - Baseline 使用 rectified 的 P2 推导：B = -P2[0,3] / P2[0,0]
# ------------------------------------------------------------
import numpy as np
import os

CALIB_FILE = "stereo_calib_rgb.npz"            # 你的 npz
# Ihre NPZ-Datei
FS_ASSETS_DIR = "/home/match/FS/FoundationStereo/assets"
OUT_K_PATH = os.path.join(FS_ASSETS_DIR, "K_d435_2RGB.txt")

def main():
    calib = np.load(CALIB_FILE)

    # rectified 投影矩阵（来自 cv2.stereoRectify）
    # Rektifizierte Projektionsmatrizen (aus cv2.stereoRectify)
    P1 = calib["P1"]  # 3x4 (links)
    P2 = calib["P2"]  # 3x4 (rechts)

    # rectified 左相机内参: 取 P1 的前 3x3
    # Rektifizierte Intrinsics der linken Kamera: P1[:,:3]
    K_rect = P1[:, :3]  # 3x3

    # 基线 (米)：由 P2 的 Tx 推导，保证与 K_rect 同一 rectified 坐标系
    # Baseline (Meter): aus P2.Tx abgeleitet, konsistent zum rektifizierten K_rect
    # OpenCV 典型形式：P2[0,3] = -fx * B  =>  B = -P2[0,3] / fx
    baseline = float(abs(-P2[0, 3] / P2[0, 0]))

    # 展平成 1x9（按行）
    # Zu 1x9 abflachen (zeilenweise)
    K_flat = K_rect.reshape(-1)  # fx, 0, cx, 0, fy, cy, 0, 0, 1

    os.makedirs(FS_ASSETS_DIR, exist_ok=True)
    with open(OUT_K_PATH, "w") as f:
        f.write(" ".join(map(str, K_flat.tolist())) + "\n")
        f.write(str(baseline) + "\n")

    print("写出 FS 内参文件到 / FS-Intrinsics geschrieben nach:", OUT_K_PATH)
    print("K_rect =\n", K_rect)
    print("baseline_rect (m) =", baseline)

if __name__ == "__main__":
    main()

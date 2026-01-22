# capture_rectified_for_fs_bilingual.py
# ------------------------------------------------------------
# 作用：运行时抓取双 RGB 图像，使用 stereo_calib_rgb.npz 做去畸变 + 矫正，并输出 left_rect.png / right_rect.png
# Zweck: Laufzeitaufnahme von Dual-RGB-Bildern, Entzerrung + Rektifizierung mit stereo_calib_rgb.npz,
#        Ausgabe als left_rect.png / right_rect.png
# ------------------------------------------------------------
import os
import cv2
import numpy as np
import pyrealsense2 as rs

# =====================================================
# 配置区
# Konfiguration
LEFT_SERIAL = "YOUR_LEFT_SERIAL"
RIGHT_SERIAL = "YOUR_RIGHT_SERIAL"

W, H = 1280, 720
FPS = 15

CALIB_FILE = "stereo_calib_rgb.npz"  # 需要与标定脚本输出一致
# Muss mit der Ausgabe des Kalibrier-Skripts übereinstimmen

FS_ASSETS_DIR = "/home/match/FS/FoundationStereo/shared_fs_test"
LEFT_OUT_PATH = os.path.join(FS_ASSETS_DIR, "left_rect.png")
RIGHT_OUT_PATH = os.path.join(FS_ASSETS_DIR, "right_rect.png")

# 也可以顺便存一份原始图像，方便调试
# Optional auch Rohbilder speichern (Debug)
DEBUG_OUT_DIR = "fs_dual_rgb_debug"
os.makedirs(DEBUG_OUT_DIR, exist_ok=True)
# =====================================================

def start_rgb(serial: str):
    """启动指定序列号的 RGB 流
    Startet den RGB-Stream für die angegebene Seriennummer
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    pipeline.start(config)
    return pipeline

def main():
    os.makedirs(FS_ASSETS_DIR, exist_ok=True)

    # 1) 读取标定参数
    # 1) Kalibrierparameter laden
    calib = np.load(CALIB_FILE)
    K_left = calib["K_left"]
    dist_left = calib["dist_left"]
    K_right = calib["K_right"]
    dist_right = calib["dist_right"]
    R1 = calib["R1"]
    R2 = calib["R2"]
    P1 = calib["P1"]
    P2 = calib["P2"]

    image_size = (W, H)

    # 2) 生成去畸变 + 矫正的 remap 表
    # 2) Remap-Tabellen für Entzerrung + Rektifizierung erzeugen
    map1x, map1y = cv2.initUndistortRectifyMap(
        K_left, dist_left, R1, P1[:, :3], image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K_right, dist_right, R2, P2[:, :3], image_size, cv2.CV_32FC1
    )

    # 3) 启动相机
    # 3) Kameras starten
    pipe_l = start_rgb(LEFT_SERIAL)
    pipe_r = start_rgb(RIGHT_SERIAL)

    print("按空格保存 rectified 图，按 q 退出。/ Leertaste: rektifizierte Bilder speichern, q: Beenden.")

    try:
        while True:
            frames_l = pipe_l.wait_for_frames()
            frames_r = pipe_r.wait_for_frames()

            color_l = frames_l.get_color_frame()
            color_r = frames_r.get_color_frame()
            if not color_l or not color_r:
                continue

            img_l = np.asanyarray(color_l.get_data())
            img_r = np.asanyarray(color_r.get_data())

            # 保存原始图用于调试（每次覆盖）
            # Rohbilder für Debug speichern (überschreiben)
            cv2.imwrite(os.path.join(DEBUG_OUT_DIR, "left_raw.png"), img_l)
            cv2.imwrite(os.path.join(DEBUG_OUT_DIR, "right_raw.png"), img_r)

            # 4) remap 得到 rectified
            # 4) Remap -> rektifizierte Bilder
            left_rect = cv2.remap(img_l, map1x, map1y, interpolation=cv2.INTER_LINEAR)
            right_rect = cv2.remap(img_r, map2x, map2y, interpolation=cv2.INTER_LINEAR)

            vis = np.hstack([left_rect, right_rect])
            cv2.imshow("Rectified L | R", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                cv2.imwrite(LEFT_OUT_PATH, left_rect)
                cv2.imwrite(RIGHT_OUT_PATH, right_rect)
                print("Saved / Gespeichert:", LEFT_OUT_PATH, RIGHT_OUT_PATH)

    finally:
        pipe_l.stop()
        pipe_r.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

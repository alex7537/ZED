# save_calib_images_dual_rgb_bilingual.py
# ------------------------------------------------------------
# 作用：从两台 RealSense(D435) 的 RGB 流采集棋盘格标定图片，并保存 left_*.png / right_*.png
# Zweck: Kalibrierbilder (Schachbrett) aus zwei RealSense(D435)-RGB-Streams aufnehmen und als left_*.png / right_*.png speichern
# ------------------------------------------------------------
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

# 输出目录
# Ausgabeordner
OUT_DIR = "stereo_calib_images_rgb"
os.makedirs(OUT_DIR, exist_ok=True)

# 分辨率与帧率（建议与后续运行一致）
# Auflösung und FPS (sollten später identisch sein)
W, H = 1280, 720
FPS = 15

# 你的序列号：请确保左右定义在所有脚本里一致
# Ihre Seriennummern: Bitte stellen Sie sicher, dass links/rechts in allen Skripten konsistent ist
LEFT_SERIAL = "YOUR_LEFT_SERIAL"
RIGHT_SERIAL = "YOUR_RIGHT_SERIAL"

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
    pipe_l = start_rgb(LEFT_SERIAL)
    pipe_r = start_rgb(RIGHT_SERIAL)

    idx = 0
    print("按空格保存一对图片，按 q 退出。/ Leertaste: Paar speichern, q: Beenden.")

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

            # 拼接预览（左|右）
            # Vorschau (links|rechts)
            vis = np.hstack([img_l, img_r])
            cv2.imshow("Dual RGB (L|R)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                lp = os.path.join(OUT_DIR, f"left_{idx:03d}.png")
                rp = os.path.join(OUT_DIR, f"right_{idx:03d}.png")
                cv2.imwrite(lp, img_l)
                cv2.imwrite(rp, img_r)
                print(f"Saved / Gespeichert: {lp} , {rp}")
                idx += 1
                time.sleep(0.2)  # 防抖
                               # Entprellen

    finally:
        pipe_l.stop()
        pipe_r.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# B1_stereo_calibrate_rgb_bilingual.py
# ------------------------------------------------------------
# 作用：用双 RGB 相机的棋盘格图片进行单目 + 双目标定，并保存 stereo_calib_rgb.npz
# Zweck: Monokamera- und Stereo-Kalibrierung mit Schachbrettbildern (Dual-RGB) und Speichern als stereo_calib_rgb.npz
# ------------------------------------------------------------
import os
import glob
import numpy as np
import cv2

IMAGE_DIR = "stereo_calib_images_rgb"
W, H = 1280, 720  # 分辨率，和采集时一致
              # Auflösung, muss mit der Aufnahme übereinstimmen

# 棋盘格参数
# Schachbrett-Parameter
CHESSBOARD = (9, 6)    # (列数, 行数) 内角点
                       # (Spalten, Zeilen) innere Ecken
SQUARE_SIZE = 0.026    # 米
                       # Meter

CALIB_OUT = "stereo_calib_rgb.npz"

def collect_image_pairs(image_dir: str):
    """收集左右图片路径并按文件名排序
    Sammelt linke/rechte Bildpfade und sortiert nach Dateinamen
    """
    left_paths = sorted(glob.glob(os.path.join(image_dir, "left_*.png")))
    right_paths = sorted(glob.glob(os.path.join(image_dir, "right_*.png")))
    assert len(left_paths) == len(right_paths) and len(left_paths) > 0, "左右图数量不一致或为 0"
    # Anzahl linke/rechte Bilder passt nicht oder ist 0
    return left_paths, right_paths

def prepare_object_points(chessboard, square_size):
    """生成棋盘格在世界坐标系中的 3D 点（Z=0）
    Erzeugt 3D-Punkte des Schachbretts im Weltkoordinatensystem (Z=0)
    """
    objp = np.zeros((chessboard[0]*chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

def main():
    left_paths, right_paths = collect_image_pairs(IMAGE_DIR)
    print("Found pairs:", len(left_paths))

    objpoints = []   # 3D 世界坐标
                     # 3D-Weltpunkte
    imgpoints_l = [] # 左相机角点
                     # Eckpunkte linke Kamera
    imgpoints_r = [] # 右相机角点
                     # Eckpunkte rechte Kamera

    objp = prepare_object_points(CHESSBOARD, SQUARE_SIZE)

    # 遍历所有左右图片对并检测角点
    # Alle linken/rechten Bildpaare durchlaufen und Ecken detektieren
    for lp, rp in zip(left_paths, right_paths):
        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            print("读取失败 / Lesen fehlgeschlagen:", lp, rp)
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD, None)

        if ret_l and ret_r:
            # 亚像素精细化
            # Subpixel-Verfeinerung
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

            # 可视化检查（可选）
            # Visualisierung zur Kontrolle (optional)
            vis_l = img_l.copy()
            vis_r = img_r.copy()
            cv2.drawChessboardCorners(vis_l, CHESSBOARD, corners_l, ret_l)
            cv2.drawChessboardCorners(vis_r, CHESSBOARD, corners_r, ret_r)
            vis = np.hstack([vis_l, vis_r])
            cv2.imshow("Corners L | R", vis)
            cv2.waitKey(100)
        else:
            print("角点检测失败 / Ecken-Detektion fehlgeschlagen:", lp, rp)

    cv2.destroyAllWindows()

    print("有效图像对数量:", len(objpoints), " / Gültige Paare:", len(objpoints))
    if len(objpoints) < 5:
        raise RuntimeError("有效图像对太少，建议多拍一些。/ Zu wenige gültige Paare, bitte mehr aufnehmen.")

    image_size = (W, H)

    # 1) 单目标定
    # 1) Monokamera-Kalibrierung
    print("Calibrating left camera...")
    ret_l, K_left, dist_left, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None
    )
    print("Left RMS:", ret_l)

    print("Calibrating right camera...")
    ret_r, K_right, dist_right, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None
    )
    print("Right RMS:", ret_r)

    # 2) 双目标定（固定单目内参）
    # 2) Stereo-Kalibrierung (fixierte Intrinsics)
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    print("Stereo calibrating...")
    ret_stereo, K_left_s, dist_left_s, K_right_s, dist_right_s, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K_left, dist_left,
        K_right, dist_right,
        image_size,
        criteria=criteria_stereo,
        flags=flags
    )

    print("Stereo RMS:", ret_stereo)
    print("R =\n", R)
    print("T =\n", T)

    # 3) 立体矫正（生成 rectified 投影矩阵 P1/P2 与 Q）
    # 3) Stereo-Rektifizierung (erzeugt P1/P2 und Q)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left_s, dist_left_s,
        K_right_s, dist_right_s,
        image_size, R, T, alpha=0.5
    )

    # 4) 保存 npz（后续 rectification 与导出 FS 内参都用这一份）
    # 4) Als NPZ speichern (dieselbe Datei für Rektifizierung und FS-Intrinsics verwenden)
    np.savez(
        CALIB_OUT,
        K_left=K_left_s,
        K_right=K_right_s,
        dist_left=dist_left_s,
        dist_right=dist_right_s,
        R=R,
        T=T,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q
    )

    print("Calibration saved to", CALIB_OUT)

if __name__ == "__main__":
    main()

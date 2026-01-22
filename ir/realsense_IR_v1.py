#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
realsense_ir_capture_v3.py

中文说明：
- 采集 RealSense: Color / Depth / IR(Left/Right)
- 保存 raw 与 aligned 两套 Color（并保存各自 intrinsics，含畸变模型与系数）
- 保存 IR-left / IR-right intrinsics（含畸变模型与系数）
- 计算并保存 IR baseline（米）
- 保存 IR(left) -> Color(raw) 以及 IR(left) -> Color(aligned) 外参
- 自动生成 FoundationStereo 需要的 K_ir_fs.txt（第一行 K(3x3) 展平；第二行 baseline_m）

Deutsche Hinweise:
- Erfasst Color/Depth/IR (links/rechts) von RealSense
- Speichert raw und aligned Color inkl. Intrinsics (inkl. Distortion)
- Speichert IR-Intrinsics und Baseline (in Metern)
- Speichert Extrinsics IR(left)->Color(raw/aligned)
- Generiert automatisch K_ir_fs.txt für FoundationStereo
"""

import os
import json
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d


# =========================
# Config
# =========================
OUT_DIR = "FoundationStereo/shared_fs_test"
os.makedirs(OUT_DIR, exist_ok=True)

# TODO: 换成你的真实序列号 / Seriennummer anpassen
SERIAL_IR_CAM = "923322072633"

# 分辨率 / Auflösung
W, H = 1280, 720
FPS = 30

# FoundationStereo K 文件输出路径 / FS K-Datei
FS_K_TXT_PATH = os.path.join(OUT_DIR, "K_ir_fs.txt")


# =========================
# Helper functions
# =========================
def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def intrinsics_to_dict(intr: rs.intrinsics) -> dict:
    """
    中文：把 RealSense intrinsics 保存成 JSON（含畸变模型与系数）
    Deutsch: Intrinsics als JSON (inkl. Distortion-Modell und Koeffizienten)
    """
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
    }


def extrinsics_to_dict(extr: rs.extrinsics) -> dict:
    """
    中文：保存 extrinsics 为 R(3x3) + t(3x1)
    Deutsch: Extrinsics als R(3x3) + t(3x1)
    """
    R = np.array(extr.rotation, dtype=np.float32).reshape(3, 3)
    t = np.array(extr.translation, dtype=np.float32).reshape(3, 1)
    return {"R": R.tolist(), "t": t.tolist()}


def write_fs_K_txt(fx: float, fy: float, cx: float, cy: float, baseline_m: float, out_path: str):
    """
    中文：生成 FoundationStereo 需要的 K_ir_fs.txt
    格式：
      第一行：K(3x3) 按行展开 9 个数字
      第二行：baseline（米）

    Deutsch: Erzeugt K_ir_fs.txt für FoundationStereo:
      Zeile 1: 9 Werte der 3x3-Kameramatrix (row-major)
      Zeile 2: Baseline in Metern
    """
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    with open(out_path, "w") as f:
        f.write(" ".join(map(str, K.reshape(-1).tolist())) + "\n")
        f.write(str(float(baseline_m)) + "\n")

    print(f"[OK] Saved FoundationStereo K file: {out_path}")


def create_pointcloud_from_aligned(depth_u16: np.ndarray,
                                   color_bgr: np.ndarray,
                                   intr_color: rs.intrinsics,
                                   depth_scale: float,
                                   ply_path: str):
    """
    中文：用 aligned depth + aligned color 生成点云（sanity check）
    Deutsch: Punktwolke aus aligned depth + aligned color (Sanity-Check)
    """
    z = depth_u16.astype(np.float32) * depth_scale  # meters
    H_d, W_d = z.shape

    fx, fy, cx, cy = intr_color.fx, intr_color.fy, intr_color.ppx, intr_color.ppy
    v, u = np.meshgrid(np.arange(H_d), np.arange(W_d), indexing="ij")

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z

    valid = (Z > 0) & np.isfinite(Z)
    pts = np.stack((X[valid], Y[valid], Z[valid]), axis=-1)

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    cols = (color_rgb[valid].reshape(-1, 3).astype(np.float32) / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(ply_path, pcd)

    print(f"[OK] RealSense point cloud saved: {ply_path}  points={pts.shape[0]}")


# =========================
# Main
# =========================
def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # 指定设备 / Gerät auswählen
    config.enable_device(SERIAL_IR_CAM)

    # Streams
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)

    # Align depth -> color
    align = rs.align(rs.stream.color)

    print("Starting pipeline...")
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print("Depth scale:", depth_scale)

    # Warm up exposure / Auto-Belichtung stabilisieren
    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    # Raw frames
    color_raw_frame = frames.get_color_frame()
    ir_left_frame = frames.get_infrared_frame(1)
    ir_right_frame = frames.get_infrared_frame(2)

    # Aligned frames
    color_aligned_frame = aligned.get_color_frame()
    depth_aligned_frame = aligned.get_depth_frame()

    if not color_raw_frame or not color_aligned_frame or not depth_aligned_frame or not ir_left_frame or not ir_right_frame:
        pipeline.stop()
        raise RuntimeError("No frames received (color/depth/IR).")

    # Convert to numpy arrays
    color_raw = np.asanyarray(color_raw_frame.get_data())
    color_aligned = np.asanyarray(color_aligned_frame.get_data())
    depth_aligned = np.asanyarray(depth_aligned_frame.get_data())
    ir_left = np.asanyarray(ir_left_frame.get_data())
    ir_right = np.asanyarray(ir_right_frame.get_data())

    # IR gray -> 3ch
    ir_left_rgb = cv2.cvtColor(ir_left, cv2.COLOR_GRAY2RGB)
    ir_right_rgb = cv2.cvtColor(ir_right, cv2.COLOR_GRAY2RGB)

    print("Color raw shape:", color_raw.shape)
    print("Color aligned shape:", color_aligned.shape)
    print("Depth aligned shape:", depth_aligned.shape)
    print("IR shape:", ir_left.shape)

    # Intrinsics
    intr_ir_left = ir_left_frame.profile.as_video_stream_profile().intrinsics
    intr_ir_right = ir_right_frame.profile.as_video_stream_profile().intrinsics
    intr_color_raw = color_raw_frame.profile.as_video_stream_profile().intrinsics
    intr_color_aligned = color_aligned_frame.profile.as_video_stream_profile().intrinsics

    # Baseline (IR L -> IR R)
    extr_lr = ir_left_frame.profile.get_extrinsics_to(ir_right_frame.profile)
    baseline_m = float(np.linalg.norm(np.array(extr_lr.translation, dtype=np.float32)))
    print("Baseline (IR L->R) [m] =", baseline_m)

    # Save intrinsics JSON
    ir_left_K = intrinsics_to_dict(intr_ir_left)
    ir_left_K["baseline_m"] = baseline_m
    save_json(os.path.join(OUT_DIR, "ir_left_intrinsics.json"), ir_left_K)
    save_json(os.path.join(OUT_DIR, "ir_right_intrinsics.json"), intrinsics_to_dict(intr_ir_right))
    save_json(os.path.join(OUT_DIR, "color_raw_intrinsics.json"), intrinsics_to_dict(intr_color_raw))
    save_json(os.path.join(OUT_DIR, "color_aligned_intrinsics.json"), intrinsics_to_dict(intr_color_aligned))

    # Save extrinsics
    extr_ir2color_raw = ir_left_frame.profile.get_extrinsics_to(color_raw_frame.profile)
    save_json(os.path.join(OUT_DIR, "ir2color_raw_extrinsics.json"), extrinsics_to_dict(extr_ir2color_raw))

    extr_ir2color_aligned = ir_left_frame.profile.get_extrinsics_to(color_aligned_frame.profile)
    save_json(os.path.join(OUT_DIR, "ir2color_aligned_extrinsics.json"), extrinsics_to_dict(extr_ir2color_aligned))

    # Save images
    cv2.imwrite(os.path.join(OUT_DIR, "color_raw_0000.png"), color_raw)
    cv2.imwrite(os.path.join(OUT_DIR, "color_aligned_0000.png"), color_aligned)
    cv2.imwrite(os.path.join(OUT_DIR, "depth_aligned_0000.png"), depth_aligned)
    cv2.imwrite(os.path.join(OUT_DIR, "ir_left_0000.png"), ir_left_rgb)
    cv2.imwrite(os.path.join(OUT_DIR, "ir_right_0000.png"), ir_right_rgb)

    # Save RS sanity point cloud
    ply_rs = os.path.join(OUT_DIR, "cloud_rs_aligned.ply")
    create_pointcloud_from_aligned(depth_aligned, color_aligned, intr_color_aligned, depth_scale, ply_rs)

    # ✅ Generate K_ir_fs.txt for FoundationStereo (integrated from make_K_txt_from_json.py)
    # 中文：FS 使用 IR-left 的内参（fx,fy,cx,cy）和 baseline（米）
    # Deutsch: FS nutzt IR-left Intrinsics + Baseline (m)
    write_fs_K_txt(
        fx=float(intr_ir_left.fx),
        fy=float(intr_ir_left.fy),
        cx=float(intr_ir_left.ppx),
        cy=float(intr_ir_left.ppy),
        baseline_m=baseline_m,
        out_path=FS_K_TXT_PATH
    )

    pipeline.stop()
    print("[DONE] All outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()

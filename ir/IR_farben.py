#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fs_ir_depth_to_color_depth_v2.py

中文说明：
- 输入：FS 输出 depth_meter.npy（IR-left 平面、单位米）
- 使用 IR 内参 + IR->Color 外参 + Color 内参，将深度投影到 Color 像素平面
- 输出：
  1) 彩色点云 cloud_fs_ir2color_*.ply（用于可视化 sanity check）
  2) 与 color_raw 对齐的深度 depth_fs_ir2color_raw_0000.png（16-bit mm）
  3) 与 color_aligned 对齐的深度 depth_fs_ir2color_aligned_0000.png（16-bit mm）
- Pose 推荐使用：color_aligned_0000.png + depth_fs_ir2color_aligned_0000.png + color_aligned_intrinsics.json

Deutsche Hinweise:
- Projektion der FS-Tiefenkarte (IR-left, Meter) auf die Color-Bildebene
- Exportiert RGB-ausgerichtete Depth-PNG (16-bit, mm) und farbige Punktwolken
- Für Pose empfohlen: color_aligned + depth_fs_ir2color_aligned + passende Intrinsics
"""

import os
import json
import numpy as np
import cv2
import open3d as o3d


# =========================
# Config
# =========================
OUT_DIR = "FoundationStereo/shared_fs_test"

FS_DEPTH_PATH = os.path.join(OUT_DIR, "depth_meter.npy")  # FS depth in meters (IR-left)
COLOR_RAW_PATH = os.path.join(OUT_DIR, "color_raw_0000.png")
COLOR_ALIGNED_PATH = os.path.join(OUT_DIR, "color_aligned_0000.png")

IR_INTR_PATH = os.path.join(OUT_DIR, "ir_left_intrinsics.json")
COLOR_RAW_INTR_PATH = os.path.join(OUT_DIR, "color_raw_intrinsics.json")
COLOR_ALIGNED_INTR_PATH = os.path.join(OUT_DIR, "color_aligned_intrinsics.json")

IR2COLOR_RAW_EXTR_PATH = os.path.join(OUT_DIR, "ir2color_raw_extrinsics.json")
IR2COLOR_ALIGNED_EXTR_PATH = os.path.join(OUT_DIR, "ir2color_aligned_extrinsics.json")

PLY_RAW_OUT = os.path.join(OUT_DIR, "cloud_fs_ir2color_raw_colored.ply")
PLY_ALIGNED_OUT = os.path.join(OUT_DIR, "cloud_fs_ir2color_aligned_colored.ply")

DEPTH_RAW_PNG_OUT = os.path.join(OUT_DIR, "depth_fs_ir2color_raw_0000.png")
DEPTH_ALIGNED_PNG_OUT = os.path.join(OUT_DIR, "depth_fs_ir2color_aligned_0000.png")


# =========================
# Helpers
# =========================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def assert_intr_matches_image(K: dict, img: np.ndarray, name: str):
    H, W = img.shape[:2]
    assert K["width"] == W and K["height"] == H, (
        f"[{name}] intrinsics size mismatch: "
        f"K says (W,H)=({K['width']},{K['height']}), "
        f"image is (W,H)=({W},{H})"
    )


def backproject_depth_to_points(depth_m: np.ndarray, K: dict):
    """
    depth_m: (H,W) meters, same resolution as K (IR-left)
    return pts: (N,3) in IR camera frame
    """
    H, W = depth_m.shape
    assert H == K["height"] and W == K["width"], "Depth size must match IR intrinsics."

    fx, fy = K["fx"], K["fy"]
    cx, cy = K["cx"], K["cy"]

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth_m.astype(np.float32)

    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return None

    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    return pts


def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray):
    t = t.reshape(3, 1)
    return (R @ pts.T + t).T


def project_points_to_pixels(pts: np.ndarray, K: dict):
    fx, fy = K["fx"], K["fy"]
    cx, cy = K["cx"], K["cy"]

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = np.isfinite(Z) & (Z > 0)

    X, Y, Z = X[valid], Y[valid], Z[valid]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u, v, Z, valid


def zbuffer_select(u, v, z, pts, colors, H, W):
    """
    每个像素只保留最近的点（z 最小），生成 depth map。
    """
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & np.isfinite(z) & (z > 0)
    ui, vi, z = ui[inb], vi[inb], z[inb]
    pts = pts[inb]
    colors = colors[inb]

    if ui.size == 0:
        return None, None, None

    pix = vi * W + ui
    order = np.lexsort((z, pix))
    pix_s = pix[order]
    z_s = z[order]
    pts_s = pts[order]
    col_s = colors[order]
    ui_s = ui[order]
    vi_s = vi[order]

    keep = np.ones_like(pix_s, dtype=bool)
    keep[1:] = pix_s[1:] != pix_s[:-1]

    pts_keep = pts_s[keep]
    col_keep = col_s[keep]
    ui_keep = ui_s[keep]
    vi_keep = vi_s[keep]
    z_keep = z_s[keep]

    depth_map = np.zeros((H, W), dtype=np.float32)
    depth_map[vi_keep, ui_keep] = z_keep

    return pts_keep, col_keep, depth_map


def export_one(color_bgr, K_color, R, t, pts_ir, ply_out, depth_png_out):
    Hc, Wc = color_bgr.shape[:2]
    assert_intr_matches_image(K_color, color_bgr, "Color")

    # IR -> Color
    pts_c = transform_points(pts_ir, R, t)

    # project
    u, v, z, valid_mask = project_points_to_pixels(pts_c, K_color)
    pts_c_valid = pts_c[valid_mask]

    # sample color
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    inb = (ui >= 0) & (ui < Wc) & (vi >= 0) & (vi < Hc)
    ui, vi = ui[inb], vi[inb]
    z = z[inb]
    pts_c_valid = pts_c_valid[inb]

    if ui.size == 0:
        raise RuntimeError("All projected points are out of RGB bounds. Check extrinsics/intrinsics.")

    cols = color_rgb[vi, ui]

    # z-buffer
    pts_keep, col_keep, depth_color_m = zbuffer_select(
        u=ui.astype(np.float32),
        v=vi.astype(np.float32),
        z=z,
        pts=pts_c_valid,
        colors=cols,
        H=Hc,
        W=Wc
    )
    if pts_keep is None:
        raise RuntimeError("Z-buffer kept no points.")

    # Export depth (16-bit mm)
    depth_mm = np.clip(depth_color_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(depth_png_out, depth_mm)
    print("[OK] Saved depth (16-bit mm):", depth_png_out)

    # Optional: depth range clip + outlier removal
    z_keep = pts_keep[:, 2]
    mask = (z_keep > 0.1) & (z_keep < 2.0)
    pts_keep = pts_keep[mask]
    col_keep = col_keep[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_keep)
    pcd.colors = o3d.utility.Vector3dVector(col_keep)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    o3d.io.write_point_cloud(ply_out, pcd)
    print("[OK] Saved colored point cloud:", ply_out)
    print("     points:", np.asarray(pcd.points).shape[0])


def main():
    # Load
    K_ir = load_json(IR_INTR_PATH)
    K_color_raw = load_json(COLOR_RAW_INTR_PATH)
    K_color_aligned = load_json(COLOR_ALIGNED_INTR_PATH)

    extr_raw = load_json(IR2COLOR_RAW_EXTR_PATH)
    extr_aligned = load_json(IR2COLOR_ALIGNED_EXTR_PATH)

    R_raw = np.array(extr_raw["R"], dtype=np.float32)
    t_raw = np.array(extr_raw["t"], dtype=np.float32)
    R_al = np.array(extr_aligned["R"], dtype=np.float32)
    t_al = np.array(extr_aligned["t"], dtype=np.float32)

    color_raw = cv2.imread(COLOR_RAW_PATH, cv2.IMREAD_COLOR)
    color_aligned = cv2.imread(COLOR_ALIGNED_PATH, cv2.IMREAD_COLOR)
    if color_raw is None or color_aligned is None:
        raise RuntimeError("Failed to read color images. Run capture script first.")

    depth_ir_m = np.load(FS_DEPTH_PATH).astype(np.float32)
    assert depth_ir_m.ndim == 2, "FS depth must be HxW"
    assert depth_ir_m.shape == (K_ir["height"], K_ir["width"]), \
        f"FS depth shape {depth_ir_m.shape} != IR intrinsics {(K_ir['height'], K_ir['width'])}"

    # Backproject IR depth -> IR 3D
    pts_ir = backproject_depth_to_points(depth_ir_m, K_ir)
    if pts_ir is None:
        print("No valid points in FS depth.")
        return

    # Export for raw color
    export_one(
        color_bgr=color_raw,
        K_color=K_color_raw,
        R=R_raw,
        t=t_raw,
        pts_ir=pts_ir,
        ply_out=PLY_RAW_OUT,
        depth_png_out=DEPTH_RAW_PNG_OUT
    )

    # Export for aligned color (recommended for pose)
    export_one(
        color_bgr=color_aligned,
        K_color=K_color_aligned,
        R=R_al,
        t=t_al,
        pts_ir=pts_ir,
        ply_out=PLY_ALIGNED_OUT,
        depth_png_out=DEPTH_ALIGNED_PNG_OUT
    )

    print("[DONE] Exports finished.")


if __name__ == "__main__":
    main()

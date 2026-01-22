# ZED
zwei RGB \
python zed-rgb.py

cp /home/match/ZED/zed_capture/K_rgb_fs.txt /home/match/FS/FoundationStereo/shared_fs_test/ \
cp /home/match/ZED/zed_capture/left_000004.png /home/match/FS/FoundationStereo/shared_fs_test/ \
cp /home/match/ZED/zed_capture/right_000004.png /home/match/FS/FoundationStereo/shared_fs_test/

python scripts/run_demo.py --left_file ./shared_fs_test/left_000004.png --right_file ./shared_fs_test/right_000004.png --ckpt_dir ./pretrained_models/model_best_bp2.pth --out_dir ./outputs_test --intrinsic_file ./shared_fs_test/K_rgb_fs.txt

# ZED → FoundationStereo → Depth → Pose Pipeline

本目录用于 **ZED 双目 RGB 相机** 的数据采集、深度生成与对比分析，
并将结果用于 **FoundationStereo (FS)** 以及后续 **6D Pose**（如 FoundationPose / SERP-6D）。

Dieses Verzeichnis enthält die komplette Pipeline für die **ZED Stereo-RGB Kamera**:
Aufnahme → Tiefenberechnung → Vergleich → Nutzung für 6D-Pose-Schätzung.

---

## 1. 目录结构 / Verzeichnisstruktur

```text
ZED/
├── zed-rgb.py            # ZED 左右 RGB 采集脚本
├── zed_capture/          # 保存 ZED 采集结果（左右图、内参）
│   ├── left_xxxxxx.png
│   ├── right_xxxxxx.png
│   └── K_rgb_fs.txt      # 提供给 FoundationStereo 的内参文件
│
├── npy_png/              # depth: .npy ↔ .png 转换
│   └── npy_png.py
│
├── depth_compare.py      # 深度结果对比（FS vs ZED 原生）
├── depth_compare/        # 对比结果输出（图像/统计）
│
├── depth_uberprufen.py   # 深度数值/尺度检查
├── compare-abstand/      # 距离（Abstand）对比实验
│
├── test---pose/          # 6D Pose 测试相关代码
├── 对齐检测/             # RGB / Depth / Pose 对齐检查
│
├── zed.txt               # 实验或调试记录
└── README.md             # 本说明文件


```

# IR → FoundationStereo → Depth → Pose Pipeline (RealSense)

## Overview

This repository implements an **IR-based stereo depth pipeline** using **FoundationStereo** and **Intel RealSense**, and converts the resulting depth into **RGB-aligned depth maps** suitable for **6D object pose estimation** (e.g. FoundationPose).

DE:
Dieses Projekt implementiert eine vollständige Pipeline zur **Stereo-Tiefenschätzung mit IR-Sensoren einer RealSense-Kamera**, basierend auf **FoundationStereo**, und projiziert die berechnete Tiefe in das **RGB-Koordinatensystem**, sodass konsistente RGB-D Daten für die 6D-Posenbestimmung entstehen.

---

## What this pipeline does (Kurzfassung)

**Input**

* RealSense IR stereo images (IR-left / IR-right)
* RealSense color image
* Camera intrinsics & extrinsics from RealSense

**Processing**

1. Capture synchronized IR + Color data from RealSense
2. Run FoundationStereo on IR stereo pair → metric depth (meters)
3. Back-project FS depth into IR camera frame
4. Transform IR depth into Color camera frame
5. Project depth onto Color image plane
6. Export RGB-aligned depth (16-bit PNG, mm)

**Output**

* RGB image
* RGB-aligned depth map
* Consistent camera intrinsics
  → directly usable for **6D Pose estimation**

DE:
Ziel ist es, **FoundationStereo als Ersatz für native RealSense-Tiefendaten** in einer industriellen 6D-Pose-Pipeline einzusetzen.

---

## Repository Structure

```
.
├── realsense_ir_capture_v2.py
│   └─ Capture IR stereo + color + intrinsics/extrinsics
│
├── FoundationStereo/
│   └─ scripts/run_demo.py
│      └─ IR stereo → depth_meter.npy
│
├── fs_ir_depth_to_color_depth_v2.py
│   └─ Project FS depth → color image plane
│
└── shared_fs_test/
    ├── ir_left_0000.png
    ├── ir_right_0000.png
    ├── color_aligned_0000.png
    ├── depth_fs_ir2color_aligned_0000.png
    ├── *_intrinsics.json
    └── *_extrinsics.json
```

---

## Recommended RGB-D Output for Pose

✅ **Use this combination for 6D Pose estimation**

* **RGB**: `color_aligned_0000.png`
* **Depth**: `depth_fs_ir2color_aligned_0000.png` (16-bit PNG, mm)
* **Intrinsics**: `color_aligned_intrinsics.json`

DE:
RGB, Depth und K müssen **im gleichen Pixel-Koordinatensystem** liegen, sonst kommt es zu Pose-Drift.

---

RGB → FoundationStereo → Depth → Pose Pipeline (Dual RGB)
0. 目标 / Ziel

本 pipeline 使用 两台 RGB 相机（Dual RGB Stereo） 构建立体系统，通过棋盘格完成标定与立体矫正，将 矫正后的左右 RGB 图像输入 FoundationStereo (FS) 生成度量深度，并最终作为 6D Pose Estimation（如 FoundationPose / SERP-6D） 的 RGB-D 输入。

DE:
Dieses Pipeline nutzt zwei RGB-Kameras zur Stereo-Rekonstruktion, führt eine vollständige Kalibrierung und Rektifizierung durch und verwendet FoundationStereo (FS) zur Tiefenschätzung. Die resultierenden RGB-D Daten sind konsistent und für 6D-Posenschätzung geeignet.

1. 输出文件概览 / Output Übersicht
1.1 标定阶段输出（offline）

由 save_calib_images_dual_rgb_bilingual.py 与
stereo_calibrate_rgb_bilingual.py 生成：

stereo_calib_images_rgb/

left_000.png / right_000.png

stereo_calib_rgb.npz（核心文件）

左右相机内参 K_left / K_right

畸变参数 dist_left / dist_right

外参 R / T

立体矫正结果 R1 / R2

矫正投影矩阵 P1 / P2

视差-深度矩阵 Q

DE:
Die Datei stereo_calib_rgb.npz enthält alle relevanten Stereo-Kalibrierparameter und wird in allen Folgeschritten wiederverwendet.

1.2 运行阶段输出（runtime）

由 capture_rectified_for_fs_bilingual.py 生成：

shared_fs_test/

left_rect.png

right_rect.png

（调试用）

fs_dual_rgb_debug/left_raw.png

fs_dual_rgb_debug/right_raw.png

由 make_fs_intrinsic_from_npz_bilingual.py 生成：

K_d435_2RGB.txt

rectified 内参 K（3×3，展开）

baseline（单位：米）

1.3 FoundationStereo 输出

由 FoundationStereo 官方脚本生成：

depth_meter.npy

FS 输出深度

分辨率：rectified RGB

单位：米

2. 推荐用于 Pose 的输入组合 / Empfohlene Pose Inputs

✅ 推荐（强烈）：

RGB: left_rect.png

Depth: depth_meter.npy → 转换为 16-bit PNG（mm）

K: K_d435_2RGB.txt

DE:
RGB、Depth 和 Intrinsics 必须来自同一个 rectified 坐标系，否则会直接导致 6D Pose 漂移。

下面给你两块内容：

1. **RGB 线路的目录结构图**（和你 IR 那种风格一致）
2. **程序运行顺序图**（含你写的那几条命令 + 我补齐 FS 推理这一步）

同时我会把你给的路径/环境写得**可直接粘贴进 README**。

---

## RGB Pipeline Repo Structure

```text
.
├── save_calib_images_dual_rgb.py
│   └─ Capture chessboard pairs (Dual RGB) → stereo_calib_images_rgb/
│
├── stereo_calibrate_rgb.py
│   └─ Mono + Stereo calibration + stereoRectify → stereo_calib_rgb.npz
│
├── make_fs_intrinsic_from_npz.py
│   └─ Export rectified K + baseline (from P1/P2) → assets/K_d435.txt
│
├── capture_rectified_for_fs.py
│   └─ Runtime capture + undistort/rectify → shared_fs_test/left_rect.png, right_rect.png
│
├── FoundationStereo/
│   └─ scripts/run_demo.py
│      └─ Rectified RGB stereo → depth_meter.npy
│
├── assets/
│   └── K_d435.txt
│
└── shared_fs_test/
    ├── left_rect.png
    ├── right_rect.png
    ├── depth_meter.npy
    └── (optional) debug/
        ├── left_raw.png
        └── right_raw.png
```

> 注：`assets/K_d435.txt` 的内容应来自 rectified 的 `P1/P2`（不要用 `||T||`）。这点在你的 `make_fs_intrinsic_from_npz*.py` 逻辑里就是这么做的。

---

## Program Run Order (Flow)

### A) 一次性准备（标定阶段 / offline）

```text
(1) 拍棋盘格数据 15–25 组
    python save_calib_images_dual_rgb.py
        ↓
    stereo_calib_images_rgb/left_*.png, right_*.png

(2) 立体标定 + 立体矫正
    python stereo_calibrate_rgb.py
        ↓
    stereo_calib_rgb.npz

(3) 生成 FS 内参（rectified K + baseline）
    python make_fs_intrinsic_from_npz.py
        ↓
    assets/K_d435.txt
```

---

### B) 每次测试 FS（运行阶段 / runtime）

你给的命令我按“可直接贴 README”的形式整理如下，并把 FS 推理补齐：

```bash
# 进入你的 RGB pipeline 工程目录
cd /home/match/foundationstereo-industrial-6dpose/RGB-pipeline

# 1) 采集并输出 rectified 左右图（给 FS 用）
conda activate foundationpose
python capture_rectified_for_fs.py
# 输出: shared_fs_test/left_rect.png, shared_fs_test/right_rect.png

# 2) 跑 FoundationStereo 得到深度
cd FoundationStereo
conda activate foundation_stereo

python scripts/run_demo.py \
  --left_file  ../shared_fs_test/left_rect.png \
  --right_file ../shared_fs_test/right_rect.png \
  --ckpt_dir   ./pretrained_models/model_best_*.pth \
  --out_dir    ./outputs_test \
  --intrinsic_file ../assets/K_d435.txt

# 输出: (通常会在 out_dir 或 shared_fs_test 生成/复制) depth_meter.npy
```




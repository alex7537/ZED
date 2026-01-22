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




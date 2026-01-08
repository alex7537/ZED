import os
import cv2
import pyzed.sl as sl


def main(
    out_dir="zed_capture",
    num_frames=5,
    fps=15,
    resolution="HD2K",
    WARMUP_FRAMES=30
):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Create ZED camera
    zed = sl.Camera()

    # 2. Init parameters
    init_params = sl.InitParameters()
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    if resolution.upper() == "HD2K":
        init_params.camera_resolution = sl.RESOLUTION.HD2K
    elif resolution.upper() == "HD1080":
        init_params.camera_resolution = sl.RESOLUTION.HD1080
    elif resolution.upper() == "HD720":
        init_params.camera_resolution = sl.RESOLUTION.HD720
    elif resolution.upper() == "VGA":
        init_params.camera_resolution = sl.RESOLUTION.VGA
    else:
        raise ValueError("Unsupported resolution")

    # 3. Open camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] Cannot open ZED:", status)
        return

    print("[INFO] Camera opened")

    # 4. Print calibration
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters

    fx, fy = calib.left_cam.fx, calib.left_cam.fy
    cx, cy = calib.left_cam.cx, calib.left_cam.cy
    tx, _, _ = calib.stereo_transform.get_translation().get()

    print("=== ZED Calibration (Left) ===")
    print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    print(f"baseline ≈ {abs(tx)/1000:.6f} m")
    # ======================================================
    # Save intrinsics for FoundationStereo (txt format)
    # ======================================================
    

    K_path = os.path.join(out_dir, "K_rgb_fs.txt")  # 名字你随便

    with open(K_path, "w") as f:
        f.write(f"{fx} 0.0 {cx} 0.0 {fy} {cy} 0.0 0.0 1.0\n")
        f.write(f"{abs(tx)/1000}\n")   # baseline：米（m）

    print(f"[INFO] Saved FS intrinsics to {K_path}")



    # 5. Runtime + Mats
    runtime = sl.RuntimeParameters()
    left_mat = sl.Mat()
    right_mat = sl.Mat()

    # ======================================================
    # 6. Warm-up (完全等价于 RealSense 的 wait_for_frames)
    # ======================================================
    print(f"[INFO] Warming up for {WARMUP_FRAMES} frames...")
    for i in range(WARMUP_FRAMES):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            pass

    print("[INFO] Warm-up finished, start capturing")

    # ======================================================
    # 7. Capture & save
    # ======================================================
    saved = 0
    while saved < num_frames:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

        left_rgba = left_mat.get_data()
        right_rgba = right_mat.get_data()

        # ZED 通常是 BGRA
        left_bgr = cv2.cvtColor(left_rgba, cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(right_rgba, cv2.COLOR_BGRA2BGR)

        cv2.imwrite(
            os.path.join(out_dir, f"left_{saved:06d}.png"),
            left_bgr
        )
        cv2.imwrite(
            os.path.join(out_dir, f"right_{saved:06d}.png"),
            right_bgr
        )

        print(f"[INFO] Saved pair {saved}")
        saved += 1

    # 8. Close camera
    zed.close()
    print("[DONE] Capture finished.")


if __name__ == "__main__":
    main(
        out_dir="zed_capture",
        num_frames=5,
        fps=15,
        resolution="HD2K",
        WARMUP_FRAMES=30
    )   


    
# ZED
zwei RGB \
python zed-rgb.py

cp /home/match/ZED/zed_capture/K_rgb_fs.txt /home/match/FS/FoundationStereo/shared_fs_test/
cp /home/match/ZED/zed_capture/left_000004.png /home/match/FS/FoundationStereo/shared_fs_test/
cp /home/match/ZED/zed_capture/right_000004.png /home/match/FS/FoundationStereo/shared_fs_test/

python scripts/run_demo.py --left_file ./shared_fs_test/left_000004.png --right_file ./shared_fs_test/right_000004.png --ckpt_dir ./pretrained_models/model_best_bp2.pth --out_dir ./outputs_test --intrinsic_file ./shared_fs_test/K_rgb_fs.txt

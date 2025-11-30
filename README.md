# NPM-VLA

## Configuration

Note1:

After the configuration of openpi, we need to replace file [.venv/lib/python3.11/site-packages/lerobot/common/datasets/video_utils.py]()

（Using [utils\video_utils.py]()）

Note2:

Replace [src\openpi\training\config.py]() with [utils\config.py]()

Note3:

Replace [src\openpi\policies\libero_policy.py]() with [utils\libero_policy.py]()

Note4:

Convert bag into Lerobot , run [utils\convert_bag2lerobot21_dualarm.py]()

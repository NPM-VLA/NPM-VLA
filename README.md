# NPM-VLA

This project is based on [OpenPI](https://github.com/Physical-Intelligence/openpi) and follows the official OpenPI workflow for robot learning and policy training.

## Installation

### 1. Clone the Repository with Submodules

```bash
git clone --recurse-submodules https://github.com/NPM-VLA/NPM-VLA.git
cd NPM-VLA
```

If you've already cloned the repository without submodules, initialize them:

```bash
git submodule update --init --recursive
```

### 2. Follow OpenPI Setup

Navigate to the `openpi` directory and follow the [OpenPI official setup guide](https://github.com/Physical-Intelligence/openpi):

```bash
cd openpi
```

Install required dependencies:

- Install `uv` package manager
- Set up the base environment
- Configure necessary dependencies according to OpenPI documentation

> **Note**: The `openpi` submodule contains our customized configurations for NPM-VLA, including modified training config and policy modules.

## Configuration

### Replace Video Utils

After setting up the OpenPI environment, you need to modify the LeRobot video utilities to fix compatibility issues with torchvision and pyav.

Replace:

```
.venv/lib/python3.11/site-packages/lerobot/common/datasets/video_utils.py
```

with:

```
utils/video_utils.py
```

> **Why?** This modification resolves issues with torchvision and pyav video encoding/decoding in the LeRobot dataset pipeline.

### Data Conversion

Convert ROS bag files to LeRobot format:

```bash
python utils/convert_bag2lerobot21_dualarm.py
```

#### Data Formats

##### ROS Bag Format (Input)

ROS bag files (`.bag`) containing the following topics:

**Camera Topics** (sensor_msgs/CompressedImage):

- `/realsense_top/color/image_raw/compressed` - Main (top) camera view
- `/realsense_left/color/image_raw/compressed` - Left wrist camera
- `/realsense_right/color/image_raw/compressed` - Right wrist camera

**Robot State Topics** (sensor_msgs/JointState):

- `/robot/arm_left/joint_states_single` - Left arm joint states (8 joints)
- `/robot/arm_right/joint_states_single` - Right arm joint states (8 joints)

**Teleoperation Action Topics** (sensor_msgs/JointState):

- `/teleop/arm_left/joint_states_single` - Left arm teleop actions (7 DOF)
- `/teleop/arm_right/joint_states_single` - Right arm teleop actions (7 DOF)

##### LeRobot 2.1 Format (Output)

After conversion, the LeRobot dataset will be organized as follows:

```
<dataset_name>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.main/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       ├── observation.images.secondary_0/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       ├── observation.images.secondary_1/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── ...
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   ├── episodes_stats.jsonl
│   └── README.md
└── README.md
```

**Data Specifications**:

- `action`: 14-dimensional float32 vector (left arm 7 DOF + right arm 7 DOF)
- `observation.state`: 16-dimensional float32 vector (left arm 8 joints + right arm 8 joints)
- Video resolution: 256×256×3 RGB @ 10 FPS
- Format: Parquet files for tabular data, MP4 for videos

## Training

### 1. Compute Normalization Statistics

Before training, compute the normalization statistics for your dataset:

```bash
cd openpi
uv run python scripts/compute_norm_stats.py --config-name pi05_npm
```

### 2. Start Training

```bash
cd openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_npm_lora --exp-name=sweep2E --overwrite
```

**Parameters:**

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: Limits GPU memory usage to 90%
- `pi05_npm_lora`: Training configuration name
- `--exp-name`: Experiment name for logging
- `--overwrite`: Overwrite existing experiment data

## Inference

TODO: Inference instructions will be added here

## Deployment

TODO: Deployment instructions will be added here

## Troubleshooting

### libgthread-2.0.so.0 not found

If you encounter the error `libgthread-2.0.so.0: cannot open shared object file`, install the missing library:

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0
```

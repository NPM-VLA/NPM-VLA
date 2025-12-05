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

Note:
Remember to update below settings when preparing data:
`utils\convert_bag2lerobot21_dualarm.py`

1. REPO_NAME # local hf dir to store data
2. HF_DATASET_REPO # remote repo
3. TASK_NAMES

## Training

### 1. Compute Normalization Statistics

Before training, compute the normalization statistics for your dataset:

```bash
cd openpi
uv run python scripts/compute_norm_stats.py --config-name pi05_npm_lora
```

### 2. Start Training

```bash
cd openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_npm_lora --exp-name=push_block_dual --overwrite
```

**Parameters:**

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: Limits GPU memory usage to 90%
- `pi05_npm_lora`: Training configuration name
- `--exp-name`: Experiment name for logging
- `--overwrite`: Overwrite existing experiment data

Note:
Remember to update below settings when finetuing:
`src\openpi\training\config.py`

1. repo_id: same as REPO_NAME
2. some params will infect the training process, like lr_schedule and so on.

## Inference

After training is complete,  run inference using the trained policy checkpoint. The policy server loads the trained model and provides action predictions based on observations and prompts.

### Start Policy Server

Navigate to the OpenPI directory and run the policy server:

```bash
cd openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_npm_lora \
  --policy.dir=/path/to/checkpoint/directory
```

**Example:**

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_npm_lora \
  --policy.dir=/home/zeno/NPM-VLA-Project/NPM-VLA/openpi/checkpoints/pi05_npm_lora/push_block_pi05
```

**Parameters:**

- `policy:checkpoint`: Specifies to load a checkpoint-based policy
- `--policy.config`: Training configuration name
- `--policy.dir`: Path to the checkpoint directory (typically under `checkpoints/<config_name>/<exp_name>/<step>`)
- `--default_prompt`: Default language instruction for the task (optional)

**Notes:**

- The checkpoint directory should contain the model weights and configuration files from training
- The policy server will initialize the model and wait for observation inputs
- Make sure the configuration name matches the one used during training

## Deployment

This section covers deploying the trained policy on real robot hardware. The deployment process involves setting up the robot control system and connecting it with the trained VLA policy.

### Prerequisites

- Trained policy checkpoint (see [Training](#training) and [Inference](#inference) sections)
- Policy server running (see [Inference](#inference) section)
- Robot hardware setup (refer to [zeno-wholebody-teleop](https://github.com/Jeong-zju/zeno-wholebody-teleop))
- ROS environment properly configured

### 1. Configure Launch Files

Before starting the robot, modify the ROS launch files to redirect control commands from teleoperation to VLA policy output.

**Edit `piper_dual_robot.launch`:**

Comment out the teleoperation command mapping and add VLA command mapping:

```xml
<!-- Original teleoperation mapping (comment out) -->
<!-- <remap from="$(arg robot_prefix_left)joint_pos_cmd" to="$(arg teleop_prefix_left)joint_states_single"/> -->
<!-- <remap from="$(arg robot_prefix_right)joint_pos_cmd" to="$(arg teleop_prefix_right)joint_states_single"/> -->

<!-- New VLA policy mapping -->
<remap from="$(arg robot_prefix_left)joint_pos_cmd" to="$(arg robot_prefix_left)vla_pos_cmd"/>
<remap from="$(arg robot_prefix_right)joint_pos_cmd" to="$(arg robot_prefix_right)vla_pos_cmd"/>
```

**Why this change?**

This remapping redirects the joint position commands from teleoperation topics to VLA policy output topics, allowing the trained model to control the robot instead of manual teleoperation.

### 2. Start Robot Control System

Source the ROS workspace and launch the robot control nodes:

```bash
# Source the workspace
source devel/setup.bash

# Launch the robot with all sensors
roslaunch piper_bridge start_robot_all.launch \
  ranger_can_port:=can0 \
  left_can_port:=can_left \
  right_can_port:=can_right \
  enable_ranger:=false \
  enable_paddle2ranger:=false \
  enable_dual_arm:=true \
  enable_cameras:=true \
  enable_rviz:=true \
  camera_left_usb_port:=2-1 \
  camera_right_usb_port:=2-8 \
  camera_top_usb_port:=2-2
```

**Parameters:**

- `ranger_can_port`: CAN port for ranger base (can0)
- `left_can_port`: CAN port for left arm (can_left)
- `right_can_port`: CAN port for right arm (can_right)
- `enable_dual_arm`: Enable dual-arm control
- `enable_cameras`: Enable all RealSense cameras
- `enable_rviz`: Launch RViz for visualization
- `camera_*_usb_port`: USB ports for each camera

### 3. Run VLA Policy Controller

In a separate terminal, activate the Python environment and run the main control script:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Set ROS master URI (adjust if running on a different machine)
export ROS_MASTER_URI=http://localhost:11311

# Run the VLA policy controller
uv run scripts/piper_pi05_main.py
```

**What this script does:**

1. **Observation Collection**: Subscribes to robot state and camera topics to gather observations

   - Robot joint states: `/robot/arm_left/joint_states_single`, `/robot/arm_right/joint_states_single`
   - Camera images: `/realsense_top/color/image_raw/compressed`, `/realsense_left/color/image_raw/compressed`, `/realsense_right/color/image_raw/compressed`
2. **Policy Inference**: Sends observations to the policy server and receives action predictions

   - Processes camera images (resizing, normalization)
   - Combines multi-modal observations (images + proprioception)
   - Queries the policy server for action predictions
3. **Action Execution**: Publishes predicted actions to robot command topics

   - Left arm actions: `/robot/arm_left/vla_pos_cmd`
   - Right arm actions: `/robot/arm_right/vla_pos_cmd`

### System Architecture

```
┌─────────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Robot Hardware     │◄─────│  ROS Bridge      │◄─────│  VLA Policy     │
│  (Dual Arms +       │ CAN/ │  (piper_bridge)  │ ROS  │  Controller     │
│   RealSense Cameras)│ USB  │                  │Topics│ (piper_pi05_    │
│                     │      │                  │      │  main.py)       │
└─────────────────────┘      └──────────────────┘      └─────────────────┘
                                      │                         │
                                      │ Observations            │ Actions
                                      ▼                         ▼
                             /robot/arm_*/             /robot/arm_*/
                             joint_states              vla_pos_cmd
                             /realsense_*/
                             color/image_raw
                                                                │
                                                                │ HTTP
                                                                ▼
                                                       ┌─────────────────┐
                                                       │  Policy Server  │
                                                       │  (serve_policy  │
                                                       │   .py)          │
                                                       └─────────────────┘
```

### Deployment Workflow

1. **Terminal 1**: Start the policy server (Inference)

```bash
  cd openpi

  uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_npm_lora --policy.dir=/home/zeno/NPM-VLA-Project/NPM-VLA/openpi/checkpoints/pi05_npm_lora/push_block_pi05_new/

```

2. **Terminal 2**: Launch robot control system

```bash
  cd <piper_ros>

  source devel/setup.bash

  export ROS_MASTER_URI=http://localhost:11311
  # remember to setup the port before roslaunch
  roslaunch piper_bridge start_robot_all.launch \
  ranger_can_port:=can0 \
  left_can_port:=can_left \
  right_can_port:=can_right \
  enable_ranger:=false \
  enable_paddle2ranger:=false \
  enable_dual_arm:=true \
  enable_cameras:=true \
  enable_rviz:=true \
  camera_left_usb_port:=2-1 \
  camera_right_usb_port:=2-8 \
  camera_top_usb_port:=2-2
```

1. Terminal 3: Run VLA policy controller

```bash
  source .venv/bin/activate

  export ROS_MASTER_URI=http://localhost:11311
  
  uv run scripts/piper_pi05_main.py
```

### Alternative: Diffusion Policy (IL Method)

If using Imitation Learning methods like Diffusion Policy instead of VLA, we only need to set up 2 terminals (no policy server needed).

#### Prerequisites

- Trained Diffusion Policy checkpoint (see training section below)
- LeRobot environment with diffusion policy support
- ROS environment properly configured

#### Training Diffusion Policy

1. **Activate LeRobot environment**:

```bash
conda activate lerobot
```

2. **Train the policy**:

```bash
cd IL_policies

# Train diffusion policy 
python train_diffusion_policy.py \
  --dataset-repo-id "Anlorla/sweep2E_lerobot30" \
  --output-dir "./checkpoints/sweep2E_dp" \
  --num-epochs 3000 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --horizon 16 \
  --n-action-steps 8
```

**Training Parameters**:

- `--dataset-repo-id`: Hugging Face dataset repository ID
- `--output-dir`: Directory to save checkpoints
- `--num-epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--horizon`: Prediction horizon (number of future steps)
- `--n-action-steps`: Number of action steps to execute per prediction

3. **Evaluate offline** (optional):

```bash
# Run offline evaluation to check prediction accuracy
python eval_dp_offline.py \
  --checkpoint-dir "./checkpoints/sweep2E_dp/checkpoints/040000/pretrained_model" \
  --dataset-repo-id "Anlorla/sweep2E_lerobot30" \
  --num-samples 3200
```

See `Debug.md` for detailed offline evaluation metrics and analysis.

#### Deployment with Diffusion Policy

**Terminal 1**: Launch ROS control node

```bash
cd <piper_ros_workspace>
source devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311

# Remember to setup CAN ports before roslaunch
roslaunch piper_bridge start_robot_all.launch \
  ranger_can_port:=can0 \
  left_can_port:=can_left \
  right_can_port:=can_right \
  enable_ranger:=false \
  enable_paddle2ranger:=false \
  enable_dual_arm:=true \
  enable_cameras:=true \
  enable_rviz:=true \
  camera_left_usb_port:=2-1 \
  camera_right_usb_port:=2-8 \
  camera_top_usb_port:=2-2
```

**Terminal 2**: Run Diffusion Policy controller

```bash
conda activate lerobot
export ROS_MASTER_URI=http://localhost:11311

cd IL_policies
python piper_dp_main.py
```

**Script Configuration** (`piper_dp_main.py`):

Edit the checkpoint path in the script:

```python
# Line ~211 in piper_dp_main.py
ckpt_dir = "/home/jovyan/workspace/IL_policies/checkpoints/sweep2E_dp/checkpoints/040000/pretrained_model"
```

**Control Parameters** (adjustable in `piper_dp_main.py`):

```python
# Control frequency
rate = rospy.Rate(10)  # 10 Hz (start with 0.5 Hz for initial testing)

# Safety clipping
MAX_JOINT_DELTA = 0.15  # Maximum joint change per step (radians)
ENABLE_ACTION_CLIPPING = True

# EMA smoothing
ENABLE_SMOOTHING = True
SMOOTHING_ALPHA = 0.3  # Lower = smoother but slower response
```

## Troubleshooting

### Missing Library: libgthread-2.0.so.0

**Error:**

```
libgthread-2.0.so.0: cannot open shared object file: No such file or directory
```

**Solution:**

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0
```

### TorchCodec FFmpeg Compatibility Issues

**Error:**

```
RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed in your environment
  2. PyTorch version is not compatible with TorchCodec
  3. FFmpeg libraries not found (libavutil.so.*)
```

**Solution:**

Switch to an alternative video backend instead of TorchCodec:

```bash
# Option 1: Use torchvision backend
export LEROBOT_VIDEO_BACKEND=torchvision

# Option 2: Use pyav backend
export LEROBOT_VIDEO_BACKEND=pyav
```

Add this export to your shell profile for persistence:

```bash
# For bash
echo 'export LEROBOT_VIDEO_BACKEND=torchvision' >> ~/.bashrc
source ~/.bashrc

# For zsh
echo 'export LEROBOT_VIDEO_BACKEND=torchvision' >> ~/.zshrc
source ~/.zshrc
```

**Alternative:** Replace the video_utils.py file as described in the [Configuration](#configuration) section.

### LeRobot Dataset Version Compatibility

**Error:**

```
BackwardCompatibilityError: The dataset you requested is in 2.1 format.
We introduced a new format since v3.0 which is not backward compatible with v2.1.
```

**Solution:**

1. **Clear Hugging Face cache** (backup important files first):

```bash
# Check cache location
ls ~/.cache/huggingface/

# Remove dataset cache (be careful!)
rm -rf ~/.cache/huggingface/hub/datasets--<your-dataset-name>
```

2. **Convert dataset from v2.1 to v3.0**:

```bash
python utils/convert_dataset_v21_to_v30.py \
    --src-repo-id=your-username/dataset-name \
    --dst-repo-id=your-username/dataset-name-v3
```

**Note:** This conversion process will:

- Download the v2.1 dataset
- Convert data and video formats
- Generate proper metadata
- Push the converted v3.0 dataset to Hugging Face Hub

### Network and SSL Issues

**Error 1: SSL Connection Error**

```
SSLError: EOF occurred in violation of protocol
```

**Solution:**

Don't use VSCode Remote SSH for downloading large files. Use a direct shell connection instead:

```bash
# SSH directly into the machine
ssh user@hostname

# Then run your download/training commands
cd NPM-VLA/openpi
uv run scripts/train.py ...
```

### ROS Connection Issues

**Error:**

```
Unable to register with master node
```

**Solution:**

Ensure ROS_MASTER_URI is properly set:

```bash
# Check current setting
echo $ROS_MASTER_URI

# Set to localhost
export ROS_MASTER_URI=http://localhost:11311

# Verify connection
rostopic list
```

unable to enable robot arm:

1. restart workstation-	4
2. recharge arms
3. plug out can
4. re launch
5. when record, teleop arms using 示教

launch config file:
ssh slave:　code /home/zeno/piper_ros/src/zeno-wholebody-teleop/common/piper_ctrl/launch/piper_dual_robot.launch

ros_bridge

硬件：

不要上电拔插，先插上再上电，先断电再拔掉。

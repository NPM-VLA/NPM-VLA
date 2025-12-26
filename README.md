# NPM-VLA

This project is based on [OpenPI](https://github.com/Physical-Intelligence/openpi) and follows the official OpenPI workflow for robot learning and policy training.

> **💡 Tip:** If you encounter any issues during setup, training, or deployment, please refer to the [Troubleshooting](#troubleshooting) section first.

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

### Data Collection (Teleoperation)

Before collecting data through teleoperation, ensure proper network configuration between the two machines.

#### Prerequisites

Configure network communication between master and slave machines by editing `/etc/hosts`:

```bash
sudo vim /etc/hosts
```

Add the IP addresses and hostnames of both machines:

```
# Example:
192.168.4.161  zeno-teleop-master
192.168.4.162  zeno-teleop
```

#### Recording Data

Use the recording script `record.sh` to collect teleoperation data:

```bash
bash record.sh
```

The recorded data will be saved in ROS bag format (`.bag` files).

#### Format Conversion

After recording, convert the bag files to the appropriate LeRobot format based on your training method:

- **For OpenPI training**: Convert to LeRobot 2.1 format
- **For ACT/Diffusion Policy training**: Convert to LeRobot 3.0 format

See the [Data Conversion](#data-conversion) section below for detailed conversion instructions.

### Data Conversion

#### Convert ROS Bag to LeRobot 2.1 Format

Convert ROS bag files to LeRobot 2.1 format for OpenPI training:

```bash
python utils/convert_bag2lerobot21_dualarm.py
```

Note: If there are multiple directories, use `utils\convert_then_combine.py` instead.

#### Convert LeRobot 2.1 to 3.0 Format

For ACT/Diffusion Policy training, convert LeRobot 2.1 datasets to 3.0 format:

```bash
python utils/convert_dataset_v21_to_v30.py \
    --src-repo-id=your-username/dataset-name \
    --dst-repo-id=your-username/dataset-name-v3
```

**Parameters:**

- `--src-repo-id`: Source repository ID (LeRobot 2.1 format dataset)
- `--dst-repo-id`: Destination repository ID for the converted 3.0 format dataset

**What this conversion does:**

- Generates per-episode statistics and writes them in `episodes_stats.jsonl`
- Updates codebase version in `info.json`
- Removes deprecated `stats.json`
- Pushes the new version to the hub with "v3.0" tag

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

### Dataset Download

If there is no need to convert, we can directly download datasets as below:

**On Saturn Cloud:**

```bash
huggingface-cli download --resume-download Anlorla/sweep2cross_lerobot21_prim_enriched --local-dir  ~/.cache/huggingface/lerobot/Anlorla/sweep2cross_lerobot21_prim_enriched --repo-type dataset
```

**On Vast.ai:**

```bash
huggingface-cli download --resume-download Anlorla/sweep2cross_lerobot21_masked_enriched--local-dir  /workspace/.hf_home/lerobot/Anlorla/sweep2cross_lerobot21_masked_enriched --repo-type dataset
```

## Training

### 0. Configure Training Settings

Before training, you need to configure your training settings in `src/openpi/training/config.py`.

#### Create Your Custom TrainConfig

You can copy an existing configuration (like `pi0_npm` or `pi05_npm`) and create your own:

```python
 # In src/openpi/training/config.py, add to _CONFIGS list:

TrainConfig(
    name="pi0_npm",  # ** Change this to your custom config name
    # Dual-arm robot with 14-dim actions (7 per arm) and 16-dim state (8 per arm)
    model=pi0_config.Pi0Config(
        pi05=False, # ** Set True for Pi0.5, False for Pi0 base model
        action_horizon=10, # ** Number of future action steps to predict (typically 10-20)
        discrete_state_input=False,
    ),
    data=LeRobotZenoDataConfig(
        repo_id="Anlorla/sweep2E_alarm_v1_primitives_200",  # ** Your HuggingFace dataset repository ID
        base_config=DataConfig(
            prompt_from_task=True,
            action_sequence_keys=("action",),  # Specify the action key from dataset
        ),
        extra_delta_transform=False,
    ),
    batch_size=32, # ** Adjust based on GPU memory (16/32/64)
    lr_schedule=_optimizer.CosineDecaySchedule( # ** Learning rate schedule configuration
        warmup_steps=10_000, # ** See warmup_steps guidelines below
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    pytorch_weight_path=None,  # not use for now
    num_train_steps=12_000, # ** Total training steps (depends on dataset size and desired epochs)
),
```

#### Key Configuration Parameters

**Model Configuration:**

- `pi05`: Set to `True` for Pi0.5 model, `False` for Pi0
- `action_horizon`: Number of future action steps to predict
- `discrete_state_input`: Use discrete state input (default: `False`)

**Data Configuration:**

- `repo_id`: Your Hugging Face dataset repository ID (e.g., `"your-username/your-dataset"`)
- `prompt_from_task`: Load task instructions from dataset
- `action_sequence_keys`: Keys in dataset containing action sequences
- `extra_delta_transform`: Apply delta action transform (set based on your data format)

**Training Hyperparameters:**

- `batch_size`: Training batch size (adjust based on GPU memory)
- `warmup_steps`: Number of warmup steps for learning rate schedule
- `peak_lr`: Peak learning rate
- `num_train_steps`: Total number of training steps

**Model Initialization:**

- `weight_loader`: Path to pretrained checkpoint to initialize from
  - Pi0 base: `"gs://openpi-assets/checkpoints/pi0_base/params"`
  - Pi0.5 base: `"gs://openpi-assets/checkpoints/pi05_base/params"`

**Warmup Steps Guidelines:**

The `warmup_steps` parameter controls the learning rate warm-up period. Recommended values:

- **Full Fine-tuning**: Set `warmup_steps` to approximately **30% of `num_train_steps`**
  - Example: If `num_train_steps=12_000`, use `warmup_steps=3_600`
- **LoRA Fine-tuning**: Set `warmup_steps` to approximately **5-10% of `num_train_steps`**
  - Example: If `num_train_steps=12_000`, use `warmup_steps=600` to `1_200`

**Monitoring Training:**

After training starts, you can monitor your training progress and configuration using Weights & Biases (wandb):

1. Check the terminal output for the wandb run URL
2. Visit the URL to view real-time training metrics, losses, and hyperparameter configuration
3. All training configurations will be automatically logged to wandb for reproducibility

**Note:** Make sure your `repo_id` matches the dataset you uploaded or downloaded in the [Data Conversion](#data-conversion) or [Dataset Download](#dataset-download) sections.

### 1. Compute Normalization Statistics

Before training, compute the normalization statistics for your dataset:

```bash
cd openpi
uv run python scripts/compute_norm_stats.py --config-name pi05_npm
```

### 2. Start Training

```bash
cd openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_npm --exp-name=sweep2cross --overwrite
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

### 3. Upload Checkpoints to HuggingFace

After training completes, it's important to upload your trained checkpoints to HuggingFace Hub for version control and easy deployment across different machines.

**Upload Script:**

Use the `upload_ckpt_to_hf.py` utility script to upload your checkpoint:

```bash
python ../utils/upload_ckpt_to_hf.py \
    --repo_id Anlorla/pi05_sweep_lora_v1 \
    --ckpt_dir /home/jovyan/workspace/openpi/checkpoints/pi05_npm_lora/sweep_100_v1_lora/11999/ \
    --repo_type model
```

**Parameters:**

- `--repo_id`: Your HuggingFace repository ID (format: `username/model-name`)
- `--ckpt_dir`: Local path to the checkpoint directory (typically under `checkpoints/<config_name>/<exp_name>/<step>/`)
- `--repo_type`: Repository type (use `model` for policy checkpoints)

**Example:**

```bash
# Upload a full fine-tuning checkpoint
python ../utils/upload_ckpt_to_hf.py \
    --repo_id your-username/pi05_sweep_full_v1 \
    --ckpt_dir /path/to/openpi/checkpoints/pi05_npm/sweep_experiment/12000/ \
    --repo_type model

# Upload a LoRA checkpoint
python ../utils/upload_ckpt_to_hf.py \
    --repo_id your-username/pi0_task_lora_v2 \
    --ckpt_dir /path/to/openpi/checkpoints/pi0_npm_lora/task_name/5999/ \
    --repo_type model
```

**Notes:**

- Make sure you're logged into HuggingFace CLI: `huggingface-cli login`
- The checkpoint directory should contain all necessary model files (params, config, etc.)
- Consider using descriptive repository names that indicate the model type, task, and version
- You can upload multiple checkpoints from the same training run to compare different steps

## Inference

Before running inference, you need to download the trained model weights from HuggingFace Hub.

### Download Model Weights

There are three methods to download model weights from HuggingFace, depending on your network setup:

#### Method 1: HuggingFace with VPN

If you have access to VPN, you can directly download from HuggingFace Hub:

```bash
# Set up proxy (adjust based on your proxy configuration)
# Option A: Use host machine proxy
export http_proxy=http://127.0.0.1:7890
export https_proxy=https://127.0.0.1:7890

# Option B: Use LAN proxy
export http_proxy=http://192.168.x.x:7890
export https_proxy=https://192.168.x.x:7890

# Download the model
huggingface-cli download --resume-download Anlorla/pi05_recover_full_v1 \
    --local-dir /home/zeno/NPM-VLA-Project/NPM-VLA/openpi/checkpoints/pi05_npm \
    --repo-type model
```

#### Method 2: HuggingFace Mirror

Use a mirror site to bypass network restrictions:

```bash
# Set HuggingFace endpoint to mirror
export HF_ENDPOINT="https://hf-mirror.com"

# Download the model
huggingface-cli download --resume-download Anlorla/pi05_recover_full_v1 \
    --local-dir /home/zeno/NPM-VLA-Project/NPM-VLA/openpi/checkpoints/pi05_npm \
    --repo-type model
```

#### Method 3: Dropbox (Coming Soon)

> **Note:** Dropbox download method is under development and will be available in a future update.

**Download Parameters:**

- `--resume-download`: Resume interrupted downloads (recommended for large models)
- `--local-dir`: Local directory to save the model (should match your checkpoint path in inference config)
- `--repo-type model`: Specify that you're downloading a model (not a dataset)

**Important Notes:**

- Make sure the `--local-dir` path matches the `--policy.dir` path you'll use in the inference command
- The download may take some time depending on model size and network speed
- Use `--resume-download` to safely resume if the download is interrupted

### Start Inference

After downloading model weights, run inference using the trained policy checkpoint. The policy server loads the trained model and provides action predictions based on observations and prompts.

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
  --policy.dir=/home/zeno/NPM-VLA-Project/NPM-VLA/openpi/checkpoints/pi05_npm_lora/push_block_dual
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

### Real-Time Chunking (RTC) Policy Server

For improved real-time performance with action chunking, use the RTC-capable policy server:

```bash
cd openpi
uv run scripts/serve_policy_rtc.py policy:checkpoint \
  --policy.config=pi05_npm \
  --policy.dir=/home/zeno/NPM-VLA-Project/NPM-VLA/openpi/checkpoints/pi05_npm/pi05_sweep2cross_lerobot21_prim_v2
```

**What is RTC?**

Real-Time Chunking (RTC) implements the algorithm from "Real-Time Execution of Action Chunking Flow Policies" (Black et al., 2025). It improves policy execution by:

1. **Guided Inference**: Uses VJP (Vector-Jacobian Product) to align new action chunks with previously executed actions
2. **Prefix Attention**: Applies soft-masking to ensure smooth transitions between action chunks
3. **Adaptive Replanning**: Dynamically adjusts execution horizon based on observed inference delays

**Key RTC Parameters:**

- `control_freq`: Control loop frequency (default: 25 Hz)
- `action_horizon`: Prediction horizon H (default: 20 steps)
- `s_min`: Minimum execution horizon (default: 8 steps)
- `delay_buf_size`: Delay buffer size for adaptive scheduling (default: 10)
- `num_flow_steps`: Number of denoising steps (default: 5)
- `max_guidance_weight`: Maximum guidance strength β (default: 5.0)
- `prefix_attention_schedule`: Weight decay schedule ("exp", "linear", default: "exp")

**RTC vs Standard Inference:**

| Feature               | Standard        | RTC                 |
| --------------------- | --------------- | ------------------- |
| Replan frequency      | Fixed           | Adaptive            |
| Action smoothness     | May have jumps  | Smooth transitions  |
| Latency handling      | No compensation | Delay-aware         |
| Real-time feasibility | Not guaranteed  | Constraint-enforced |

**When to use RTC:**

- ✅ High-frequency control tasks (>20 Hz)
- ✅ Tasks requiring smooth action execution
- ✅ Environments with variable network latency
- ⚠️ Requires model trained with flow matching (Pi0, OpenVLA)
- ⚠️ Adds ~10-20% inference overhead due to guided sampling

## Deployment Workflow

This section covers deploying the trained policy on real robot hardware. The deployment process involves setting up the robot control system and connecting it with the trained VLA policy.

### Prerequisites

- **⚠️ Model Weights Downloaded**: Before deployment, you MUST download the trained model weights from HuggingFace Hub following the instructions in the [Download Model Weights](#download-model-weights) section
- Trained policy checkpoint (see [Training](#training) and [Inference](#inference) sections)
- Policy server running (see [Inference](#inference) section, remember to modify `src\openpi\training\config.py`)
- Robot hardware setup (refer to [zeno-wholebody-teleop](https://github.com/Jeong-zju/zeno-wholebody-teleop))
- ROS environment properly configured

### 0. Download Model Weights (If Not Already Done)

**Before starting deployment**, ensure you have downloaded the trained model weights. If you haven't done so, follow the [Download Model Weights](#download-model-weights) section in the Inference chapter.

**Quick Download Example:**

```bash
# Method 1: HuggingFace with VPN
export http_proxy=http://127.0.0.1:7890
export https_proxy=https://127.0.0.1:7890
huggingface-cli download --resume-download your-username/your-model \
    --local-dir /home/zeno/NPM-VLA-Project/NPM-VLA/openpi/checkpoints/pi05_npm \
    --repo-type model

# Method 2: HuggingFace Mirror
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download --resume-download your-username/your-model \
    --local-dir /home/zeno/NPM-VLA-Project/NPM-VLA/openpi/checkpoints/pi05_npm \
    --repo-type model
```

### 1. Start Policy Server

Start the policy server before configuring and launching the robot system:

```bash
cd openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_npm\
  --policy.dir=/home/zeno/NPM-VLA-Project/NPM-VLA/openpi/checkpoints/pi05_npm/pi05_sweep2cross_enriched
```

See the [Inference](#inference) section for more details on policy server configuration.

### 1. Configure Launch Files

Before starting the robot, you need to configure the ROS launch files to switch between different operation modes: teleoperation, VLA testing with gripper, or VLA testing without gripper.

**Configuration File Location:**

Edit the main configuration file:

```bash
vim /home/zeno/piper_ros/src/zeno-wholebody-teleop/common/piper_ctrl/config/piper_dual.yaml
```

**Configuration Modes:**

The key configuration section is the `remap` settings for each arm. Below are three common modes (using right arm as example):

#### Mode 1: VLA Testing with Gripper

Use this mode when testing the VLA policy with gripper control:

```yaml
remap:
  joint_pos_cmd_to: "/robot/arm_right/vla_joint_cmd"
  gripper_pos_cmd_to: "/robot/arm_right/vla_joint_cmd"    # Gripper controlled by VLA
  # gripper_pos_cmd_to: "/robot/arm_right/joint_pos_cmd"  # Commented out
```

**What this does:** Both arm joints and gripper are controlled by the VLA policy output on `/robot/arm_right/vla_joint_cmd` topic.

#### Mode 2: VLA Testing without Gripper

Use this mode when testing the VLA policy without gripper control (gripper controlled separately):

```yaml
remap:
  joint_pos_cmd_to: "/robot/arm_right/vla_joint_cmd"
  gripper_pos_cmd_to: "/robot/arm_right/joint_pos_cmd"    # Gripper controlled separately
  # gripper_pos_cmd_to: "/robot/arm_right/vla_joint_cmd"  # Commented out
```

**What this does:** Arm joints are controlled by VLA policy, but the gripper is controlled separately through the standard control interface.

#### Mode 3: Teleoperation Mode

Use this mode for manual teleoperation (data collection or manual control):

```yaml
remap:
  # joint_pos_cmd_to: "/robot/arm_right/vla_joint_cmd"     # Commented out
  # gripper_pos_cmd_to: "/robot/arm_right/joint_pos_cmd"   # Commented out
  # gripper_pos_cmd_to: "/robot/arm_right/vla_joint_cmd"   # Commented out
  joint_pos_cmd_to: "/teleop/arm_right/joint_states_single"
  gripper_pos_cmd_to: "/teleop/arm_right/joint_states_single"
```

**What this does:** Both arm and gripper are controlled by teleoperation commands for manual operation.

**Important Notes:**

- Apply the same configuration for both left and right arms in the file
- **⚠️ CRITICAL: After modifying the YAML configuration, you MUST source the workspace before launching:**
  ```bash
  source devel/setup.bash
  ```
  Without sourcing, the configuration changes will not take effect!
- Remember to restart the ROS nodes after modifying the configuration
- For dual-arm tasks, ensure both arms use consistent control modes

**Quick Reference:**

| Mode | Arm Control | Gripper Control | Use Case |
|------|-------------|-----------------|----------|
| VLA with Gripper | VLA | VLA | Full VLA policy control |
| VLA without Gripper | VLA | Separate | VLA arm control only |
| Teleoperation | Teleop | Teleop | Data collection/Manual control |

### 2. Start Robot Control System

Source the ROS workspace and launch the robot control nodes:

```bash
# Source the workspace
source devel/setup.bash

bash can_activate.sh can_left 1000000 "1-8.3:1.0"
bash can_activate.sh can_right 1000000 "1-8.4:1.0"

export ROS_MASTER_URI=http://localhost:11311

# Launch the robot with all sensors
roslaunch robot_setup start_robot_all.launch ranger_can_port:=can0 left_can_port:=can_left right_can_port:=can_right enable_ranger:=false enable_paddle2ranger:=false enable_dual_arm:=true enable_cameras:=true enable_rviz:=true enable_gravity_compensation:=false camera_left_usb_port:=2-1 camera_right_usb_port:=2-8 camera_top_usb_port:=2-2
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

# Run the VLA policy controller (Standard)
uv run scripts/piper_pi05_main.py

# OR: Run with Real-Time Chunking (RTC) for improved performance
uv run scripts/piper_main_rtc.py
```

**Script Options:**

- `piper_pi05_main.py`: Standard policy controller with simple action chunking
- `piper_main_rtc.py`: RTC-enabled controller with adaptive execution and guided inference

**What these scripts do:**

1. **Observation Collection**: Subscribes to robot state and camera topics to gather observations

   - Robot joint states: `/robot/arm_left/joint_states_single`, `/robot/arm_right/joint_states_single`
   - Camera images: `/realsense_top/color/image_raw/compressed`, `/realsense_left/color/image_raw/compressed`, `/realsense_right/color/image_raw/compressed`
2. **Policy Inference**: Sends observations to the policy server and receives action predictions

   - Processes camera images (resizing, normalization)
   - Combines multi-modal observations (images + proprioception)
   - Queries the policy server for action predictions
3. **Action Execution**: Publishes predicted actions to robot command topics

   - Left arm actions: `/robot/arm_left/vla_pos_cmd` (or `/vla_joint_cmd` for RTC)
   - Right arm actions: `/robot/arm_right/vla_pos_cmd` (or `/vla_joint_cmd` for RTC)

**RTC-specific behavior:**

- **Background inference**: Runs policy inference in a separate thread
- **Adaptive replanning**: Adjusts execution horizon based on observed delays
- **Delay compensation**: Maintains real-time constraints (d ≤ s ≤ H - d)
- **Smooth transitions**: Uses prefix attention to align action chunks

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

## Alternative: Diffusion Policy (IL Method)

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
(lerobot) zeno@zeno-teleop ~/NPM-VLA-Project/NPM-VLA/IL_policies (main) $ python piper_act_main.py
[INFO] [1766063733.469552]: Loading ACT Policy from: /home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/push_block_act
[INFO] [1766063734.133805]:   → Loading weights from model.safetensors
[INFO] [1766063734.330887]:   → Loading state normalizer from policy_preprocessor_step_3_normalizer_processor.safetensors
Traceback (most recent call last):
  File "/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/piper_act_main.py", line 505, in <module>
    main()
  File "/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/piper_act_main.py", line 295, in main
    load_normalization_stats(ckpt_path)
  File "/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/piper_act_main.py", line 58, in load_normalization_stats
    STATE_MEAN = stats['mean'].numpy().astype(np.float32)
KeyError: 'mean'
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

### Unable to Enable Robot Arms

If the robot arms fail to enable, try the following steps:

Always power off **before** unplugging cables
Always plug in cables **before** powering on

1. Power off arms
2. Unplug the CAN cables
3. Recharge the robot arms
4. Re-launch the system

### ROS launch file

For the latest version of launch file of this program, refer to:
https://github.com/Jeong-zju/zeno-wholebody-teleop/tree/master

### Camara "No Image"

After configuration, we need to source ~/.bashrc even if we don't modify anything.(Don't know why at present)

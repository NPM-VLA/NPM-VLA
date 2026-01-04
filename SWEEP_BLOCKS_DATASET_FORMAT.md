# LeRobot Dataset Format for Sweep Blocks with Mask

This document describes the required format for LeRobot datasets used to train the sweep blocks task with `sweep_mask` as the 4th image input.

## Overview

The dataset must follow the LeRobot format and include:
- 3 standard camera views (base/top, left wrist, right wrist)
- 1 sweep mask image (spatial guidance for the task)
- Robot state (joint positions)
- Actions (dual-arm joint commands)
- Task prompts (language instructions)

## Dataset Structure

### Required Files

```
dataset_name/
├── meta/
│   ├── info.json                 # Dataset metadata
│   └── tasks.json                # Task descriptions
├── data/
│   ├── chunk-000/
│   │   ├── observation.images.main/         # Base/top camera
│   │   ├── observation.images.secondary_0/  # Left wrist camera
│   │   ├── observation.images.secondary_1/  # Right wrist camera
│   │   ├── observation.images.sweep_mask/   # Sweep mask (NEW!)
│   │   └── ...
│   └── ...
└── videos/
    └── ...
```

## Required Fields

### 1. Images (4 cameras)

All images must be **224×224×3 RGB** format.

#### `observation.images.main` (Base/Top Camera)
- **Type**: RGB image
- **Shape**: `(224, 224, 3)`
- **Format**: PNG or JPEG
- **Description**: Main third-person view of the workspace

#### `observation.images.secondary_0` (Left Wrist Camera)
- **Type**: RGB image
- **Shape**: `(224, 224, 3)`
- **Format**: PNG or JPEG
- **Description**: Left arm wrist-mounted camera view

#### `observation.images.secondary_1` (Right Wrist Camera)
- **Type**: RGB image
- **Shape**: `(224, 224, 3)`
- **Format**: PNG or JPEG
- **Description**: Right arm wrist-mounted camera view

#### `observation.images.sweep_mask` (Sweep Mask - 4th Image)
- **Type**: RGB image (mask visualized as RGB)
- **Shape**: `(224, 224, 3)`
- **Format**: PNG or JPEG
- **Description**: Spatial guidance mask indicating sweep regions
- **Note**: This is treated as a regular RGB image by the vision encoder

**Important**: The sweep_mask should be an RGB visualization of the spatial mask (e.g., colored regions showing where to sweep blocks). It will be processed by the SigLIP vision encoder just like other camera views, generating 196 tokens.

### 2. State

#### `observation.state`
- **Type**: `float32` array
- **Shape**: `(14,)` for dual-arm robot
- **Description**: Joint positions for both arms
  ```
  [left_joint_0, left_joint_1, ..., left_joint_5, left_gripper,
   right_joint_0, right_joint_1, ..., right_joint_5, right_gripper]
  ```
- **Units**: Radians for joints, normalized [0,1] for grippers
- **Range**: Typically [-π, π] for joints, [0, 1] for grippers

### 3. Actions

#### `action`
- **Type**: `float32` array
- **Shape**: `(14,)` for dual-arm robot
- **Description**: Target joint positions or velocities
  ```
  [left_joint_0, left_joint_1, ..., left_joint_5, left_gripper,
   right_joint_0, right_joint_1, ..., right_joint_5, right_gripper]
  ```
- **Note**: Should match the action space used during data collection

### 4. Task Prompts

#### `task`
- **Type**: `string`
- **Example**: `"sweep blocks to target region"`
- **Description**: Natural language instruction for the task
- **Note**: Will be loaded as `prompt` during training if `prompt_from_task=True`

## Data Storage Format

LeRobot uses Apache Parquet format for efficient storage:

```python
# Example schema
{
    "observation.images.main": Image(shape=(224, 224, 3)),
    "observation.images.secondary_0": Image(shape=(224, 224, 3)),
    "observation.images.secondary_1": Image(shape=(224, 224, 3)),
    "observation.images.sweep_mask": Image(shape=(224, 224, 3)),  # NEW!
    "observation.state": Array(shape=(14,), dtype=float32),
    "action": Array(shape=(14,), dtype=float32),
    "task": String(),
    "episode_index": Int64(),
    "frame_index": Int64(),
    "timestamp": Float64(),
}
```

## Metadata Files

### `info.json`

```json
{
    "codebase_version": "v2.0",
    "robot_type": "piper_dual_arm",
    "total_episodes": 100,
    "total_frames": 50000,
    "fps": 10,
    "encoding": {
        "observation.images.main": {"type": "video"},
        "observation.images.secondary_0": {"type": "video"},
        "observation.images.secondary_1": {"type": "video"},
        "observation.images.sweep_mask": {"type": "video"}
    }
}
```

### `tasks.json`

```json
{
    "0": "sweep red blocks to left region",
    "1": "sweep blue blocks to right region",
    "2": "sweep all blocks to center region"
}
```

## Creating the sweep_mask

The `sweep_mask` should provide spatial guidance for the sweep task. Here are recommended approaches:

### Option 1: Target Region Visualization
```python
import numpy as np
import cv2

def create_sweep_mask(target_region_coords, image_size=(224, 224)):
    """
    Create RGB mask highlighting target region.

    Args:
        target_region_coords: List of (x, y) coordinates defining target polygon
        image_size: Output image size

    Returns:
        RGB mask image (H, W, 3)
    """
    mask = np.zeros((*image_size, 3), dtype=np.uint8)

    # Draw target region in green
    target_poly = np.array(target_region_coords, dtype=np.int32)
    cv2.fillPoly(mask, [target_poly], color=(0, 255, 0))

    # Optionally add block positions in red
    # cv2.circle(mask, block_pos, radius=10, color=(255, 0, 0), thickness=-1)

    return mask
```

### Option 2: Segmentation Mask
```python
def create_segmentation_mask(blocks_mask, target_mask):
    """
    Create RGB mask from segmentation.

    Args:
        blocks_mask: Binary mask of blocks to sweep (H, W)
        target_mask: Binary mask of target region (H, W)

    Returns:
        RGB mask image (H, W, 3)
    """
    mask = np.zeros((*blocks_mask.shape, 3), dtype=np.uint8)

    # Blocks in red
    mask[blocks_mask > 0] = [255, 0, 0]

    # Target region in green
    mask[target_mask > 0] = [0, 255, 0]

    return mask
```

### Option 3: Distance Field Visualization
```python
def create_distance_field_mask(goal_position, image_size=(224, 224)):
    """
    Create RGB mask visualizing distance to goal.

    Args:
        goal_position: (x, y) goal coordinates
        image_size: Output image size

    Returns:
        RGB mask image (H, W, 3)
    """
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    dist = np.sqrt((x - goal_position[0])**2 + (y - goal_position[1])**2)

    # Normalize to [0, 255]
    dist_norm = (dist / dist.max() * 255).astype(np.uint8)

    # Create heatmap (blue to red)
    mask = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)

    return mask
```

## Data Preprocessing Pipeline

When using this dataset with the training config, the data flows through:

```
1. LeRobot Dataset Load
   ↓
2. RepackTransform (config.py:381-395)
   - observation/image → observation.images.main
   - observation/wrist_image → observation.images.secondary_0
   - observation/right_wrist_image → observation.images.secondary_1
   - observation/sweep_mask → observation.images.sweep_mask  ← NEW!
   - observation/state → observation.state
   - actions → action
   - prompt → task
   ↓
3. PiperSweepInputs (piper_policy_mask.py:39-98)
   - Parse all 4 images to uint8 (H,W,C)
   - Create image dict with sweep_mask
   - Set all image_masks to True
   ↓
4. ResizeImages (224, 224)
   ↓
5. TokenizePrompt (max_len=200)
   - Tokenize prompt + discretized state
   ↓
6. Model Forward Pass
   - Each image → Vision Encoder → 196 tokens
   - Total: 4 × 196 = 784 image tokens
   - Plus language tokens (≤200) and action tokens (25)
```

## Token Budget Analysis

With 4 images, the token budget is:

| Token Type | Count | Limited by max_token_len? |
|------------|-------|---------------------------|
| Image tokens | 4 × 196 = 784 | ❌ No |
| Language tokens (prompt + state) | ≤ 200 | ✅ Yes |
| Action tokens | 25 | ❌ No |
| **Total** | **≤ 1009** | Sufficient! |

## Example: Converting Existing Dataset to Add sweep_mask

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from pathlib import Path

def add_sweep_mask_to_dataset(
    dataset_path: str,
    output_path: str,
    mask_generator_fn,
):
    """
    Add sweep_mask field to existing LeRobot dataset.

    Args:
        dataset_path: Path to original dataset
        output_path: Path to save modified dataset
        mask_generator_fn: Function to generate mask from episode data
    """
    # Load original dataset
    dataset = LeRobotDataset(dataset_path)

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each episode
    for episode_idx in range(dataset.num_episodes):
        episode_data = dataset.get_episode(episode_idx)

        # Generate sweep_mask for each frame
        sweep_masks = []
        for frame_idx in range(len(episode_data)):
            # Generate mask based on your task logic
            mask = mask_generator_fn(
                frame_data=episode_data[frame_idx],
                episode_idx=episode_idx,
                frame_idx=frame_idx
            )
            sweep_masks.append(mask)

        # Add to episode data
        episode_data["observation.images.sweep_mask"] = sweep_masks

    # Save modified dataset
    # ... (use LeRobot API to save)
```

## Validation Checklist

Before training, verify your dataset has:

- [ ] All 4 image fields present with correct names
- [ ] Images are 224×224×3 RGB format
- [ ] sweep_mask contains meaningful spatial information
- [ ] State dimension is 14 (7 per arm)
- [ ] Action dimension is 14 (7 per arm)
- [ ] Task prompts are present and descriptive
- [ ] No NaN or inf values in state/actions
- [ ] Episode lengths are reasonable (not too short/long)
- [ ] Metadata files (info.json, tasks.json) are complete

## Quick Validation Script

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt

def validate_sweep_dataset(repo_id: str):
    """Validate dataset has correct format for sweep blocks."""

    dataset = LeRobotDataset(repo_id)

    # Check required fields
    required_fields = [
        "observation.images.main",
        "observation.images.secondary_0",
        "observation.images.secondary_1",
        "observation.images.sweep_mask",  # NEW!
        "observation.state",
        "action",
        "task",
    ]

    for field in required_fields:
        assert field in dataset.features, f"Missing field: {field}"
        print(f"✓ {field}")

    # Check dimensions
    sample = dataset[0]
    assert sample["observation.state"].shape == (14,), "State should be 14-dim"
    assert sample["action"].shape == (14,), "Action should be 14-dim"

    # Visualize first frame with all 4 images
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(sample["observation.images.main"])
    axes[0, 0].set_title("Base Camera")
    axes[0, 1].imshow(sample["observation.images.secondary_0"])
    axes[0, 1].set_title("Left Wrist")
    axes[1, 0].imshow(sample["observation.images.secondary_1"])
    axes[1, 0].set_title("Right Wrist")
    axes[1, 1].imshow(sample["observation.images.sweep_mask"])
    axes[1, 1].set_title("Sweep Mask (4th Image)")
    plt.tight_layout()
    plt.savefig("dataset_validation.png")
    print("✓ Saved visualization to dataset_validation.png")

    print(f"\n✓ Dataset validation passed!")
    print(f"  - Episodes: {dataset.num_episodes}")
    print(f"  - Total frames: {len(dataset)}")
    print(f"  - Task: {sample['task']}")

# Run validation
validate_sweep_dataset("Anlorla/sweep2E")
```

## Training Config Reference

Use the following config name to train with sweep_mask:

```bash
python openpi/training/train.py pi05_npm_with_sweepmask \
  --exp_name=my_sweep_experiment \
  --data.repo_id=Anlorla/sweep2E
```

This config is defined in `config.py:1217-1249` and uses:
- **Model**: Pi05 with max_token_len=200
- **Data**: LeRobotZenoSweepDataConfig
- **Policy**: PiperSweepInputs (handles 4 images)
- **Action horizon**: 25
- **Batch size**: 256

## Troubleshooting

### Issue: "Missing field observation.images.sweep_mask"
**Solution**: Ensure your dataset includes the sweep_mask field with correct naming.

### Issue: "Image shape mismatch"
**Solution**: All images must be exactly 224×224×3. Resize during data collection or preprocessing.

### Issue: "Token length exceeded"
**Solution**: This shouldn't happen with 4 images. Check that max_token_len=200 in model config.

### Issue: "sweep_mask is all zeros"
**Solution**: Verify your mask generation logic is producing meaningful spatial information.

## References

- LeRobot Documentation: https://github.com/huggingface/lerobot
- OpenPI Training Guide: `openpi/README.md`
- Mask Usage Guide: `MASK_USAGE.md`
- Config Implementation: `openpi/src/openpi/training/config.py:358-425`
- Policy Implementation: `openpi/src/openpi/policies/piper_policy_mask.py`

---

**Last Updated**: 2026-01-05
**Author**: Claude Code
**Config Name**: `pi05_npm_with_sweepmask`
**Dataset Example**: `Anlorla/sweep2E`

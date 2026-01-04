# Pi05 训练引入 Mask 图片支持

## 概述

本文档说明如何在 pi05 训练中将 mask 作为第四张图片引入。我们创建了新的文件而不是修改原有代码，以保持向后兼容性。

## 修改文件列表

### 1. 新增文件

1. **`openpi/src/openpi/policies/piper_policy_mask.py`**
   - 新的数据转换类，支持 mask 作为第四张图片
   - 包含 `PiperMaskInputs` 和 `PiperMaskOutputs`

2. **`openpi/src/openpi/training/config_mask.py`**
   - 新的数据配置工厂类 `LeRobotPiperMaskDataConfig`
   - 包含完整的训练配置示例

3. **`MASK_USAGE.md`** (本文件)
   - 使用说明文档

## 数据格式要求

### 数据集中需要包含的字段

你的 LeRobot 数据集需要包含以下字段：

```python
{
    "observation/image": np.array,          # 主相机图片 (H, W, 3)
    "observation/wrist_image": np.array,    # 左手腕相机图片 (H, W, 3)
    "observation/right_wrist_image": np.array,  # 右手腕相机图片 (H, W, 3)
    "observation/mask_image": np.array,     # Mask 图片 (H, W, 3) - 新增！
    "observation/state": np.array,          # 关节状态 (14,)
    "observation/ee_pose": np.array,        # 末端执行器位姿 (14,)
    "actions": np.array,                    # 动作 (action_horizon, 14)
    "prompt": str,                          # 任务描述
}
```

### Mask 图片格式

- **格式**: RGB 图片，shape 为 `(H, W, 3)` 或 `(3, H, W)`
- **数据类型**: `uint8` (0-255) 或 `float` (0.0-1.0)
- **处理**: 会被自动转换为 `uint8` 格式，并调整为 `(H, W, 3)` 布局
- **Resize**: 会被自动 resize 到 224x224（与其他图片一致）

## 使用方法

### 方法 1: 使用独立的配置文件（推荐）

```python
from openpi.training.config_mask import create_pi05_npm_mask_config

# 创建配置
config = create_pi05_npm_mask_config()

# 修改数据集 ID
config = dataclasses.replace(
    config,
    data=dataclasses.replace(
        config.data,
        repo_id="your_hf_username/your_dataset_with_mask"
    )
)

# 使用配置进行训练
# ... training code ...
```

### 方法 2: 直接导入配置类

```python
from openpi.training.config_mask import LeRobotPiperMaskDataConfig
from openpi.training.config import TrainConfig
from openpi.models import pi0_config
from openpi.training import optimizer, weight_loaders

config = TrainConfig(
    name="my_pi05_mask_training",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=25,
        discrete_state_input=False,
        max_token_len=180,
    ),
    data=LeRobotPiperMaskDataConfig(
        repo_id="your_hf_username/your_dataset_with_mask",
        base_config=DataConfig(
            prompt_from_task=True,
            action_sequence_keys=("action",),
        ),
        extra_delta_transform=False,
        concat_ee_pose=True,
    ),
    batch_size=256,
    # ... 其他配置 ...
)
```

### 方法 3: 添加到主配置文件

如果你想将配置添加到主 `config.py` 文件的 `_CONFIGS` 列表中：

1. 在 `config.py` 顶部添加导入：
```python
import openpi.policies.piper_policy_mask as piper_policy_mask
```

2. 在 `_CONFIGS` 列表中添加新配置（在文件末尾，第 1270 行附近）：
```python
TrainConfig(
    name="pi05_npm_mask",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=25,
        discrete_state_input=False,
        max_token_len=180,
    ),
    data=LeRobotPiperMaskDataConfig(  # 需要在 config.py 中定义或导入
        repo_id="your_dataset_id",
        base_config=DataConfig(
            prompt_from_task=True,
            action_sequence_keys=("action",),
        ),
        extra_delta_transform=False,
        concat_ee_pose=True,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
),
```

## 数据流程说明

### 1. 数据加载
- LeRobot dataset 加载原始数据
- 包含 4 张图片：main, wrist, right_wrist, mask

### 2. Repack Transform
```python
{
    "observation/image": "observation.images.main",
    "observation/wrist_image": "observation.images.secondary_0",
    "observation/right_wrist_image": "observation.images.secondary_1",
    "observation/mask_image": "observation.images.mask",  # 新增
    "observation/state": "observation.state",
    "observation/ee_pose": "observation.ee_pose",
    "actions": "action",
    "prompt": "task",
}
```

### 3. Data Transform (PiperMaskInputs)
将数据转换为模型输入格式：
```python
{
    "state": np.array,  # (28,) if concat_ee_pose=True else (14,)
    "image": {
        "base_0_rgb": np.array,         # (224, 224, 3)
        "left_wrist_0_rgb": np.array,   # (224, 224, 3)
        "right_wrist_0_rgb": np.array,  # (224, 224, 3)
        "mask_rgb": np.array,           # (224, 224, 3) - 新增
    },
    "image_mask": {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
        "mask_rgb": True,  # 新增
    },
    "actions": np.array,  # (action_horizon, 14)
    "prompt": str,
}
```

### 4. Model Transform
- 图片 resize 到 224x224
- Tokenize prompt
- 标准化 state 和 actions

### 5. 模型处理
- 4 张图片都会被 Vision Encoder 处理
- 生成对应的 image tokens
- 与 language tokens 和 action tokens 一起输入 Transformer

## 配置参数说明

### LeRobotPiperMaskDataConfig 参数

- **`repo_id`**: HuggingFace 数据集 ID
- **`concat_ee_pose`**: 是否将 ee_pose 拼接到 joint state 后
  - `True`: state 为 28 维 (14 joint + 14 ee_pose)
  - `False`: state 为 14 维 (只有 joint)
- **`extra_delta_transform`**: 是否应用额外的 delta transform
  - 通常设为 `False`
- **`base_config`**: 基础数据配置
  - `prompt_from_task`: 是否从 task 字段读取 prompt
  - `action_sequence_keys`: 指定 action 的键名

### TrainConfig 参数

- **`model.pi05`**: 设为 `True` 使用 pi05 模型
- **`model.action_horizon`**: 动作序列长度（如 25）
- **`model.discrete_state_input`**: 是否使用离散状态输入（通常为 `False`）
- **`model.max_token_len`**: 最大 token 长度（推荐 180）
- **`batch_size`**: 批次大小（如 256）
- **`num_train_steps`**: 训练步数（如 30,000）

## 验证数据流

### 验证脚本

```python
from openpi.training.config_mask import create_pi05_npm_mask_config
from openpi.training import data_loader
import dataclasses

# 创建配置
config = create_pi05_npm_mask_config()

# 修改为你的数据集
config = dataclasses.replace(
    config,
    data=dataclasses.replace(
        config.data,
        repo_id="your_dataset_id"
    )
)

# 创建数据加载器
loader = data_loader.create_data_loader(
    config,
    shuffle=False,
    num_batches=1,
    skip_norm_stats=True,  # 如果还没有 norm stats
)

# 加载一个 batch
for observation, actions in loader:
    print("Observation keys:", observation.__dict__.keys())
    print("Images:", observation.images.keys())
    print("Image shapes:")
    for name, img in observation.images.items():
        print(f"  {name}: {img.shape}")
    print("State shape:", observation.state.shape)
    print("Actions shape:", actions.shape)
    break
```

期望输出：
```
Observation keys: dict_keys(['images', 'image_masks', 'state', 'tokenized_prompt', ...])
Images: dict_keys(['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb', 'mask_rgb'])
Image shapes:
  base_0_rgb: (batch, 224, 224, 3)
  left_wrist_0_rgb: (batch, 224, 224, 3)
  right_wrist_0_rgb: (batch, 224, 224, 3)
  mask_rgb: (batch, 224, 224, 3)
State shape: (batch, 28) 或 (batch, 14)
Actions shape: (batch, action_horizon, 14)
```

## 常见问题

### Q1: 如果数据集中没有 mask_image 怎么办？

如果你的数据集暂时没有 mask，可以：
1. 使用原来的 `LeRobotPiperDataConfig`（3 张图片）
2. 或者在数据集中添加一个占位的 mask 图片（全黑或全白）

### Q2: Mask 图片应该包含什么内容？

Mask 图片可以是：
- 语义分割 mask
- 深度图
- 注意力区域 mask
- 任何你认为对机器人控制有用的额外视觉信息

### Q3: 性能影响？

添加第 4 张图片会：
- 增加 ~25% 的图片编码时间
- 增加 ~25% 的显存使用
- 可能需要适当调整 batch_size

### Q4: 如何计算 norm stats？

运行以下命令计算归一化统计信息：
```bash
python scripts/compute_norm_stats.py --config-name=pi05_npm_mask
```

注意：需要先将配置添加到主 `config.py` 文件中。

## 训练命令示例

```bash
# 使用 JAX 训练
python openpi/scripts/train.py \
    --config-name pi05_npm_mask \
    --exp-name my_experiment \
    --batch-size 256 \
    --num-train-steps 30000

# 使用 PyTorch 训练
python openpi/scripts/train_pytorch.py \
    --config-name pi05_npm_mask \
    --exp-name my_experiment \
    --batch-size 256 \
    --num-train-steps 30000
```

## 技术细节

### 图片处理流程

1. **加载**: 从数据集加载 4 张图片
2. **解析**: 转换为 uint8 格式，shape (H, W, 3)
3. **Resize**: 调整到 224x224
4. **编码**: 通过 SigLIP Vision Encoder
5. **Tokens**: 生成 image tokens (每张图片约 196 tokens)
6. **注意力**: 所有 image tokens 可以互相 attend

### 内存占用估算

以 batch_size=256, action_horizon=25 为例：

- 4 张图片: 256 * 4 * 224 * 224 * 3 * 1 byte ≈ 154 MB
- Image tokens: 256 * 4 * 196 * hidden_dim * 4 bytes
- 总增加: 约 25-30% 显存

## 下一步

1. 准备包含 mask_image 字段的数据集
2. 上传到 HuggingFace
3. 使用本文档中的配置进行训练
4. 根据训练结果调整超参数

## 联系方式

如有问题，请查看：
- OpenPI 官方文档
- GitHub Issues
- 或联系团队成员

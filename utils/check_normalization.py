#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查归一化前后的数值信息。

用法示例：
    python check_normalization.py --config_name pi05_npm_lora
"""

import argparse
import numpy as np
import sys
import os

# 添加 openpi 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.shared.normalize as normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True, help="Config name to use")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to check")
    args = parser.parse_args()

    # 1. 加载配置
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 2. 尝试加载已保存的归一化统计信息
    norm_stats_path = config.assets_dirs / data_config.repo_id
    try:
        norm_stats = normalize.load(norm_stats_path)
        print(f"\n========== 已保存的归一化统计信息 ==========")
        print(f"路径: {norm_stats_path}")
        for key in ["state", "actions"]:
            if key in norm_stats:
                stats = norm_stats[key]
                mean = np.array(stats["mean"])
                std = np.array(stats["std"])
                print(f"\n{key}:")
                print(f"  mean shape: {mean.shape}, values: {mean}")
                print(f"  std shape:  {std.shape}, values: {std}")
    except Exception as e:
        print(f"\n警告: 无法加载归一化统计信息: {e}")
        norm_stats = None

    # 3. 创建 dataloader（带归一化和不带归一化）
    print(f"\n========== 创建 DataLoader ==========")

    # 3.1 不带归一化的 dataloader
    data_config_no_norm = data_config
    # 临时移除 normalize transform
    original_transforms = data_config.data_transforms.inputs
    data_config_no_norm.data_transforms.inputs = [
        t for t in original_transforms if not isinstance(t, normalize.Normalize)
    ]

    if data_config_no_norm.rlds_data_dir is not None:
        print("使用 RLDS 数据集")
        dataset_no_norm = _data_loader.create_rlds_dataset(
            data_config_no_norm, config.model.action_horizon, config.batch_size, shuffle=False
        )
    else:
        print("使用 Torch 数据集")
        dataset_no_norm = _data_loader.create_torch_dataset(
            data_config_no_norm, config.model.action_horizon, config.model
        )
        dataset_no_norm = _data_loader.TorchDataLoader(
            dataset_no_norm,
            local_batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

    # 3.2 带归一化的 dataloader (恢复原始 transforms)
    data_config.data_transforms.inputs = original_transforms

    if data_config.rlds_data_dir is not None:
        dataset_norm = _data_loader.create_rlds_dataset(
            data_config, config.model.action_horizon, config.batch_size, shuffle=False
        )
    else:
        dataset_norm = _data_loader.create_torch_dataset(
            data_config, config.model.action_horizon, config.model
        )
        dataset_norm = _data_loader.TorchDataLoader(
            dataset_norm,
            local_batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

    # 4. 采样几个 batch 进行对比
    print(f"\n========== 检查前 {args.num_batches} 个 batch ==========")

    iter_no_norm = iter(dataset_no_norm)
    iter_norm = iter(dataset_norm)

    for i in range(args.num_batches):
        try:
            batch_no_norm = next(iter_no_norm)
            batch_norm = next(iter_norm)
        except StopIteration:
            print(f"只有 {i} 个 batch 可用")
            break

        print(f"\n----- Batch {i} -----")

        # 检查 state
        if "state" in batch_no_norm:
            state_orig = np.array(batch_no_norm["state"])
            state_norm = np.array(batch_norm["state"])

            print(f"\nstate (原始数据):")
            print(f"  shape: {state_orig.shape}")
            print(f"  min:   {state_orig.min():.4f}")
            print(f"  max:   {state_orig.max():.4f}")
            print(f"  mean:  {state_orig.mean():.4f}")
            print(f"  std:   {state_orig.std():.4f}")
            print(f"  sample[0, 0]: {state_orig[0, 0]}")

            print(f"\nstate (归一化后):")
            print(f"  shape: {state_norm.shape}")
            print(f"  min:   {state_norm.min():.4f}")
            print(f"  max:   {state_norm.max():.4f}")
            print(f"  mean:  {state_norm.mean():.4f}")
            print(f"  std:   {state_norm.std():.4f}")
            print(f"  sample[0, 0]: {state_norm[0, 0]}")

        # 检查 actions
        if "actions" in batch_no_norm:
            actions_orig = np.array(batch_no_norm["actions"])
            actions_norm = np.array(batch_norm["actions"])

            print(f"\nactions (原始数据):")
            print(f"  shape: {actions_orig.shape}")
            print(f"  min:   {actions_orig.min():.4f}")
            print(f"  max:   {actions_orig.max():.4f}")
            print(f"  mean:  {actions_orig.mean():.4f}")
            print(f"  std:   {actions_orig.std():.4f}")
            print(f"  sample[0, 0]: {actions_orig[0, 0]}")

            print(f"\nactions (归一化后):")
            print(f"  shape: {actions_norm.shape}")
            print(f"  min:   {actions_norm.min():.4f}")
            print(f"  max:   {actions_norm.max():.4f}")
            print(f"  mean:  {actions_norm.mean():.4f}")
            print(f"  std:   {actions_norm.std():.4f}")
            print(f"  sample[0, 0]: {actions_norm[0, 0]}")

            # 验证归一化是否正确
            if norm_stats is not None and "actions" in norm_stats:
                mean = np.array(norm_stats["actions"]["mean"])
                std = np.array(norm_stats["actions"]["std"])
                # 手动计算归一化
                actions_manual = (actions_orig - mean) / std
                diff = np.abs(actions_manual - actions_norm).max()
                print(f"\n手动归一化 vs 实际归一化的最大差异: {diff:.6f}")
                if diff > 1e-3:
                    print(f"  警告：差异较大！可能归一化有问题")


if __name__ == "__main__":
    main()

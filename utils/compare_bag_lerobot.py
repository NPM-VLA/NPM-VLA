#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare one push_block bag with one LeRobot episode parquet.

用法示例：
    python compare_bag_lerobot.py \
        --bag "D:\\DESKTOP\\push_block_000.bag" \
        --parquet "D:\\DESKTOP\\episode_000000.parquet"
"""

import argparse
import numpy as np
import rosbag
import pandas as pd


ROBOT_LEFT_TOPIC = "/robot/arm_left/joint_states_single"
ROBOT_RIGHT_TOPIC = "/robot/arm_right/joint_states_single"
TELEOP_LEFT_TOPIC = "/teleop/arm_left/joint_states_single"
TELEOP_RIGHT_TOPIC = "/teleop/arm_right/joint_states_single"


def load_bag_joints(bag_path: str, left_topic: str, right_topic: str):
    """从 bag 里读左右臂 joint_states，返回 times, q (N, 14)."""
    times = []
    qs = []

    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[left_topic, right_topic]):
            q = np.array(msg.position, dtype=np.float32)
            # 只取前 7 维，和你训练时一致
            q = q[:7]

            if topic == left_topic:
                side = "L"
            else:
                side = "R"

            times.append((t.to_sec(), side, q))

    # 按时间排序
    times.sort(key=lambda x: x[0])

    # 把 left/right 拼成 14 维
    merged_times = []
    left_last = None
    right_last = None
    for t, side, q in times:
        if side == "L":
            left_last = q
        else:
            right_last = q
        if left_last is not None and right_last is not None:
            merged_times.append((t, np.concatenate([left_last, right_last], axis=0)))

    if not merged_times:
        raise RuntimeError("No merged L/R joint states found in bag.")

    ts = np.array([x[0] for x in merged_times], dtype=np.float64)
    qs = np.stack([x[1] for x in merged_times], axis=0)  # (N, 14)

    return ts, qs


def load_lerobot_episode(parquet_path: str):
    """从 LeRobot episode parquet 里读 observation.state 和 action."""
    df = pd.read_parquet(parquet_path)
    print(f"[parquet] columns: {list(df.columns)}")

    # 这里假设这两个列名和你生成时一致
    states_col = df["observation.state"].to_numpy()
    actions_col = df["action"].to_numpy()

    # 每一行是一个 14 维向量（可能是 list 或 ndarray）
    states = np.stack([np.array(x, dtype=np.float32) for x in states_col], axis=0)
    actions = np.stack([np.array(x, dtype=np.float32) for x in actions_col], axis=0)

    print(f"[parquet] states shape:  {states.shape}")
    print(f"[parquet] actions shape: {actions.shape}")

    return states, actions


def downsample_or_interp_bag_to_len(q_bag: np.ndarray, target_len: int):
    """
    简单对齐：把 bag 里的序列按 index 线性采样到和 episode 一样长。
    不追求严格时间对齐，只为了看数值大致是否一致。
    """
    n = len(q_bag)
    if n == 0:
        raise RuntimeError("Empty bag joint sequence")

    idx = np.linspace(0, n - 1, target_len).astype(int)
    return q_bag[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag", type=str, required=True, help="Path to push_block_000.bag"
    )
    parser.add_argument(
        "--parquet", type=str, required=True, help="Path to episode_000000.parquet"
    )
    args = parser.parse_args()

    # 1) 读 bag：robot joints 和 teleop joints
    print(f"[info] Loading bag: {args.bag}")
    t_robot, q_robot = load_bag_joints(args.bag, ROBOT_LEFT_TOPIC, ROBOT_RIGHT_TOPIC)
    t_teleop, q_teleop = load_bag_joints(
        args.bag, TELEOP_LEFT_TOPIC, TELEOP_RIGHT_TOPIC
    )
    print(f"[bag] robot joints:  {q_robot.shape}")
    print(f"[bag] teleop joints: {q_teleop.shape}")

    # 2) 读 parquet：episode 的 state/action
    states, actions = load_lerobot_episode(args.parquet)

    # 3) 把 bag 里的轨迹对齐到和 episode 一样长（按 index 采样）
    q_robot_ds = downsample_or_interp_bag_to_len(q_robot, len(states))
    q_tele_ds = downsample_or_interp_bag_to_len(q_teleop, len(actions))

    # 4) 计算误差统计
    err_state = np.linalg.norm(states - q_robot_ds, axis=1)
    err_action = np.linalg.norm(actions - q_tele_ds, axis=1)

    print("\n========== 简单误差统计（L2 per-frame） ==========")
    print(f"state vs robot:  mean={err_state.mean():.4f}, max={err_state.max():.4f}")
    print(f"action vs teleop: mean={err_action.mean():.4f}, max={err_action.max():.4f}")

    # 5) 打印前 5 帧做 sanity check
    print("\n========== 前 5 帧对比（左臂前 3 维） ==========")
    for i in range(min(5, len(states))):
        print(f"\nFrame {i}:")
        print("  state[0:3] (LeRobot)    :", states[i][:3])
        print("  robot[0:3] (bag-sampled):", q_robot_ds[i][:3])
        print("  action[0:3] (LeRobot)   :", actions[i][:3])
        print("  teleop[0:3] (bag-sampled):", q_tele_ds[i][:3])


if __name__ == "__main__":
    main()

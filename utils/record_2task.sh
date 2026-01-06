#!/usr/bin/env bash
set -euo pipefail

# =======================
# Config: output prefixes
# =======================
RECOVER_PREFIX="recover2yellow_v2/recover2yellow_v2_"
SWEEP_PREFIX="sweep2yellow_v2/sweep2yellow_v2_"

# Start task: "recover" or "sweep"
TASK="sweep"

# Ensure output dirs exist
mkdir -p "$(dirname "$RECOVER_PREFIX")" "$(dirname "$SWEEP_PREFIX")"

# Find next available index for a given prefix
find_next_idx() {
  local prefix="$1"
  local i=0
  while [[ -f "${prefix}$(printf "%03d" "$i").bag" ]]; do
    i=$((i+1))
  done
  echo "$i"
}

recover_idx="$(find_next_idx "$RECOVER_PREFIX")"
sweep_idx="$(find_next_idx "$SWEEP_PREFIX")"

# Topics to record (edit if needed)
TOPICS=(
  /robot/arm_left/end_pose
  /robot/arm_right/end_pose
  /robot/arm_left/joint_states_single
  /robot/arm_right/joint_states_single
  /robot/arm_left/pos_cmd
  /robot/arm_right/pos_cmd
  /teleop/arm_left/joint_states_single
  /teleop/arm_right/joint_states_single
  # New fisheye cameras (replacing realsense left/right)
  /fisheye_left/image_raw/compressed
  /fisheye_right/image_raw/compressed
  /fisheye_left/camera_info
  /fisheye_right/camera_info
  # Realsense top camera (unchanged)
  /realsense_top/color/image_raw/compressed
  /realsense_top/aligned_depth_to_color/image_raw/compressed
  /realsense_top/color/camera_info
  /realsense_top/aligned_depth_to_color/camera_info
  # New wide top camera
  /wide_top/image_raw/compressed
  /wide_top/camera_info
  # Old realsense left/right cameras (commented out)
  # /realsense_left/color/image_raw/compressed
  # /realsense_right/color/image_raw/compressed
  # /realsense_left/aligned_depth_to_color/image_raw/compressed
  # /realsense_right/aligned_depth_to_color/image_raw/compressed
  # /realsense_left/color/camera_info
  # /realsense_right/color/camera_info
  # /realsense_left/aligned_depth_to_color/camera_info
  # /realsense_right/aligned_depth_to_color/camera_info
)

echo "=== Alternating rosbag episodic recorder (RECOVER <-> SWEEP) ==="
echo "当前目录: $(pwd)"
echo "RECOVER 前缀: ${RECOVER_PREFIX}XXX.bag (next=$(printf "%03d" "$recover_idx"))"
echo "SWEEP   前缀: ${SWEEP_PREFIX}XXX.bag (next=$(printf "%03d" "$sweep_idx"))"
echo "操作说明："
echo "  回车     : 开始录制 / 停止录制"
echo "  录完后 n : 保留该条，并切换到下一任务"
echo "  录完后 r : 覆盖该条（删除重录，不切任务）"
echo "  录完后 q : 退出脚本"
echo "==============================================================="
echo

while true; do
  if [[ "$TASK" == "recover" ]]; then
    idx="$recover_idx"
    prefix="$RECOVER_PREFIX"
    task_name="RECOVER"
    next_task="SWEEP"
  else
    idx="$sweep_idx"
    prefix="$SWEEP_PREFIX"
    task_name="SWEEP"
    next_task="RECOVER"
  fi

  bag_name="${prefix}$(printf "%03d" "$idx").bag"

  echo "准备录制任务 = ${task_name} | index = ${idx} -> ${bag_name}"
  read -p "按回车开始录制（q 退出）: " cmd
  if [[ "${cmd:-}" == "q" ]]; then
    echo "退出。"
    exit 0
  fi

  echo
  echo "[REC] 任务=${task_name} 开始录制到: $bag_name"
  echo "      rosbag record -O \"$bag_name\" --bz2 -b 4096 <topics...>"
  echo

  rosbag record -O "$bag_name" --bz2 -b 4096 "${TOPICS[@]}" &
  BAG_PID=$!

  echo "[REC] rosbag PID = $BAG_PID"
  echo "[REC] 现在可以开始遥操作。录完后按回车停止。"
  read -p "" _

  echo "[REC] 正在停止 rosbag (SIGINT)..."
  kill -INT "$BAG_PID" 2>/dev/null || true
  wait "$BAG_PID" 2>/dev/null || true

  if [[ ! -f "$bag_name" ]]; then
    echo "[WARN] 找不到 $bag_name，可能录制失败，这一条将重录。"
    continue
  fi

  echo
  echo "录制完成: $bag_name"
  echo "选择： n = 保留并切到 ${next_task}   r = 删除重录本条   q = 退出"
  read -p "[n / r / q]: " post

  case "${post:-}" in
    r|R)
      echo "[DEL] 删除 $bag_name，准备重录 任务=${task_name} index=$idx"
      rm -f "$bag_name"
      # 不递增 idx，不切任务
      ;;
    q|Q)
      echo "退出。"
      exit 0
      ;;
    *)
      # 默认视为 n：保留并切任务 + 递增对应任务 idx
      if [[ "$TASK" == "recover" ]]; then
        recover_idx=$((recover_idx+1))
        TASK="sweep"
      else
        sweep_idx=$((sweep_idx+1))
        TASK="recover"
      fi
      ;;
  esac

  echo
done

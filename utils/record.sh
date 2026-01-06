#!/usr/bin/env bash


BASE_NAME="sweep2yellow_v2/sweep2yellow_v2_"

idx=0
while ls ${BASE_NAME}_$(printf "%03d" $idx).bag >/dev/null 2>&1; do
    idx=$((idx+1))
done

echo "=== Simple rosbag episodic recorder ==="
echo "当前目录: $(pwd)"
echo "文件前缀: ${BASE_NAME}_XXX.bag"
echo "操作说明："
echo "  回车     : 开始录制 / 停止录制"
echo "  录完后 n : 保留该条并进入下一条"
echo "  录完后 r : 覆盖该条（删除重录）"
echo "  录完后 q : 退出脚本"
echo "======================================="
echo

while true; do
    bag_name=${BASE_NAME}_$(printf "%03d" $idx).bag
    echo "准备录制轨迹 index = $idx  ->  $bag_name"
    read -p "按回车开始录制（q 退出）: " cmd
    if [[ "$cmd" == "q" ]]; then
        echo "退出。"
        exit 0
    fi

    echo
    echo "[REC] 开始录制到: $bag_name"
    echo "      使用命令："
    echo "      rosbag record -O $bag_name --bz2 -b 4096 <topics...>"
    echo

    rosbag record -O "$bag_name" --bz2 -b 4096 \
        /robot/arm_left/end_pose \
        /robot/arm_right/end_pose \
        /robot/arm_left/joint_states_single \
        /robot/arm_right/joint_states_single \
        /robot/arm_left/pos_cmd \
        /robot/arm_right/pos_cmd \
        /teleop/arm_left/joint_states_single \
        /teleop/arm_right/joint_states_single \
        /fisheye_left/image_raw/compressed \
        /fisheye_right/image_raw/compressed \
        /fisheye_left/camera_info \
        /fisheye_right/camera_info \
        /realsense_top/color/image_raw/compressed \
        /realsense_top/aligned_depth_to_color/image_raw/compressed \
        /realsense_top/color/camera_info \
        /realsense_top/aligned_depth_to_color/camera_info \
        /wide_top/image_raw/compressed \
        /wide_top/camera_info &
        # Old realsense left/right cameras (commented out)
        # /realsense_left/color/image_raw/compressed \
        # /realsense_right/color/image_raw/compressed \
        # /realsense_left/aligned_depth_to_color/image_raw/compressed \
        # /realsense_right/aligned_depth_to_color/image_raw/compressed \
        # /realsense_left/color/camera_info \
        # /realsense_right/color/camera_info \
        # /realsense_left/aligned_depth_to_color/camera_info \
        # /realsense_right/aligned_depth_to_color/camera_info \

    BAG_PID=$!

    echo "[REC] rosbag PID = $BAG_PID"
    echo "[REC] 现在可以开始遥操作。录完后按回车停止。"
    read -p "" _

    echo "[REC] 正在停止 rosbag (SIGINT)..."
    kill -INT $BAG_PID 2>/dev/null
    wait $BAG_PID 2>/dev/null

    if [[ ! -f "$bag_name" ]]; then
        echo "[WARN] 找不到 $bag_name，可能录制失败，这一条将重录。"
        continue
    fi

    echo
    echo "录制完成: $bag_name"
    echo "选择： n = 下一条   r = 覆盖重录本条   q = 退出"
    read -p "[n / r / q]: " post

    case "$post" in
        r|R)
            echo "[DEL] 删除 $bag_name，准备重录 index = $idx"
            rm -f "$bag_name"
            # 不加 idx，继续重录同一个编号
            ;;
        q|Q)
            echo "退出。"
            exit 0
            ;;
        *)
            # 默认视为 n：保留并进入下一条
            idx=$((idx+1))
            ;;
    esac

    echo
done

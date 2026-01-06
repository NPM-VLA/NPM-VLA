from pathlib import Path
import shutil
import numpy as np
import cv2
import time
from multiprocessing import Pool, Manager, Lock
from functools import partial

from rosbags.highlevel import AnyReader
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from huggingface_hub import hf_hub_download, list_repo_files

REPO_NAME = "zeno/push_block_dual"

# ============================================================
# DATA SOURCE CONFIGURATION
# ============================================================
#   "local":      Load from local directory
#   "huggingface": Load from Hugging Face dataset repository
DATA_SOURCE = "huggingface"  # or "local"

# Local data directory (used when DATA_SOURCE="local")
DATA_ROOT = Path("/home/jovyan/workspace/openpi/push_block_dual_data")


# Mapping from Hugging Face repo to task name
HF_DATASET_REPOS_WITH_TASKS = [
    ("Anlorla/sweep2yellow", "Sweep lego blocks to yellow cross marker."),
    ("Anlorla/sweep2red", "Sweep lego blocks to red cross marker."),
    ("Anlorla/sweep2green", "Sweep lego blocks to green cross marker."),
]

# ROS topics
# Camera topics updated for fisheye cameras and wide top camera
CAM_MAIN = "/realsense_top/color/image_raw/compressed"
CAM_WRIST_LEFT = "/fisheye_left/image_raw/compressed"  # Changed from realsense_left
CAM_WRIST_RIGHT = "/fisheye_right/image_raw/compressed"  # Changed from realsense_right
CAM_WIDE_TOP = "/wide_top/image_raw/compressed"
STATE_LEFT = "/robot/arm_left/joint_states_single"
STATE_RIGHT = "/robot/arm_right/joint_states_single"
ACTION_LEFT = "/teleop/arm_left/joint_states_single"
ACTION_RIGHT = "/teleop/arm_right/joint_states_single"

END_POSE_LEFT = "/robot/arm_left/end_pose"
END_POSE_RIGHT = "/robot/arm_right/end_pose"

# ============================================================
# CONVERSION SETTINGS
# ============================================================
FPS = 10
IMG_SIZE = (224, 224)
# TODO:
NUM_WORKERS = 15  # Number of CPU cores to use


def decode_compressed_image(msg):
    """sensor_msgs/CompressedImage -> HxWx3 uint8 RGB (resized)."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return img_rgb


def nearest_idx(times, t):
    idx = np.searchsorted(times, t)
    if idx == 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    before = times[idx - 1]
    after = times[idx]
    return idx if abs(after - t) < abs(t - before) else idx - 1


def collect_bag_files_local(task_name):
    """Collect all .bag files from local directory."""
    task_dir = DATA_ROOT / task_name
    if not task_dir.exists():
        print(f"Warning: Task directory not found: {task_dir}")
        return []

    bag_files = sorted(task_dir.glob("*.bag"))
    if not bag_files:
        print(f"Warning: No .bag files found in {task_dir}")
    else:
        print(f"Found {len(bag_files)} bag file(s) for task '{task_name}':")
        for bag_file in bag_files:
            print(f"  - {bag_file.name}")

    return bag_files


def collect_bag_files_huggingface():
    """Collect all .bag files from Hugging Face dataset repositories with their task names.

    Returns:
        List of tuples: [(bag_path, task_name), ...]
    """
    all_bag_data = []  # List of (bag_path, task_name) tuples

    for repo, task_name in HF_DATASET_REPOS_WITH_TASKS:
        print(f"\nFetching file list from Hugging Face repo: {repo}")
        print(f"  Task: {task_name}")

        try:
            repo_files = list_repo_files(repo_id=repo, repo_type="dataset")
            bag_filenames = [f for f in repo_files if f.endswith(".bag")]
            bag_filenames = sorted(bag_filenames)

            if not bag_filenames:
                print(f"  Warning: No .bag files found in {repo}")
                continue

            print(f"  Found {len(bag_filenames)} bag file(s) in {repo}:")
            for filename in bag_filenames:
                print(f"    - {filename}")

            print("  Downloading/caching files from Hugging Face...")
            for filename in bag_filenames:
                print(f"    Fetching: {filename}")
                local_path = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    repo_type="dataset",
                )
                local_path = Path(local_path)
                all_bag_data.append((local_path, task_name))
                file_size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"      -> Cached at: {local_path} ({file_size_mb:.2f} MB)")

        except Exception as e:
            print(f"Error fetching files from Hugging Face repo {repo}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\nTotal .bag files collected from all repos: {len(all_bag_data)}")
    return all_bag_data


def collect_bag_files(task_name=None):
    """Dispatch between local / Hugging Face data source.

    For huggingface source: Returns list of (bag_path, task_name) tuples
    For local source: Returns list of bag_paths (task_name from parameter)
    """
    if DATA_SOURCE == "huggingface":
        return collect_bag_files_huggingface()
    elif DATA_SOURCE == "local":
        if task_name is None:
            raise ValueError("task_name is required for local data source")
        # Return as tuples for consistency
        bag_paths = collect_bag_files_local(task_name)
        return [(path, task_name) for path in bag_paths]
    else:
        raise ValueError(
            f"Unknown DATA_SOURCE: {DATA_SOURCE}. Must be 'local' or 'huggingface'"
        )


def process_single_bag(args):
    """
    Process a single bag file and return episode data.
    This function runs in a separate process.
    """
    bag_path, task_name, bag_idx, total_bags, progress_dict, lock = args

    with lock:
        print(
            f"\n[Bag {bag_idx}/{total_bags}] Processing: {bag_path.name} (Worker PID: {os.getpid()})"
        )

    bag_start_time = time.time()

    try:
        with AnyReader([bag_path]) as reader:
            interested_topics = {
                CAM_MAIN,
                CAM_WRIST_LEFT,
                CAM_WRIST_RIGHT,
                CAM_WIDE_TOP,
                STATE_LEFT,
                STATE_RIGHT,
                ACTION_LEFT,
                ACTION_RIGHT,
                END_POSE_LEFT,
                END_POSE_RIGHT,
            }
            topic_to_msgs = {topic: [] for topic in interested_topics}

            connections = [
                c for c in reader.connections if c.topic in interested_topics
            ]

            if not connections:
                with lock:
                    print(
                        f"[Bag {bag_idx}/{total_bags}] Warning: No relevant topics found, skipping."
                    )
                return None

            for conn, t, raw in reader.messages(connections=connections):
                if conn.topic not in topic_to_msgs:
                    continue
                msg = reader.deserialize(raw, conn.msgtype)
                topic_to_msgs[conn.topic].append((t, msg))

            cam_main_msgs = topic_to_msgs[CAM_MAIN]
            cam_wrist_left_msgs = topic_to_msgs[CAM_WRIST_LEFT]
            cam_wrist_right_msgs = topic_to_msgs[CAM_WRIST_RIGHT]
            cam_wide_top_msgs = topic_to_msgs[CAM_WIDE_TOP]
            state_left_msgs = topic_to_msgs[STATE_LEFT]
            state_right_msgs = topic_to_msgs[STATE_RIGHT]
            action_left_msgs = topic_to_msgs[ACTION_LEFT]
            action_right_msgs = topic_to_msgs[ACTION_RIGHT]

            # New topics
            end_pose_left_msgs = topic_to_msgs[END_POSE_LEFT]
            end_pose_right_msgs = topic_to_msgs[END_POSE_RIGHT]

            if not cam_main_msgs or not cam_wrist_left_msgs or not cam_wrist_right_msgs or not cam_wide_top_msgs:
                with lock:
                    print(
                        f"[Bag {bag_idx}/{total_bags}] Warning: Missing camera data, skipping"
                    )
                return None
            if (
                not state_left_msgs
                or not state_right_msgs
                or not action_left_msgs
                or not action_right_msgs
            ):
                with lock:
                    print(
                        f"[Bag {bag_idx}/{total_bags}] Warning: Missing state/action data, skipping"
                    )
                return None

            state_left_times = np.array([t for t, _ in state_left_msgs], dtype=np.int64)
            state_right_times = np.array(
                [t for t, _ in state_right_msgs], dtype=np.int64
            )
            action_left_times = np.array(
                [t for t, _ in action_left_msgs], dtype=np.int64
            )
            action_right_times = np.array(
                [t for t, _ in action_right_msgs], dtype=np.int64
            )
            cam_main_times = np.array([t for t, _ in cam_main_msgs], dtype=np.int64)
            wrist_left_times = np.array(
                [t for t, _ in cam_wrist_left_msgs], dtype=np.int64
            )
            wrist_right_times = np.array(
                [t for t, _ in cam_wrist_right_msgs], dtype=np.int64
            )
            wide_top_times = np.array(
                [t for t, _ in cam_wide_top_msgs], dtype=np.int64
            )

            end_pose_left_times = (
                np.array([t for t, _ in end_pose_left_msgs], dtype=np.int64)
                if end_pose_left_msgs
                else np.array([], dtype=np.int64)
            )
            end_pose_right_times = (
                np.array([t for t, _ in end_pose_right_msgs], dtype=np.int64)
                if end_pose_right_msgs
                else np.array([], dtype=np.int64)
            )

            with lock:
                print(f"\n[Bag {bag_idx}/{total_bags}] Camera time ranges:")
                main_duration = (cam_main_times[-1] - cam_main_times[0]) / 1e9
                left_duration = (wrist_left_times[-1] - wrist_left_times[0]) / 1e9
                right_duration = (wrist_right_times[-1] - wrist_right_times[0]) / 1e9
                wide_duration = (wide_top_times[-1] - wide_top_times[0]) / 1e9
                print(
                    f"    Main camera:   {main_duration:.2f} seconds ({len(cam_main_msgs)} frames)"
                )
                print(
                    f"    Left wrist:    {left_duration:.2f} seconds ({len(cam_wrist_left_msgs)} frames)"
                )
                print(
                    f"    Right wrist:   {right_duration:.2f} seconds ({len(cam_wrist_right_msgs)} frames)"
                )
                print(
                    f"    Wide top:      {wide_duration:.2f} seconds ({len(cam_wide_top_msgs)} frames)"
                )

            t_start = max(
                cam_main_times[0],
                wrist_left_times[0],
                wrist_right_times[0],
                wide_top_times[0],
            )
            t_end = min(
                cam_main_times[-1],
                wrist_left_times[-1],
                wrist_right_times[-1],
                wide_top_times[-1],
            )

            if t_end <= t_start:
                with lock:
                    print(
                        f"[Bag {bag_idx}/{total_bags}] Warning: Non-positive common duration, skipping"
                    )
                return None

            common_duration = (t_end - t_start) / 1e9
            with lock:
                print(
                    f"\n[Bag {bag_idx}/{total_bags}] Common time range: {common_duration:.2f} seconds"
                )

            min_dt = int(1e9 / FPS)
            num_frames = int((t_end - t_start) / min_dt)
            if num_frames <= 0:
                with lock:
                    print(
                        f"[Bag {bag_idx}/{total_bags}] Warning: num_frames <= 0, skipping"
                    )
                return None

            uniform_timestamps = np.linspace(t_start, t_end, num_frames, dtype=np.int64)

            with lock:
                print(
                    f"[Bag {bag_idx}/{total_bags}] Generating {num_frames} frames at {FPS} FPS"
                )

            # Collect all frames for this episode
            episode_frames = []
            frame_count = 0
            start_time = time.time()
            last_print_time = start_time

            for t_frame in uniform_timestamps:
                frame_count += 1

                current_time = time.time()
                if current_time - last_print_time >= 5.0:
                    elapsed = current_time - start_time
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (
                        (num_frames - frame_count) / fps_processing
                        if fps_processing > 0
                        else 0.0
                    )
                    progress_pct = 100.0 * frame_count / num_frames
                    with lock:
                        print(
                            f"    [Bag {bag_idx}/{total_bags}] Progress: {frame_count}/{num_frames} ({progress_pct:.1f}%) | "
                            f"Speed: {fps_processing:.1f} fps | ETA: {eta_seconds:.0f}s"
                        )
                    last_print_time = current_time

                idx_main = nearest_idx(cam_main_times, t_frame)
                idx_wl = nearest_idx(wrist_left_times, t_frame)
                idx_wr = nearest_idx(wrist_right_times, t_frame)
                idx_wt = nearest_idx(wide_top_times, t_frame)

                main_image = decode_compressed_image(cam_main_msgs[idx_main][1])
                wrist_left_image = decode_compressed_image(
                    cam_wrist_left_msgs[idx_wl][1]
                )
                wrist_right_image = decode_compressed_image(
                    cam_wrist_right_msgs[idx_wr][1]
                )
                wide_top_image = decode_compressed_image(
                    cam_wide_top_msgs[idx_wt][1]
                )

                idx_sl = nearest_idx(state_left_times, t_frame)
                idx_sr = nearest_idx(state_right_times, t_frame)
                idx_al = nearest_idx(action_left_times, t_frame)
                idx_ar = nearest_idx(action_right_times, t_frame)

                state_left_msg = state_left_msgs[idx_sl][1]
                state_right_msg = state_right_msgs[idx_sr][1]
                action_left_msg = action_left_msgs[idx_al][1]
                action_right_msg = action_right_msgs[idx_ar][1]

                q_robot_left = np.array(state_left_msg.position, dtype=np.float32)
                q_robot_right = np.array(state_right_msg.position, dtype=np.float32)

                state_vec = np.zeros(14, dtype=np.float32)
                n_sl = min(7, q_robot_left.shape[0])
                n_sr = min(7, q_robot_right.shape[0])

                state_vec[:n_sl] = q_robot_left[:n_sl]
                state_vec[7 : 7 + n_sr] = q_robot_right[:n_sr]

                q_tele_left = np.array(action_left_msg.position, dtype=np.float32)
                q_tele_right = np.array(action_right_msg.position, dtype=np.float32)
                action_vec = np.zeros(14, dtype=np.float32)
                n_al = min(7, q_tele_left.shape[0])
                n_ar = min(7, q_tele_right.shape[0])
                action_vec[:n_al] = q_tele_left[:n_al]
                action_vec[7 : 7 + n_ar] = q_tele_right[:n_ar]

                # Process end effector poses (if available)
                end_pose_vec = np.zeros(14, dtype=np.float32)  # 7 for left + 7 for right
                if len(end_pose_left_times) > 0:
                    idx_epl = nearest_idx(end_pose_left_times, t_frame)
                    end_pose_left_msg = end_pose_left_msgs[idx_epl][1]
                    # PoseStamped message: pose.position.{x,y,z} and pose.orientation.{x,y,z,w}
                    if hasattr(end_pose_left_msg, "pose"):
                        pose = end_pose_left_msg.pose
                        end_pose_vec[0:3] = [
                            pose.position.x,
                            pose.position.y,
                            pose.position.z,
                        ]
                        end_pose_vec[3:7] = [
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ]

                if len(end_pose_right_times) > 0:
                    idx_epr = nearest_idx(end_pose_right_times, t_frame)
                    end_pose_right_msg = end_pose_right_msgs[idx_epr][1]
                    if hasattr(end_pose_right_msg, "pose"):
                        pose = end_pose_right_msg.pose
                        end_pose_vec[7:10] = [
                            pose.position.x,
                            pose.position.y,
                            pose.position.z,
                        ]
                        end_pose_vec[10:14] = [
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ]

                frame = {
                    "observation.images.main": main_image,
                    "observation.images.secondary_0": wrist_left_image,
                    "observation.images.secondary_1": wrist_right_image,
                    "observation.images.secondary_2": wide_top_image,
                    "observation.state": state_vec,
                    "observation.ee_pose": end_pose_vec,
                    "action": action_vec,
                    "task": task_name,
                }

                episode_frames.append(frame)

            elapsed = time.time() - start_time
            bag_elapsed = time.time() - bag_start_time

            with lock:
                print(
                    f"    [Bag {bag_idx}/{total_bags}] Progress: {frame_count}/{num_frames} (100.0%) | "
                    f"Speed: {frame_count / elapsed:.1f} fps | Done!"
                )
                print(
                    f"  [Bag {bag_idx}/{total_bags}] ✓ Completed in {bag_elapsed:.1f}s "
                    f"({frame_count} frames, {frame_count / bag_elapsed:.1f} fps)"
                )
                progress_dict["completed"] += 1
                print(
                    f"  Progress: {progress_dict['completed']}/{total_bags} bags completed"
                )

            return episode_frames

    except Exception as e:
        with lock:
            print(f"[Bag {bag_idx}/{total_bags}] Error processing bag: {e}")
            import traceback

            traceback.print_exc()
        return None


# ============================================================
# MAIN CONVERSION LOOP WITH MULTIPROCESSING
# ============================================================

if __name__ == "__main__":
    import os

    total_start_time = time.time()

    # ============================================================
    # CREATE LEROBOT DATASET
    # ============================================================

    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    features = {
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"left_joint_{i}" for i in range(7)]
            + [f"right_joint_{i}" for i in range(7)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),  # Each arm: 6 DOF + 1 gripper = 7, total 14
            "names": [f"left_joint_{i}" for i in range(7)]
            + [f"right_joint_{i}" for i in range(7)],
        },
        "observation.ee_pose": {
            "dtype": "float32",
            "shape": (14,),  # 7 per arm: [x, y, z, qx, qy, qz, qw]
            "names": [
                "left_pos_x",
                "left_pos_y",
                "left_pos_z",
                "left_quat_x",
                "left_quat_y",
                "left_quat_z",
                "left_quat_w",
                "right_pos_x",
                "right_pos_y",
                "right_pos_z",
                "right_quat_x",
                "right_quat_y",
                "right_quat_z",
                "right_quat_w",
            ],
        },
        "observation.images.main": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),  # (H, W, C)
            "names": ["height", "width", "channels"],
        },
        "observation.images.secondary_0": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.secondary_1": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.secondary_2": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="zeno",
        fps=FPS,
        features=features,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=4,
    )

    # Collect all bag files with their corresponding task names
    print(f"\n{'=' * 60}")
    print("Collecting bag files from all repositories...")
    print(f"{'=' * 60}")

    bag_data = collect_bag_files()  # Returns [(bag_path, task_name), ...]
    total_bags = len(bag_data)

    if total_bags == 0:
        print("No bag files to process.")
    else:
        # Print summary by task
        task_counts = {}
        for bag_path, task_name in bag_data:
            task_counts[task_name] = task_counts.get(task_name, 0) + 1

        print(f"\nTotal bags to process: {total_bags}")
        print("\nBreakdown by task:")
        for task_name, count in task_counts.items():
            print(f"  - {task_name}: {count} bags")

        # Create shared objects for multiprocessing
        manager = Manager()
        progress_dict = manager.dict()
        progress_dict["completed"] = 0
        lock = manager.Lock()

        # Prepare arguments for each bag file with its corresponding task name
        bag_args = [
            (bag_path, task_name, bag_idx, total_bags, progress_dict, lock)
            for bag_idx, (bag_path, task_name) in enumerate(bag_data, 1)
        ]

        print(f"\nStarting parallel processing with {NUM_WORKERS} workers...")

        # Process bags in parallel
        with Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(process_single_bag, bag_args)

        # Add all successfully processed episodes to dataset
        print(f"\n{'=' * 60}")
        print("Adding processed episodes to dataset...")
        print(f"{'=' * 60}")

        successful_episodes = 0
        for result in results:
            if result is not None:
                for frame in result:
                    dataset.add_frame(frame)
                dataset.save_episode()
                successful_episodes += 1

        print(f"\nSuccessfully processed {successful_episodes}/{total_bags} bags")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    total_time = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print("CONVERSION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
    print(f"Dataset saved to: {output_path}")
    print(f"Number of workers used: {NUM_WORKERS}")
    print(f"{'=' * 60}")

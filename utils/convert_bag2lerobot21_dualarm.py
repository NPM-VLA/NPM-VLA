from pathlib import Path
import shutil
import numpy as np
import cv2
import time
from multiprocessing import Pool, Manager, Lock
from functools import partial

from rosbags.highlevel import AnyReader
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from huggingface_hub import hf_hub_download, list_repo_files

REPO_NAME = "zeno/sweep2E_v1"

# ============================================================
# DATA SOURCE CONFIGURATION
# ============================================================
#   "local":      Load from local directory
#   "huggingface": Load from Hugging Face dataset repository
DATA_SOURCE = "huggingface"  # or "local"

# Local data directory (used when DATA_SOURCE="local")
DATA_ROOT = Path("/home/jovyan/workspace/openpi/npm_sweep2E_data")

# Hugging Face dataset repository (used when DATA_SOURCE="huggingface")
HF_DATASET_REPO = "Anlorla/sweep2E"

# List of task names to process (each task folder contains multiple .bag files)
TASK_NAMES = [
    "sweep lego blocks into 'E' character shape",
]

# ROS topics
CAM_MAIN = "/realsense_top/color/image_raw/compressed"
CAM_WRIST_LEFT = "/realsense_left/color/image_raw/compressed"
CAM_WRIST_RIGHT = "/realsense_right/color/image_raw/compressed"
STATE_LEFT = "/robot/arm_left/joint_states_single"
STATE_RIGHT = "/robot/arm_right/joint_states_single"
ACTION_LEFT = "/teleop/arm_left/joint_states_single"
ACTION_RIGHT = "/teleop/arm_right/joint_states_single"

# ============================================================
# CONVERSION SETTINGS
# ============================================================
FPS = 10
IMG_SIZE = (256, 256)
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


def collect_bag_files_huggingface(task_name):
    """Collect all .bag files from Hugging Face dataset repository."""
    print(f"Fetching file list from Hugging Face: {HF_DATASET_REPO}")

    try:
        all_files = list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")

        # Filter for .bag files for this task
        bag_filenames = [f for f in all_files if f.endswith(".bag")]
        bag_filenames = sorted(bag_filenames)

        if not bag_filenames:
            print(
                f"Warning: No .bag files found for task '{task_name}' in {HF_DATASET_REPO}"
            )
            return []

        print(f"Found {len(bag_filenames)} bag file(s) for task '{task_name}':")
        for filename in bag_filenames:
            print(f"  - {filename}")

        print("\nDownloading/caching files from Hugging Face...")
        bag_paths = []
        for filename in bag_filenames:
            print(f"  Fetching: {filename}")
            local_path = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=filename,
                repo_type="dataset",
            )
            bag_paths.append(Path(local_path))
            file_size_mb = Path(local_path).stat().st_size / (1024 * 1024)
            print(f"    -> Cached at: {local_path} ({file_size_mb:.2f} MB)")

        return bag_paths

    except Exception as e:
        print(f"Error fetching files from Hugging Face: {e}")
        import traceback

        traceback.print_exc()
        return []


def collect_bag_files(task_name):
    """Dispatch between local / Hugging Face data source."""
    if DATA_SOURCE == "huggingface":
        return collect_bag_files_huggingface(task_name)
    elif DATA_SOURCE == "local":
        return collect_bag_files_local(task_name)
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
                STATE_LEFT,
                STATE_RIGHT,
                ACTION_LEFT,
                ACTION_RIGHT,
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
            state_left_msgs = topic_to_msgs[STATE_LEFT]
            state_right_msgs = topic_to_msgs[STATE_RIGHT]
            action_left_msgs = topic_to_msgs[ACTION_LEFT]
            action_right_msgs = topic_to_msgs[ACTION_RIGHT]

            if not cam_main_msgs or not cam_wrist_left_msgs or not cam_wrist_right_msgs:
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

            with lock:
                print(f"\n[Bag {bag_idx}/{total_bags}] Camera time ranges:")
                main_duration = (cam_main_times[-1] - cam_main_times[0]) / 1e9
                left_duration = (wrist_left_times[-1] - wrist_left_times[0]) / 1e9
                right_duration = (wrist_right_times[-1] - wrist_right_times[0]) / 1e9
                print(
                    f"    Main camera:   {main_duration:.2f} seconds ({len(cam_main_msgs)} frames)"
                )
                print(
                    f"    Left wrist:    {left_duration:.2f} seconds ({len(cam_wrist_left_msgs)} frames)"
                )
                print(
                    f"    Right wrist:   {right_duration:.2f} seconds ({len(cam_wrist_right_msgs)} frames)"
                )

            t_start = max(
                cam_main_times[0],
                wrist_left_times[0],
                wrist_right_times[0],
            )
            t_end = min(
                cam_main_times[-1],
                wrist_left_times[-1],
                wrist_right_times[-1],
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

                main_image = decode_compressed_image(cam_main_msgs[idx_main][1])
                wrist_left_image = decode_compressed_image(
                    cam_wrist_left_msgs[idx_wl][1]
                )
                wrist_right_image = decode_compressed_image(
                    cam_wrist_right_msgs[idx_wr][1]
                )

                idx_sl = nearest_idx(state_left_times, t_frame)
                idx_sr = nearest_idx(state_right_times, t_frame)
                idx_al = nearest_idx(action_left_times, t_frame)
                idx_ar = nearest_idx(action_right_times, t_frame)

                state_left_msg = state_left_msgs[idx_sl][1]
                state_right_msg = state_right_msgs[idx_sr][1]
                action_left_msg = action_left_msgs[idx_al][1]
                action_right_msg = action_right_msgs[idx_ar][1]

                # 16 维 state: 8 左 + 8 右
                q_robot_left = np.array(state_left_msg.position, dtype=np.float32)
                q_robot_right = np.array(state_right_msg.position, dtype=np.float32)
                state_vec = np.zeros(16, dtype=np.float32)
                n_sl = min(8, q_robot_left.shape[0])
                n_sr = min(8, q_robot_right.shape[0])
                state_vec[:n_sl] = q_robot_left[:n_sl]
                state_vec[8 : 8 + n_sr] = q_robot_right[:n_sr]

                q_tele_left = np.array(action_left_msg.position, dtype=np.float32)
                q_tele_right = np.array(action_right_msg.position, dtype=np.float32)
                action_vec = np.zeros(14, dtype=np.float32)
                n_al = min(7, q_tele_left.shape[0])
                n_ar = min(7, q_tele_right.shape[0])
                action_vec[:n_al] = q_tele_left[:n_al]
                action_vec[7 : 7 + n_ar] = q_tele_right[:n_ar]

                frame = {
                    "observation.images.main": main_image,
                    "observation.images.secondary_0": wrist_left_image,
                    "observation.images.secondary_1": wrist_right_image,
                    "observation.state": state_vec,
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
            "shape": (16,),
            "names": [f"left_joint_{i}" for i in range(8)]
            + [f"right_joint_{i}" for i in range(8)],
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

    for task_name in TASK_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Processing task: {task_name}")
        print(f"{'=' * 60}")

        bag_files = collect_bag_files(task_name)
        total_bags = len(bag_files)

        if total_bags == 0:
            print("No bag files to process.")
            continue

        # Create shared objects for multiprocessing
        manager = Manager()
        progress_dict = manager.dict()
        progress_dict["completed"] = 0
        lock = manager.Lock()

        # Prepare arguments for each bag file
        bag_args = [
            (bag_path, task_name, bag_idx, total_bags, progress_dict, lock)
            for bag_idx, bag_path in enumerate(bag_files, 1)
        ]

        print(f"\nStarting parallel processing with {NUM_WORKERS} workers...")
        print(f"Total bags to process: {total_bags}")

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

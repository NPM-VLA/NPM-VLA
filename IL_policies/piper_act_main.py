#!/usr/bin/env python3
import json
from pathlib import Path

import draccus
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
import cv2
import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


# ----------------- Global buffers -----------------

def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    """
    Decode a ROS CompressedImage message to a BGR uint8 OpenCV image.
    Equivalent to CvBridge.compressed_imgmsg_to_cv2(..., 'bgr8').
    """
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_bgr


latest_imgs = {
    "main": None,
    "wrist_l": None,
    "wrist_r": None,
}
latest_q = {
    "left": None,
    "right": None,
}

smoothed_action = {
    "left": None,
    "right": None,
}


# ----------------- Image callbacks -----------------
def cb_main(msg: CompressedImage):
    latest_imgs["main"] = decode_compressed_image(msg)
    rospy.logdebug("Received main camera image")


def cb_wrist_l(msg: CompressedImage):
    latest_imgs["wrist_l"] = decode_compressed_image(msg)
    rospy.logdebug("Received left wrist camera image")


def cb_wrist_r(msg: CompressedImage):
    latest_imgs["wrist_r"] = decode_compressed_image(msg)
    rospy.logdebug("Received right wrist camera image")


# ----------------- Joint callbacks -----------------
def cb_joints_left(msg: JointState):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)
    rospy.logdebug(f"Received left arm joint states: {latest_q['left'][:3]}...")


def cb_joints_right(msg: JointState):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)
    rospy.logdebug(f"Received right arm joint states: {latest_q['right'][:3]}...")


# ----------------- Utility: image -> torch tensor -----------------
def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert BGR uint8 (H,W,3) to float32 torch tensor (1,3,256,256) in [0,1].
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0  # [H,W,3]
    img = np.transpose(img, (2, 0, 1))    # [3,H,W]
    img = np.expand_dims(img, 0)          # [1,3,H,W]
    return torch.from_numpy(img)


# ----------------- Load ACT policy -----------------
def load_policy(ckpt_dir: str, device: str) -> ACTPolicy:
    """
    Load an ACTPolicy either from a local checkpoint directory
    (e.g. .../checkpoints/040000/pretrained_model) or from a HF repo id.

    - If `ckpt_dir` is an existing directory -> load from local files.
    - Otherwise -> treat `ckpt_dir` as HF repo_id and call from_pretrained.
    """
    ckpt_path = Path(ckpt_dir).expanduser().resolve()
    rospy.loginfo(f"Loading ACT Policy from: {ckpt_path}")

    # ===== Case 1: local directory checkpoint =====
    if ckpt_path.is_dir():
        # 1) Load config.json
        config_path = ckpt_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        # 2) Load the JSON and use draccus.decode to properly deserialize nested dataclasses
        #    This ensures nested dataclasses like PolicyFeature are properly reconstructed
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)

        # Remove the 'type' field which is used for registry but not a constructor argument
        cfg_dict.pop("type", None)

        # Use draccus.decode to convert dict to proper dataclass with nested structures
        from draccus.parsers import decoding

        config = decoding.decode(ACTConfig, cfg_dict)

        # 3) Create empty policy and move to device
        policy = ACTPolicy(config=config).to(device)

        # 4) Find a weight file inside the directory
        weight_path = None
        candidate_patterns = [
            "model*.safetensors",
            "pytorch_model*.bin",
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
        ]
        for pattern in candidate_patterns:
            matches = list(ckpt_path.glob(pattern))
            if matches:
                # Prefer names that start with "model" or "pytorch_model"
                matches.sort(
                    key=lambda p: (
                        not (
                            p.name.startswith("model")
                            or p.name.startswith("pytorch_model")
                        ),
                        p.name,
                    )
                )
                weight_path = matches[0]
                break

        if weight_path is None:
            raise FileNotFoundError(
                f"Could not find any weight file (.safetensors/.bin/.pt/.pth) in {ckpt_path}"
            )

        rospy.loginfo(f"  → Loading weights from {weight_path.name}")

        # 5) Load state_dict
        if weight_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as safe_load_file
            except ImportError:
                raise RuntimeError(
                    f"Found safetensors weights at {weight_path} but safetensors is not installed."
                )
            state = safe_load_file(str(weight_path), device=device)
        else:
            state = torch.load(weight_path, map_location=device)

        # Some checkpoints wrap the real state_dict in a top-level key
        if isinstance(state, dict):
            for key in ["state_dict", "model", "model_state_dict"]:
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        missing, unexpected = policy.load_state_dict(state, strict=False)
        if missing:
            rospy.logwarn(
                f"  ⚠️ Missing {len(missing)} keys when loading state dict (strict=False)."
            )
        if unexpected:
            rospy.logwarn(
                f"  ⚠️ Unexpected {len(unexpected)} keys when loading state dict (strict=False)."
            )

        policy.eval()
        return policy

    # ===== Case 2: not a local dir → treat as HF repo id =====
    rospy.loginfo("  → Path does not exist locally, assuming Hugging Face repo id.")
    policy = ACTPolicy.from_pretrained(
        pretrained_name_or_path=ckpt_dir,
        device=device,
    )
    policy.eval()
    return policy


def main():
    rospy.init_node("piper_act_main")

    # ----------------- Load policy -----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # You can either:
    # 1) load from local checkpoint dir,
    # 2) or from HF repo id (e.g. "Anlorla/sweep2E_act").
    #
    # Update this path to your actual checkpoint:
    ckpt_dir = "/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/push_block_act"

    policy = load_policy(ckpt_dir, device)
    policy.reset()  # Reset the action queue for ACT policy

    # ----------------- ROS subs / pubs -----------------
    rospy.Subscriber(
        "/realsense_top/color/image_raw/compressed",
        CompressedImage,
        cb_main,
        queue_size=1,
    )
    rospy.Subscriber(
        "/realsense_left/color/image_raw/compressed",
        CompressedImage,
        cb_wrist_l,
        queue_size=1,
    )
    rospy.Subscriber(
        "/realsense_right/color/image_raw/compressed",
        CompressedImage,
        cb_wrist_r,
        queue_size=1,
    )

    rospy.Subscriber(
        "/robot/arm_left/joint_states_single",
        JointState,
        cb_joints_left,
        queue_size=1,
    )
    rospy.Subscriber(
        "/robot/arm_right/joint_states_single",
        JointState,
        cb_joints_right,
        queue_size=1,
    )

    pub_left = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher(
        "/robot/arm_left/vla_joint_cmd", JointState, queue_size=1
    )

    rospy.loginfo("Robot arm command publishers initialized")

    # ----------------- Control parameters -----------------
    # SAFETY: Start with moderate frequency (10 Hz) for testing
    # For initial testing on new robot, consider starting with 0.5 Hz (rate = rospy.Rate(0.5))
    rate = rospy.Rate(20)  # Control frequency in Hz

    # Safety parameters
    MAX_JOINT_DELTA = 0.15  # Maximum joint position change per step (radians)
    ENABLE_ACTION_CLIPPING = True  # Clip large action deltas for safety

    # EMA smoothing
    ENABLE_SMOOTHING = True
    SMOOTHING_ALPHA = 0.3

    rospy.loginfo("Waiting for first images and joint states...")

    data_ready_logged = False

    # Number of actions to execute from each predicted action chunk
    num_actions_to_execute = 8

    while not rospy.is_shutdown():
        if any(v is None for v in latest_imgs.values()) or any(
            v is None for v in latest_q.values()
        ):
            rate.sleep()
            continue

        if not data_ready_logged:
            rospy.loginfo(
                "✓ Successfully receiving observations from robot arms (cameras + joint states)"
            )
            data_ready_logged = True

        # --------------- Build observation dict for policy ---------------
        main_img = preprocess_image(latest_imgs["main"]).to(device)   # [1,3,256,256]
        wrist_l = preprocess_image(latest_imgs["wrist_l"]).to(device)
        wrist_r = preprocess_image(latest_imgs["wrist_r"]).to(device)

        state_np = np.concatenate(
            [latest_q["left"], latest_q["right"]], axis=0
        ).astype(np.float32)
        state = torch.from_numpy(state_np[None, :]).to(device)  # [1,14]

        obs = {
            # These keys must match the training config input_features
            "observation.images.main": main_img,
            "observation.images.secondary_0": wrist_l,
            "observation.images.secondary_1": wrist_r,
            "observation.state": state,
        }

        # --------------- Policy inference (get multiple actions from queue) ---------------
        # ACT manages an internal action queue, so we call select_action multiple times
        # to get a chunk of actions to execute
        rospy.logdebug("Requesting action chunk from ACT policy...")
        actions = []

        for _ in range(num_actions_to_execute):
            with torch.no_grad():
                action_tensor = policy.select_action(obs)  # [1, 14] - single action

            # Extract action from batch dimension
            action = action_tensor[0, :].cpu().numpy()  # [14]

            if len(action) != 14:
                rospy.logwarn(
                    f"[SAFETY] Invalid action dimension: expected 14, got {len(action)}. Stopping chunk collection."
                )
                break

            actions.append(action)

        if len(actions) == 0:
            rospy.logwarn("[SAFETY] No valid actions received, skipping this cycle.")
            rate.sleep()
            continue

        actions = np.array(actions)  # [num_actions, 14]
        rospy.loginfo_throttle(5.0, f"✓ Successfully received action chunk, shape: {actions.shape}")

        num_to_exec = len(actions)
        rospy.loginfo(f"Executing {num_to_exec} actions from ACT policy queue")

        # Execute the actions from the chunk
        for i in range(num_to_exec):
            if rospy.is_shutdown():
                break

            action = actions[i]

            # Split into left/right (14-dim total: 7 joints per arm)
            action_left = action[:7].copy()
            action_right = action[7:14].copy()
            # Swap left/right to match hardware configuration
            action_left, action_right = action_right, action_left

            # --------------- EMA smoothing ---------------
            global smoothed_action
            if ENABLE_SMOOTHING:
                if smoothed_action["left"] is None:
                    smoothed_action["left"] = action_left
                    smoothed_action["right"] = action_right
                else:
                    smoothed_action["left"] = (
                        SMOOTHING_ALPHA * action_left
                        + (1.0 - SMOOTHING_ALPHA) * smoothed_action["left"]
                    )
                    smoothed_action["right"] = (
                        SMOOTHING_ALPHA * action_right
                        + (1.0 - SMOOTHING_ALPHA) * smoothed_action["right"]
                    )
                action_left = smoothed_action["left"]
                action_right = smoothed_action["right"]

            # --------------- Safety clipping ---------------
            if ENABLE_ACTION_CLIPPING:
                cur_left = latest_q["left"][:7]
                cur_right = latest_q["right"][:7]

                delta_left = action_left - cur_left
                delta_right = action_right - cur_right

                max_dl = np.abs(delta_left).max()
                max_dr = np.abs(delta_right).max()

                if max_dl > MAX_JOINT_DELTA:
                    rospy.logwarn(
                        f"[SAFETY] Left delta {max_dl:.3f} > {MAX_JOINT_DELTA}, clipping."
                    )
                    delta_left = np.clip(delta_left, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
                    action_left = cur_left + delta_left

                if max_dr > MAX_JOINT_DELTA:
                    rospy.logwarn(
                        f"[SAFETY] Right delta {max_dr:.3f} > {MAX_JOINT_DELTA}, clipping."
                    )
                    delta_right = np.clip(delta_right, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
                    action_right = cur_right + delta_right

            # --------------- Publish to robot ---------------
            msg_left = JointState()
            msg_left.header.stamp = rospy.Time.now()
            msg_left.position = action_left.tolist()

            msg_right = JointState()
            msg_right.header.stamp = rospy.Time.now()
            msg_right.position = action_right.tolist()

            pub_left.publish(msg_left)
            pub_right.publish(msg_right)

            rospy.loginfo(f"✓ Sent action {i+1}/{num_to_exec} to robot arms")
            rospy.logdebug(f"  Left arm:  [{', '.join([f'{x:.3f}' for x in action_left])}]")
            rospy.logdebug(
                f"  Right arm: [{', '.join([f'{x:.3f}' for x in action_right])}]"
            )

            rate.sleep()


if __name__ == "__main__":
    main()

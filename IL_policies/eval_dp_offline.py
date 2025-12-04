#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline eval for Diffusion Policy on sweep2E dataset"
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default="Anlorla/sweep2E_lerobot30",
        help="HF dataset repo id for evaluation",
    )
    parser.add_argument(
        "--policy_ckpt",
        type=str,
        default="~/workspace/IL_policies/checkpoints/sweep2E_dp/checkpoints/040000/pretrained_model",
        help="Path to DP policy checkpoint directory (the `policy/` folder)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=50,
        help="Maximum number of batches to evaluate (to avoid processing the entire dataset)",
    )
    return parser.parse_args()


def build_dataloader(dataset_repo: str, batch_size: int) -> DataLoader:
    print(f"📚 Loading dataset: {dataset_repo}")
    ds = LeRobotDataset(dataset_repo)

    print(f"  Number of frames: {len(ds)}")
    print(f"  Example keys from first sample: {list(ds[0].keys())}")

    # Default collate_fn is sufficient (dict of numpy/tensors will be automatically stacked)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dl


def load_policy(ckpt_dir: str, device: str):
    """
    Load a DiffusionPolicy either from a local checkpoint directory
    (e.g. .../checkpoints/040000/pretrained_model) or from a HF repo id.

    - If `ckpt_dir` is an existing directory -> load from local files.
    - Otherwise -> treat `ckpt_dir` as HF repo_id and call from_pretrained.
    """
    ckpt_path = Path(ckpt_dir).expanduser().resolve()
    print(f"🧠 Loading Diffusion Policy from: {ckpt_path}")

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

        config = decoding.decode(DiffusionConfig, cfg_dict)

        # 3) Create empty policy and move to device
        policy = DiffusionPolicy(config=config).to(device)

        # 4) Find a weight file inside the directory
        weight_path = None
        candidate_patterns = [
            "diffusion_policy*.safetensors",  # very likely for DP
            "pytorch_model*.bin",
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
        ]
        for pattern in candidate_patterns:
            matches = list(ckpt_path.glob(pattern))
            if matches:
                # Prefer names that start with "diffusion_policy" or "pytorch_model"
                matches.sort(
                    key=lambda p: (
                        not (
                            p.name.startswith("diffusion_policy")
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

        print(f"  → Loading weights from {weight_path.name}")

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
            print(
                f"  ⚠️ Missing {len(missing)} keys when loading state dict (strict=False)."
            )
        if unexpected:
            print(
                f"  ⚠️ Unexpected {len(unexpected)} keys when loading state dict (strict=False)."
            )

        policy.eval()
        # print(policy)
        return policy

    # ===== Case 2: not a local dir → treat as HF repo id =====
    print("  → Path does not exist locally, assuming Hugging Face repo id.")
    policy = DiffusionPolicy.from_pretrained(
        pretrained_model_name_or_path=ckpt_dir,
        device=device,
    )
    policy.eval()
    return policy


@torch.no_grad()
def run_eval(dl: DataLoader, policy, max_batches: int, device: str):
    all_sq_err = []  # list of [B, A] arrays
    all_abs_err = []  # list of [B, A] arrays
    action_dim = None

    batches_seen = 0

    for batch in dl:
        batches_seen += 1
        if batches_seen > max_batches:
            break

        # Build observation dict (move tensors to device)
        obs = {
            "observation.images.main": batch["observation.images.main"].to(device),
            "observation.images.secondary_0": batch[
                "observation.images.secondary_0"
            ].to(device),
            "observation.images.secondary_1": batch[
                "observation.images.secondary_1"
            ].to(device),
            "observation.state": batch["observation.state"].to(device),
        }
        print(
            f"Debug: obs['observation.images.main'].shape = {obs['observation.images.main'].shape}"
        )
        print(
            f"Debug: obs['observation.images.secondary_0'].shape = {obs['observation.images.secondary_0'].shape}"
        )
        print(
            f"Debug: obs['observation.images.secondary_1'].shape = {obs['observation.images.secondary_1'].shape}"
        )
        print(
            f"Debug: obs['observation.state'].shape = {obs['observation.state'].shape}"
        )

        gt_action = batch["action"].to(device)  # [B, 14]
        print(f"Debug: gt_action.shape = {gt_action.shape}")
        if action_dim is None:
            action_dim = gt_action.shape[-1]
        # Forward policy
        out = policy.select_action(obs)
        print(f"Debug: out.shape = {out.shape}")
        # print(out)

        if out.ndim == 3:
            # [B, H, A] -> take first timestep
            pred_action = out[:, 0, :]
        elif out.ndim == 2:
            pred_action = out
        else:
            raise RuntimeError(f"Unexpected action shape from policy: {out.shape}")

        assert (
            pred_action.shape == gt_action.shape
        ), f"pred {pred_action.shape}, gt {gt_action.shape}"

        err = pred_action - gt_action
        all_sq_err.append((err**2).cpu().numpy())  # each is [B, A]
        all_abs_err.append(err.abs().cpu().numpy())  # each is [B, A]

    if not all_sq_err:
        print("⚠️ No batches evaluated, check your dataloader / max_batches.")
        return

    # ---- concatenate over all batches: [N, A] ----
    all_sq_err = np.concatenate(all_sq_err, axis=0)
    all_abs_err = np.concatenate(all_abs_err, axis=0)

    # ---- mask out joints 6 and 13 (almost constant) ----
    mask = np.ones(action_dim, dtype=bool)
    mask[[6, 13]] = False  # ignore these two joints

    all_sq_err_masked = all_sq_err[:, mask]  # [N, A_eff]
    all_abs_err_masked = all_abs_err[:, mask]  # [N, A_eff]

    # ========= Common Metrics (on effective joints only) =========
    mse_per_dim = all_sq_err_masked.mean(axis=0)  # [A_eff]
    rmse_per_dim = np.sqrt(mse_per_dim)  # [A_eff]
    mae_per_dim = all_abs_err_masked.mean(axis=0)  # [A_eff]

    # L2 error per timestep, then average
    l2_per_step = np.sqrt(all_sq_err_masked.sum(axis=1))  # [N]
    mean_l2 = l2_per_step.mean()

    # Fraction of joint angles within error thresholds (5 deg / 10 deg)
    DEG5 = np.deg2rad(5.0)
    DEG10 = np.deg2rad(10.0)
    within_5deg = (all_abs_err_masked < DEG5).mean()
    within_10deg = (all_abs_err_masked < DEG10).mean()

    print("\n===== Offline Evaluation Metrics (masked joints 6 & 13) =====")
    print(
        f"Evaluated samples: {all_sq_err_masked.shape[0]}  (action_dim={action_dim}, effective_dim={mask.sum()})"
    )
    print(f"Mean L2 action error (rad): {mean_l2:.4f}")
    print(f"Mean per-dim RMSE (rad): {rmse_per_dim.mean():.4f}")
    print(f"Mean per-dim MAE  (rad): {mae_per_dim.mean():.4f}")
    print(f"Fraction of (joint, timestep) with |error| < 5deg:  {within_5deg*100:.2f}%")
    print(
        f"Fraction of (joint, timestep) with |error| < 10deg: {within_10deg*100:.2f}%"
    )

    # Per-joint breakdown (only for kept joints)
    kept_indices = np.where(mask)[0]  # original joint indices
    for i, j in enumerate(kept_indices):
        print(
            f"  Joint {j:02d}: RMSE={rmse_per_dim[i]:.4f} rad, "
            f"MAE={mae_per_dim[i]:.4f} rad"
        )


def main():
    args = parse_args()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dl = build_dataloader(args.dataset_repo, args.batch_size)
    policy = load_policy(args.policy_ckpt, device)
    run_eval(dl, policy, args.max_batches, device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to upload trained checkpoint directory to Hugging Face Hub.

Typical usage (uploading a DiffusionPolicy / ACT pretrained_model folder):

    python upload_ckpt_to_hf.py \
        --repo_id Anlorla/pass_cucumber\
        --ckpt_dir ../openpi/checkpoints/pi05_npm_lora/pass_cucumber/ \
        --repo_type model 

Notes:
- Run `huggingface-cli login` first, or provide --token in the command.
- repo_type is usually "model" (for policy weights), use "dataset" for datasets.
"""

import os
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login


def upload_ckpt(
    repo_id: str,
    ckpt_dir: str,
    repo_type: str = "model",
    token: str | None = None,
    private: bool = True,
    commit_message: str | None = None,
):
    ckpt_path = Path(ckpt_dir).expanduser().resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ ckpt_dir does not exist: {ckpt_path}")

    if not ckpt_path.is_dir():
        raise NotADirectoryError(f"❌ ckpt_dir is not a directory: {ckpt_path}")

    print(f"📁 Directory to upload: {ckpt_path}")
    print(f"📦 Target HF repository: {repo_id} (type={repo_type}, private={private})")

    # Login to HF (if token is provided)
    if token:
        print("🔑 Logging in to Hugging Face with provided token...")
        login(token=token)
    else:
        print(
            "ℹ️  No token provided, using locally logged-in Hugging Face account (if any)."
        )

    api = HfApi()

    # Create or reuse repository
    print("🛠️  Creating/checking repository...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,  # Reuse if already exists, won't raise error
    )
    print("✅ Repository ready.")

    # Default commit message
    if commit_message is None:
        commit_message = f"Upload checkpoint from {ckpt_path.name}"

    print("⬆️  Starting upload of entire directory to HF Hub...")
    print(
        "💡 Large files will be uploaded with multi-commit support for better reliability."
    )
    api.upload_folder(
        folder_path=str(ckpt_path),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        # multi_commits=True,  # Enable multi-commit for large files (>5GB)
        # multi_commits_verbose=True,  # Show detailed progress for each file
        ignore_patterns=[
            "*.pyc",
            "__pycache__",
            ".git*",
            "*.tmp",
            "*.swp",
        ],  # Ignore temp files
    )

    print("🎉 Upload completed!")
    print(f"🔗 You can now open in browser: https://huggingface.co/{repo_id}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload checkpoint directory to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HF repository ID, e.g., your-name/your-model-name",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Checkpoint directory to upload, e.g., .../checkpoints/last/pretrained_model",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="HF repository type, usually 'model' for policy/weights, 'dataset' for datasets",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional: Hugging Face token. If not provided, uses locally logged-in credentials.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/keep as private repository (default False for public)",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Optional: Commit message. If not provided, auto-generates a simple message.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upload_ckpt(
        repo_id=args.repo_id,
        ckpt_dir=args.ckpt_dir,
        repo_type=args.repo_type,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
    )

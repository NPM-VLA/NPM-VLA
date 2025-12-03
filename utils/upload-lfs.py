"""
上传大文件（例如 sweep2z.zip）到 Hugging Face 的脚本（通过 git + git-lfs）

用法示例：
    python upload-lfs.py --repo_id Anlorla/Sweep2Z --file_path "D:\DESKTOP\Sweep2Z\sweep2z.zip"  --repo_type dataset

注意：
  - 单个文件远大于 5GB 时，不能再用 HfApi.upload_file / upload_folder，
    必须走 git-lfs。
  - 运行前请确保：
      1) 已安装 git 和 git-lfs，并且在 PATH 中
      2) 已执行过: huggingface-cli login   （或者传入 --token）
"""

import os
import argparse
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login


def run_cmd(cmd, cwd=None, allow_fail=False):
    """在子进程里运行命令行并输出，可选允许失败"""
    print(f"[CMD] {' '.join(cmd)}  (cwd={cwd or os.getcwd()})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"命令执行失败，退出码 {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def upload_large_file(
    repo_id: str,
    file_path: str,
    repo_type: str = "dataset",
    token: str | None = None,
    clone_dir: str | None = None,
):
    """
    使用 git + git-lfs 上传一个大文件到 Hugging Face 仓库

    Args:
        repo_id: "username/dataset-name" 或 "org/name"
        file_path: 本地大文件路径，例如 "D:/DESKTOP/sweep2z.zip"
        repo_type: "dataset" 或 "model"（默认 dataset）
        token: 可选，如果已经 huggingface-cli login 过可以不传
        clone_dir: 可选，本地 clone 仓库的目录名，默认用 repo_id 里的 name
    """
    print(f"🚀 开始上传大文件到 {repo_id}")
    print(f"📁 本地文件: {file_path}")
    print(f"📦 仓库类型: {repo_type}")

    # 登录 Hugging Face（可选）
    if token:
        login(token=token)
        print("✅ 已使用 token 登录 Hugging Face")
        api = HfApi(token=token)
    else:
        print("ℹ️  使用已有的登录信息（huggingface-cli login 保存的）")
        api = HfApi()

    # 1) 创建/确认仓库存在
    print("📂 确保远端仓库已存在...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        exist_ok=True,
    )
    if repo_type == "dataset":
        print(f"✅ 数据集仓库: https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"✅ 模型仓库:   https://huggingface.co/{repo_id}")

    # 2) 检查本地文件
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到文件: {file_path}")

    if clone_dir is None:
        # 默认 clone 到当前目录下的 repo 名称部分
        clone_dir = repo_id.split("/")[-1]
    clone_dir = Path(clone_dir)

    # 3) 克隆仓库（如果本地目录不存在）
    if repo_type == "dataset":
        clone_url = f"https://huggingface.co/datasets/{repo_id}"
    else:
        clone_url = f"https://huggingface.co/{repo_id}"

    if not clone_dir.exists():
        print(f"📥 克隆仓库到本地: {clone_url} -> {clone_dir}")
        run_cmd(["git", "clone", clone_url, str(clone_dir)])
    else:
        print(f"ℹ️  本地目录已存在，跳过 clone: {clone_dir}")

    # 4) 在仓库目录里初始化 git-lfs 并启用 largefiles
    print("🔧 初始化 git-lfs")
    run_cmd(["git", "lfs", "install"], cwd=str(clone_dir), allow_fail=True)

    print("🔧 启用 largefiles 支持（避免大文件限制）")
    run_cmd(["huggingface-cli", "lfs-enable-largefiles", "."], cwd=str(clone_dir))

    # 5) 拷贝大文件到仓库目录
    dest_path = clone_dir / file_path.name
    if dest_path.resolve() != file_path.resolve():
        print(f"📄 拷贝文件到仓库目录: {dest_path}")
        shutil.copy2(file_path, dest_path)
    else:
        print("ℹ️  文件已经在仓库目录中，跳过拷贝")

    # 6) git add / commit / push
    print("➕ git add")
    run_cmd(["git", "add", dest_path.name], cwd=str(clone_dir))

    print("📝 git commit")
    # commit 时如果没有变化会返回非 0，这里允许失败
    run_cmd(
        ["git", "commit", "-m", f"Add {dest_path.name}"],
        cwd=str(clone_dir),
        allow_fail=True,
    )

    print("📤 git push 到 Hugging Face")
    run_cmd(["git", "push"], cwd=str(clone_dir))

    print("\n✅ 上传完成！")
    if repo_type == "dataset":
        print(f"🔗 访问地址: https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"🔗 访问地址: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 git-lfs 上传大文件到 Hugging Face"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Anlorla/Sweep2Z",
        help="Hugging Face 仓库 ID，例如: username/dataset-name",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="本地大文件路径，例如: D:\\DESKTOP\\sweep2z.zip",
    )
    parser.add_argument(
        "--repo_type",
        type=str,
        default="dataset",
        choices=["dataset", "model"],
        help="仓库类型（默认: dataset）",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="可选，如果已经 huggingface-cli login 过可以不填",
    )
    parser.add_argument(
        "--clone_dir",
        type=str,
        default=None,
        help="本地 clone 仓库的目录名（不填则默认用 repo 名称）",
    )

    args = parser.parse_args()

    upload_large_file(
        repo_id=args.repo_id,
        file_path=args.file_path,
        repo_type=args.repo_type,
        token=args.token,
        clone_dir=args.clone_dir,
    )


if __name__ == "__main__":
    main()

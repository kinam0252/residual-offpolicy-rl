from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def _default_iface_script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "workspace" / "franka_cogact_motion_generation_abs_window_gr00t_training_only_iface.py"


def _default_isaaclab_sh() -> Path:
    return Path("/workspace/isaaclab/isaaclab.sh")


def main() -> None:
    parser = argparse.ArgumentParser(description="Iface-based residual-zero rollout on IsaacLab")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--groot_model_path", type=str, required=True)
    parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--groot_policy_device", type=str, default=None)
    parser.add_argument("--groot_policy_strict", action="store_true")
    parser.add_argument("--task_description", type=str, default="Pick up the white object.")
    parser.add_argument("--language_override", type=str, default=None)
    parser.add_argument("--csv_base_dir", type=str, default=None)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--isaaclab_sh", type=str, default=str(_default_isaaclab_sh()))
    parser.add_argument("--iface_script_path", type=str, default=str(_default_iface_script_path()))
    args = parser.parse_args()

    isaaclab_sh = Path(args.isaaclab_sh).expanduser().resolve()
    iface_script = Path(args.iface_script_path).expanduser().resolve()
    output_npz = Path(args.output_npz).expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(isaaclab_sh),
        "-p",
        str(iface_script),
        "--enable_cameras",
        "--num_envs",
        str(args.num_envs),
        "--model_path",
        str(Path(args.groot_model_path).expanduser().resolve()),
        "--embodiment_tag",
        str(args.groot_embodiment_tag),
        "--task_description",
        str(args.task_description),
        "--max_steps",
        str(args.max_steps),
        "--max_attempts_per_episode",
        "1",
        "--rl_disable_retries",
        "--online_buffer_npz",
        str(output_npz),
    ]

    if args.csv_base_dir:
        cmd.extend(["--csv_dir", str(Path(args.csv_base_dir).expanduser().resolve())])
    if args.language_override is not None:
        cmd.extend(["--language_override", str(args.language_override)])
    if args.groot_policy_device:
        cmd.extend(["--policy_device", str(args.groot_policy_device)])
    if args.groot_policy_strict:
        cmd.append("--policy_strict")
    if args.headless:
        cmd.append("--headless")

    print("[rollout-iface] cmd=" + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"iface rollout failed with code {proc.returncode}")
    if not output_npz.exists():
        raise RuntimeError(f"iface rollout finished but npz not found: {output_npz}")

    print(f"saved_npz={output_npz}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np


def _default_iface_script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "workspace" / "franka_cogact_motion_generation_abs_window_gr00t_training_only_iface.py"


def _default_isaaclab_sh() -> Path:
    return Path("/workspace/isaaclab/isaaclab.sh")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one online buffer episode via iface-based runner")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gr00t_host", type=str, default="127.0.0.1")
    parser.add_argument("--gr00t_port", type=int, default=5555)
    parser.add_argument("--groot_model_path", type=str, required=True)
    parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--groot_policy_device", type=str, default=None)
    parser.add_argument("--groot_policy_strict", action="store_true")
    parser.add_argument("--task_description", type=str, default="Pick up the white object.")
    parser.add_argument("--language_override", type=str, default=None)
    parser.add_argument("--csv_base_dir", type=str, default=None)
    parser.add_argument("--csv_init_row_index", type=int, default=0)
    parser.add_argument("--lift_reward_threshold_m", type=float, default=0.01)
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--iface_script_path", type=str, default=str(_default_iface_script_path()))
    parser.add_argument("--isaaclab_sh", type=str, default=str(_default_isaaclab_sh()))
    return parser


def _run_iface_export(args: argparse.Namespace) -> None:
    iface_script = Path(args.iface_script_path).expanduser().resolve()
    isaaclab_sh = Path(args.isaaclab_sh).expanduser().resolve()
    output_npz = Path(args.output_npz).expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    if not iface_script.exists():
        raise FileNotFoundError(f"iface script not found: {iface_script}")
    if not isaaclab_sh.exists():
        raise FileNotFoundError(f"isaaclab.sh not found: {isaaclab_sh}")

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
        "--lift_reward_threshold_m",
        str(args.lift_reward_threshold_m),
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

    print("[export-iface] launching iface-based exporter", flush=True)
    print("[export-iface] cmd=" + " ".join(cmd), flush=True)

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"iface-based export failed with code {proc.returncode}")

    if not output_npz.exists():
        raise RuntimeError(f"iface-based export finished but npz not found: {output_npz}")


def _print_npz_summary(npz_path: Path) -> None:
    data = np.load(npz_path)
    steps = int(data["timestep"].shape[0]) if "timestep" in data else -1
    done = bool(data["done"][-1]) if ("done" in data and data["done"].size > 0) else False
    terminal_reward = float(data["reward"][-1]) if ("reward" in data and data["reward"].size > 0) else 0.0
    reward_trigger_step = int(data["reward_trigger_step"]) if "reward_trigger_step" in data else -1
    episode_success = bool(data["episode_success"]) if "episode_success" in data else bool(np.any(data.get("reward", np.asarray([])) >= 1.0))

    print(f"saved_npz={npz_path}")
    print(f"steps={steps} done={done}")
    print(f"terminal_reward={terminal_reward:.6f}")
    print(f"reward_trigger_step={reward_trigger_step}")
    print(f"episode_success={episode_success}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _run_iface_export(args)
    _print_npz_summary(Path(args.output_npz).expanduser().resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[export-iface][fatal] {exc}", flush=True)
        raise

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def _default_iface_script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "workspace" / "franka_cogact_motion_generation_abs_window_gr00t_training_only_iface.py"


def _default_isaaclab_sh() -> Path:
    return Path("/workspace/isaaclab/isaaclab.sh")


def run_iface_evaluation(
    *,
    csv_base_dir: str,
    groot_model_path: str,
    num_episodes: int,
    max_steps: int,
    output_dir: str | Path,
    run_name: str,
    global_step: int,
    num_envs: int = 1,
    policy_device: str | None = None,
    embodiment_tag: str = "new_embodiment",
    task_description: str = "Pick up the white object.",
    language_override: str | None = None,
    headless: bool = True,
    save_video: bool = False,
    lift_reward_threshold_m: float = 0.01,
    isaaclab_sh: str | None = None,
    iface_script_path: str | None = None,
) -> tuple[dict[str, float], Path | None]:
    isaaclab_sh_path = Path(isaaclab_sh).expanduser().resolve() if isaaclab_sh else _default_isaaclab_sh()
    iface_script = Path(iface_script_path).expanduser().resolve() if iface_script_path else _default_iface_script_path()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_parent = output_root / run_name.split("__")[0]
    run_parent.mkdir(parents=True, exist_ok=True)

    csv_dir = Path(csv_base_dir).expanduser().resolve()
    episode_name = csv_dir.name

    episode_successes: list[float] = []
    episode_returns: list[float] = []
    last_video_path: Path | None = None

    for ep_idx in range(max(1, int(num_episodes))):
        npz_path = run_parent / f"iface_eval_ep{ep_idx + 1:03d}_step{global_step}.npz"
        success_csv = run_parent / f"iface_eval_ep{ep_idx + 1:03d}_step{global_step}.csv"

        cmd = [
            str(isaaclab_sh_path),
            "-p",
            str(iface_script),
            "--enable_cameras",
            "--num_envs",
            str(num_envs),
            "--model_path",
            str(Path(groot_model_path).expanduser().resolve()),
            "--embodiment_tag",
            str(embodiment_tag),
            "--task_description",
            str(task_description),
            "--csv_dir",
            str(csv_dir),
            "--max_steps",
            str(max_steps),
            "--lift_reward_threshold_m",
            str(lift_reward_threshold_m),
            "--max_attempts_per_episode",
            "1",
            "--rl_disable_retries",
            "--online_buffer_npz",
            str(npz_path),
            "--success_log_csv",
            str(success_csv),
        ]

        if language_override is not None:
            cmd.extend(["--language_override", str(language_override)])
        if policy_device:
            cmd.extend(["--policy_device", str(policy_device)])
        if headless:
            cmd.append("--headless")

        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            tail_lines = []
            if proc.stdout:
                tail_lines.extend(proc.stdout.strip().splitlines()[-20:])
            if proc.stderr:
                tail_lines.extend(proc.stderr.strip().splitlines()[-20:])
            tail_msg = "\n".join(tail_lines[-40:]).strip()
            detail = f"\n{tail_msg}" if tail_msg else ""
            raise RuntimeError(
                f"iface evaluation failed at episode {ep_idx + 1} with code {proc.returncode}{detail}"
            )

        if not npz_path.exists():
            raise RuntimeError(f"iface evaluation npz missing: {npz_path}")

        data = np.load(npz_path)
        reward = np.asarray(data["reward"], dtype=np.float32) if "reward" in data else np.asarray([], dtype=np.float32)
        done = np.asarray(data["done"], dtype=np.bool_) if "done" in data else np.asarray([], dtype=np.bool_)
        episode_success = float(np.any(reward >= 1.0))
        episode_return = float(np.sum(reward)) if reward.size else 0.0
        episode_successes.append(episode_success)
        episode_returns.append(episode_return)

        src_video = Path(f"/home/t-kinamkim/Repos/VLA_RL/outputs_gr00t/{episode_name}.try01.mp4")
        if save_video and src_video.exists():
            dst_video = run_parent / f"eval_{run_name}_step_{global_step}_ep{ep_idx + 1:03d}.mp4"
            frames = [fr for fr in imageio.get_reader(str(src_video))]
            writer = imageio.get_writer(str(dst_video), fps=20)
            for fr in frames:
                writer.append_data(fr)
            writer.close()
            last_video_path = dst_video

    metrics = {
        "eval/success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "eval/mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "eval/mean_successful_episode_length": 0.0,
    }
    return metrics, last_video_path

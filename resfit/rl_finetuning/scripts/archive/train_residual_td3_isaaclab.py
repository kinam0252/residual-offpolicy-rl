"""
Residual TD3 training on IsaacLab + GR00T.

Adapted from ``resfit/rl_finetuning/scripts/train_residual_td3.py``.
Replaces robosuite + ACT with IsaacLab + GR00T while reusing:
  - QAgent (TD3 actor/critic)
  - Replay buffers (online + offline with n-step returns)
  - Prioritised experience replay
  - Critic warmup + progressive clipping

Launch (from isaaclab.sh runtime):
    isaaclab.sh -p resfit/rl_finetuning/scripts/train_residual_td3_isaaclab.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")  # RTX 5090 (sm_120)

# ── Ensure resfit's isaaclab_tasks takes priority over Honda_IsaacLab's ──
_resfit_tasks_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "isaaclab", "source", "isaaclab_tasks")
_resfit_tasks_path = os.path.abspath(_resfit_tasks_path)
if os.path.isdir(_resfit_tasks_path) and _resfit_tasks_path not in sys.path:
    sys.path.insert(0, _resfit_tasks_path)

# ── CRITICAL: AppLauncher must run BEFORE any isaaclab imports ──
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Residual TD3 on IsaacLab + GR00T")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--total_timesteps", type=int, default=300_000)
parser.add_argument("--learning_starts", type=int, default=None)
parser.add_argument("--critic_warmup_steps", type=int, default=None)
parser.add_argument("--update_every_n_steps", type=int, default=None)
parser.add_argument("--num_updates_per_iteration", type=int, default=None)
parser.add_argument("--offline_fraction", type=float, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--buffer_size", type=int, default=None)
parser.add_argument("--gamma", type=float, default=None)
parser.add_argument("--n_step", type=int, default=None)
parser.add_argument("--stddev_max", type=float, default=None)
parser.add_argument("--stddev_min", type=float, default=None)
parser.add_argument("--actor_lr", type=float, default=None)
parser.add_argument("--action_scale", type=float, default=None)
parser.add_argument("--random_action_noise_scale", type=float, default=None)
parser.add_argument("--gr00t_host", type=str, default="127.0.0.1")
parser.add_argument("--gr00t_port", type=int, default=5555)
parser.add_argument("--groot_model_path", type=str, default=None)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default=None)
parser.add_argument("--groot_policy_strict", action="store_true")
parser.add_argument("--task_description", type=str, default=None)
parser.add_argument("--language_override", type=str, default=None)
parser.add_argument("--csv_base_dir", type=str, default=None)
parser.add_argument("--csv_init_row_index", type=int, default=None)
parser.add_argument("--max_episode_steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb_mode", type=str, default="disabled")
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--wandb_notes", type=str, default=None)
parser.add_argument("--wandb_continue_run_id", type=str, default=None)
parser.add_argument("--wandb_log_every_steps", type=int, default=10)
parser.add_argument("--eval_interval_every_steps", type=int, default=None)
parser.add_argument("--eval_num_episodes", type=int, default=None)
parser.add_argument("--eval_num_envs", type=int, default=None)
parser.add_argument("--eval_first", action="store_true")
parser.add_argument("--eval_use_iface_runner", type=int, choices=[0, 1], default=None)
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--eval_save_video", action="store_true", help="Save video in async eval")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--debug_zero_residual", action="store_true")
parser.add_argument("--debug_first_episode_video", action="store_true", help="Record first training episode with reward component overlay")
parser.add_argument("--disable_eval", action="store_true", help="Skip inline eval (use async eval process)")
parser.add_argument("--heartbeat_interval_sec", type=int, default=20)
parser.add_argument("--stack_dump_interval_sec", type=int, default=180)
parser.add_argument("--warmup_video_path", type=str, default=None)
parser.add_argument("--warmup_video_fps", type=int, default=20)
parser.add_argument("--warmup_video_every_n_steps", type=int, default=1)
parser.add_argument("--iface_shm_bytes", type=int, default=268435456)
parser.add_argument("--iface_timeout_s", type=int, default=0)
parser.add_argument("--warmup_iface_max_steps", type=int, default=1000)
parser.add_argument("--warmup_collector", type=str, choices=["inprocess", "iface_ipc", "inprocess_iface_render"], default="inprocess_iface_render")
parser.add_argument("--groot_input_dump_jsonl", type=str, default=None)
parser.add_argument("--replay_actions_npz", type=str, default=None)
parser.add_argument("--force_infer_on_reset", action="store_true")
parser.add_argument("--skip_groot_model_load", action="store_true")
parser.add_argument("--max_groot_dumps", type=int, default=None)
parser.add_argument("--iface_camera_warmup_steps", type=int, default=10)
parser.add_argument("--iface_post_reset_settle_steps", type=int, default=200)
parser.add_argument("--offline_data_dir", type=str, default=None, help="Path to LeRobot-format offline data dir")
parser.add_argument("--success_threshold", type=float, default=None, help="Cube lift success threshold in meters")
parser.add_argument("--cube_perturb_range", type=float, default=None, help="Cube XY position perturbation range in meters")
parser.add_argument("--cube_perturb_table_path", type=str, default=None, help="Path to JSON cube perturbation table")
parser.add_argument("--random_cube_perturb", action="store_true", help="Randomize cube XY+yaw on each episode reset during training")
parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from (loads weights only, not optimizer)")
parser.add_argument("--reward_type", type=str, default=None, choices=["sparse", "dense", "dense_clipped"], help="Reward type")
parser.add_argument("--critic_hidden_dim", type=int, default=None, help="Critic MLP hidden dim (default 1024)")
parser.add_argument("--actor_hidden_dim", type=int, default=None, help="Actor MLP hidden dim (default 1024)")
parser.add_argument("--vlm_projected_dim", type=int, default=None, help="VLM projector output dim (default 256)")
parser.add_argument("--phase_probe_path", type=str, default=None, help="Path to pre-trained phase probe .pt (frozen linear 2048->3)")
parser.add_argument("--no_clip_q", action="store_true", help="Disable Q-target clipping to reward range")
parser.add_argument("--actor_lr_warmup_steps", type=int, default=None, help="Actor LR warmup steps (0 to full lr)")
parser.add_argument("--action_scale_anneal_steps", type=int, default=None, help="Anneal action_scale from 0.01 to target over N steps")
parser.add_argument("--use_depth", action="store_true", help="Enable depth observations for residual actor (sim2real mode)")
parser.add_argument("--depth_norm_path", type=str, default=None, help="Path to depth_normalization.json")
parser.add_argument("--use_vlm", action="store_true", help="Enable VLM latent features for actor/critic")
parser.add_argument("--use_state", action="store_true", help="Enable object state (cube 6D pose) as additional low-dim input")
parser.add_argument("--asymmetric_critic", action="store_true", help="Asymmetric actor-critic: actor=state-only, critic=depth+state (requires --use_depth + --use_state)")
parser.add_argument("--object_state_mode", type=str, default="raw", choices=["raw", "relative", "full"],
                    help="Object state representation: raw=pos+quat(7D), relative=relpos+contact(4D), full=relpos+quat+contact(8D)")
parser.add_argument("--contact_binary", action="store_true", help="Use binary contact (0/1) instead of continuous force magnitude")
parser.add_argument("--min_save_success_rate", type=float, default=0.0, help="Only save periodic checkpoints when best success rate >= this threshold (0=always save)")
parser.add_argument("--checkpoint_min_success", type=float, default=0.0, help="Only save periodic checkpoints when best_success >= this threshold (0=always save)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.groot_input_dump_jsonl:
    os.environ["RESFIT_GROOT_INPUT_DUMP"] = os.path.abspath(os.path.expanduser(str(args_cli.groot_input_dump_jsonl)))
if args_cli.replay_actions_npz:
    os.environ["RESFIT_REPLAY_ACTIONS_NPZ"] = os.path.abspath(os.path.expanduser(str(args_cli.replay_actions_npz)))
if args_cli.force_infer_on_reset:
    os.environ["RESFIT_FORCE_INFER_ON_RESET"] = "1"
if args_cli.skip_groot_model_load:
    os.environ["RESFIT_SKIP_GROOT_MODEL_LOAD"] = "1"
if args_cli.max_groot_dumps is not None:
    os.environ["RESFIT_MAX_GROOT_DUMPS"] = str(int(args_cli.max_groot_dumps))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Cancel Isaac Sim's 3-minute faulthandler watchdog ──
# Isaac Sim sets faulthandler.dump_traceback_later(180, exit=True) at startup.
# Extensions can re-set it asynchronously, so we cancel repeatedly for 5 min.
import faulthandler as _fh
import threading as _threading
try:
    _fh.cancel_dump_traceback_later()
except Exception:
    pass

def _keep_cancelling_faulthandler(duration_sec=300, interval_sec=10):
    """Background thread that keeps cancelling faulthandler for `duration_sec`."""
    import time as _time
    deadline = _time.time() + duration_sec
    while _time.time() < deadline:
        try:
            _fh.cancel_dump_traceback_later()
        except Exception:
            pass
        _time.sleep(interval_sec)

_fh_cancel_thread = _threading.Thread(
    target=_keep_cancelling_faulthandler, daemon=True, name="fh-cancel"
)
_fh_cancel_thread.start()

# ── Now safe to import isaaclab / isaaclab_tasks ──
import hashlib
import io
import json
import logging
import pprint
import random
import secrets
import socket
import subprocess
import tempfile
import threading
import time
import faulthandler
from multiprocessing import shared_memory
from multiprocessing.connection import Client
from dataclasses import asdict, is_dataclass
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import tensordict
import torch
import torchrl
import gymnasium as gym
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

import isaaclab.sim as sim_utils

try:
    import wandb
except ImportError:
    wandb = None

from resfit.rl_finetuning.config.residual_td3_isaaclab import ResidualTD3IsaacLabConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.offline_buffer_csv import populate_offline_buffer_from_csv
from resfit.rl_finetuning.utils.offline_buffer_lerobot_local import (
    populate_offline_buffer_from_lerobot_local,
    validate_offline_reward_formula,
)
from resfit.rl_finetuning.utils.iface_eval_runner import run_iface_evaluation
from resfit.rl_finetuning.utils.custom_env import (
    compute_env_hash, snapshot_exists, save_env_snapshot,
    load_env_snapshot, capture_sanity_frames,
)


# ─────────────────────────────────────────────────────────────────
# Async Evaluator — launches eval_async_isaaclab.py as a subprocess
# and collects results via JSON files, never blocking training.
# ─────────────────────────────────────────────────────────────────
class AsyncEvaluator:
    """Manages a long-running eval subprocess with its own Isaac Sim instance.

    Lifecycle:
      1. __init__: builds the eval launch command (does NOT start yet)
      2. start(): launches the subprocess in background
      3. trigger_eval(agent, step, checkpoint_dir): saves checkpoint, eval picks it up
      4. collect_results(): non-blocking check for finished eval results
      5. shutdown(): kills the subprocess
    """

    def __init__(self, cfg, container: str = "isaaclab_resfit"):
        self._proc = None
        self._container = container
        self._cfg = cfg
        self._pending_step: int | None = None
        self._results_dir: Path | None = None
        self._best_success = 0.0

    def start(self, checkpoint_dir: Path, results_dir: Path, wandb_run_id: str | None = None):
        """Launch the async eval subprocess.

        Works in both Docker (isaaclab.sh) and Apptainer (/isaac-sim/python.sh)
        environments by detecting which runtime is available.
        """
        self._results_dir = results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        self._wandb_run_id = wandb_run_id

        ecfg = self._cfg.isaaclab_env
        gcfg = self._cfg.groot_policy

        eval_script = str(Path(__file__).parent / "eval_async_isaaclab.py")
        launcher_path = results_dir / "_eval_launcher.sh"

        # Detect runtime: Apptainer (has /isaac-sim/python.sh) vs Docker (has /workspace/isaaclab/isaaclab.sh)
        _is_apptainer = Path("/isaac-sim/python.sh").exists()
        _is_docker = Path("/workspace/isaaclab/isaaclab.sh").exists()

        # Inherit PYTHONPATH from current environment (set by env.sh or Docker)
        inherited_pythonpath = os.environ.get("PYTHONPATH", "")
        wandb_key = os.environ.get("WANDB_API_KEY", "")

        if _is_apptainer:
            # Apptainer: we're already inside the container, use /isaac-sim/python.sh directly
            launcher_content = f"""#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="{inherited_pythonpath}"
export WANDB_API_KEY="{wandb_key}"
exec /isaac-sim/python.sh {eval_script} \\
  --headless --enable_cameras \\"""
        elif _is_docker:
            # Docker: use isaaclab.sh launcher
            launcher_content = f"""#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="{inherited_pythonpath}"
export WANDB_API_KEY="{wandb_key}"
cd /workspace/isaaclab
exec ./isaaclab.sh -p {eval_script} \\
  --headless --enable_cameras \\"""
        else:
            # Fallback: try python directly (for local dev)
            _log("[AsyncEval] WARNING: Neither Apptainer nor Docker detected, using sys.executable")
            launcher_content = f"""#!/usr/bin/env bash
set -e
export TERM=xterm
export PYTHONUNBUFFERED=1
export PYTHONPATH="{inherited_pythonpath}"
export WANDB_API_KEY="{wandb_key}"
exec {sys.executable} {eval_script} \\
  --headless --enable_cameras \\"""

        launcher_content += f"""
  --checkpoint_dir "{checkpoint_dir}" \\
  --num_envs {int(getattr(self._cfg, 'eval_num_envs', 10))} \\
  --max_episode_steps {int(ecfg.max_episode_steps)} \\
  --success_threshold {float(getattr(ecfg, 'success_threshold', 0.005))} \\
  --groot_model_path "{gcfg.model_path}" \\
  --groot_embodiment_tag "{gcfg.embodiment_tag}" \\
  --groot_policy_device "{gcfg.policy_device or 'cuda:0'}" \\
  --csv_base_dir "{ecfg.csv_base_dir}" \\
  --language_override "{gcfg.language_override or gcfg.task_description}" \\
  --output_dir "{results_dir}" \\
  --wandb_mode "{self._cfg.wandb.mode}" \\
  --wandb_project "{self._cfg.wandb.project}" \\
  --wandb_entity "{getattr(self._cfg.wandb, 'entity', '')}" \\
  --wandb_name "{getattr(self._cfg.wandb, 'name', 'train')}_async_eval" \\
  --poll_interval_sec 15"""
        # Optional: save video in eval
        _eval_save_video = bool(getattr(self._cfg, 'eval_save_video', getattr(self._cfg, 'save_video', False)))
        if _eval_save_video:
            launcher_content = launcher_content.rstrip() + ' \\\n  --save_video'
        # Add optional args
        _perturb_path = getattr(ecfg, 'cube_perturb_table_path', None)
        if _perturb_path:
            launcher_content = launcher_content.rstrip() + f' \\\n  --cube_perturb_table_path "{_perturb_path}"'
        # Pass custom_env_dir so eval uses same cube snapshot as training
        _custom_env_dir = str(checkpoint_dir.parent / "custom_env")
        launcher_content = launcher_content.rstrip() + f' \\\n  --custom_env_dir "{_custom_env_dir}"'
        # Pass action_scale so eval uses same scale as training
        _action_scale = getattr(self._cfg.agent.actor, 'action_scale', 0.1)
        launcher_content = launcher_content.rstrip() + f' \\\n  --action_scale {_action_scale}'
        # Pass reward_type so eval uses same reward as training
        _reward_type = getattr(ecfg, 'reward_type', 'dense_clipped')
        launcher_content = launcher_content.rstrip() + f' \\\n  --reward_type {_reward_type}'
        # Pass phase_probe_path if it was used in training
        if hasattr(self, '_phase_probe_path') and self._phase_probe_path:
            launcher_content = launcher_content.rstrip() + f' \\\n  --phase_probe_path "{self._phase_probe_path}"'
        # Pass depth mode if enabled
        if hasattr(self, '_use_depth') and self._use_depth:
            launcher_content = launcher_content.rstrip() + ' \\\n  --use_depth'
            if hasattr(self, '_depth_norm_path') and self._depth_norm_path:
                launcher_content = launcher_content.rstrip() + f' \\\n  --depth_norm_path "{self._depth_norm_path}"'
        # Pass VLM mode if enabled
        if hasattr(self, '_use_vlm') and self._use_vlm:
            launcher_content = launcher_content.rstrip() + ' \\\n  --use_vlm'
        # Pass state mode if enabled
        if hasattr(self, '_use_state') and self._use_state:
            launcher_content = launcher_content.rstrip() + ' \\\n  --use_state'
            _osm = getattr(self, '_object_state_mode', 'raw')
            if _osm != 'raw':
                launcher_content = launcher_content.rstrip() + f' \\\n  --object_state_mode {_osm}'
            if getattr(self, '_contact_binary', False):
                launcher_content = launcher_content.rstrip() + ' \\\n  --contact_binary'
        # Pass critic hidden dim if non-default
        _chd = getattr(self, '_critic_hidden_dim', None)
        if _chd is not None:
            launcher_content = launcher_content.rstrip() + f' \\\n  --critic_hidden_dim {_chd}'
        _ahd = getattr(self, '_actor_hidden_dim', None)
        if _ahd is not None:
            launcher_content = launcher_content.rstrip() + f' \\\n  --actor_hidden_dim {_ahd}'
        # Pass asymmetric critic mode
        if getattr(self, '_asymmetric_critic', False):
            launcher_content = launcher_content.rstrip() + ' \\\n  --asymmetric_critic'
        launcher_content += "\n"
        launcher_path.write_text(launcher_content)
        launcher_path.chmod(0o755)

        log_path = results_dir / "async_eval.log"
        import subprocess
        # We're already INSIDE the container, so run bash directly (no docker exec)
        self._proc = subprocess.Popen(
            ["bash", str(launcher_path)],
            stdout=open(str(log_path), "w"),
            stderr=subprocess.STDOUT,
        )
        self._log_path = log_path
        _log(f"[AsyncEval] Subprocess started PID={self._proc.pid}, log={log_path}")

    def trigger_eval(self, agent, global_step: int, checkpoint_dir: Path):
        """Save checkpoint so the polling eval process picks it up. Non-blocking."""
        ckpt_path = checkpoint_dir / f"agent_step{global_step}.pt"
        if not ckpt_path.exists():
            _save_checkpoint(agent, ckpt_path)
        self._pending_step = global_step
        _log(f"[AsyncEval] Triggered eval at step {global_step}")

    def collect_results(self) -> dict | None:
        """Non-blocking: check if eval results JSON exists for any recent step.
        Renames processed files to .done to avoid re-reading."""
        if self._results_dir is None:
            return None
        result_files = sorted(self._results_dir.glob("eval_step*.json"))
        if not result_files:
            return None
        # Read the latest result, mark all as done
        latest = result_files[-1]
        try:
            import json
            with open(latest) as f:
                data = json.load(f)
            # Rename all result files to .done so we don't re-read them
            for rf in result_files:
                try:
                    rf.rename(rf.with_suffix(".done"))
                except Exception:
                    pass
            step = data.get("step", 0)
            succ = data.get("eval/success_rate", 0.0)
            _log(f"[AsyncEval] Got results for step {step}: success={succ:.3f}")
            return data
        except Exception:
            return None

    def is_alive(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def shutdown(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            _log("[AsyncEval] Subprocess terminated")


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_PHASE = "startup"
_PHASE_SINCE = time.time()


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)


def _set_phase(phase: str):
    global _PHASE, _PHASE_SINCE
    _PHASE = phase
    _PHASE_SINCE = time.time()
    _log(f"PHASE -> {phase}")


def _start_phase_heartbeat(interval_sec: int):
    interval_sec = max(1, int(interval_sec))

    def _worker():
        while True:
            elapsed = time.time() - _PHASE_SINCE
            _log(f"heartbeat phase={_PHASE} elapsed={elapsed:.1f}s")
            time.sleep(interval_sec)

    t = threading.Thread(target=_worker, name="phase-heartbeat", daemon=True)
    t.start()


# ── Timing utility (same as original) ──
class TrainingTimer:
    def __init__(self):
        self.times = defaultdict(list)

    @contextmanager
    def time(self, stage_name: str):
        start = time.perf_counter()
        yield
        self.times[stage_name].append(time.perf_counter() - start)

    def get_timing_stats(self) -> dict[str, float]:
        total = sum(sum(t) for t in self.times.values())
        if total == 0:
            return {}
        stats = {}
        for name, tl in self.times.items():
            s = sum(tl)
            stats[f"timing/{name}_percentage"] = (s / total) * 100
            stats[f"timing/{name}_avg_ms"] = (s / len(tl)) * 1000 if tl else 0
            stats[f"timing/{name}_total_s"] = s
        return stats

    def reset(self):
        self.times = defaultdict(list)


# ── Phase probe debugging utilities ──
_PHASE_NAMES = ["approach", "grasp", "lift"]


def _compute_gt_phase_from_debug(debug_info: dict, env_id: int) -> int:
    """Determine ground truth task phase from env debug info.

    Returns 0=approach, 1=grasp, 2=lift.
    Logic mirrors the phase labeling used to train the probe:
      - lift:     cube_height >= 0.005m (cube being lifted with contact)
      - grasp:    finger close to cube (< 5cm) OR has_contact
      - approach: everything else (moving toward cube)
    """
    cube_h = debug_info["cube_height"][env_id].item()
    has_contact = debug_info["has_contact"][env_id].item() > 0.5
    finger_dist = debug_info["finger_cube_dist"][env_id].item()

    if cube_h >= 0.005 and has_contact:
        return 2  # lift
    elif finger_dist < 0.05 or has_contact:
        return 1  # grasp
    else:
        return 0  # approach


def _compute_phase_probe_stats(
    agent,
    obs: dict[str, torch.Tensor],
    debug_info: dict | None,
    num_envs: int,
    device: torch.device,
) -> dict[str, float] | None:
    """Compute phase probe predictions vs GT for all envs.

    Returns a dict of metrics (accuracy, per-class accuracy, distribution)
    or None if phase probe is not active.
    """
    if agent.vlm_projector is None:
        return None
    if "observation.vlm_latent" not in obs:
        return None
    if debug_info is None:
        return None

    vlm_raw = obs["observation.vlm_latent"][:num_envs].detach()  # (N, 2048)
    with torch.no_grad():
        logits = agent.vlm_projector(vlm_raw.to(device))  # (N, 3)
        probs = torch.softmax(logits, dim=-1).cpu()  # (N, 3)
    pred_phases = probs.argmax(dim=-1).numpy()  # (N,)

    gt_phases = np.array([
        _compute_gt_phase_from_debug(debug_info, eid)
        for eid in range(num_envs)
    ])

    correct = (pred_phases == gt_phases)
    overall_acc = correct.mean()

    stats: dict[str, float] = {
        "phase_probe/accuracy": float(overall_acc),
    }

    # Per-class accuracy & distribution
    for cls_id, cls_name in enumerate(_PHASE_NAMES):
        gt_mask = (gt_phases == cls_id)
        pred_mask = (pred_phases == cls_id)
        stats[f"phase_probe/gt_{cls_name}_count"] = float(gt_mask.sum())
        stats[f"phase_probe/pred_{cls_name}_count"] = float(pred_mask.sum())
        if gt_mask.sum() > 0:
            stats[f"phase_probe/acc_{cls_name}"] = float(correct[gt_mask].mean())
        # Mean predicted probability for this class
        stats[f"phase_probe/mean_prob_{cls_name}"] = float(probs[:, cls_id].mean().item())

    # Confidence: mean max probability
    stats["phase_probe/mean_confidence"] = float(probs.max(dim=-1).values.mean().item())

    return stats


def _format_phase_probe_summary(phase_stats: dict[str, float], num_envs: int) -> str:
    """Format a one-line console summary of phase probe accuracy."""
    acc = phase_stats.get("phase_probe/accuracy", 0.0)
    conf = phase_stats.get("phase_probe/mean_confidence", 0.0)
    parts = [f"phase_acc={acc*100:.0f}%"]
    for cls_name in _PHASE_NAMES:
        gt_n = int(phase_stats.get(f"phase_probe/gt_{cls_name}_count", 0))
        cls_acc = phase_stats.get(f"phase_probe/acc_{cls_name}", -1)
        if gt_n > 0:
            parts.append(f"{cls_name[0].upper()}:{gt_n}({cls_acc*100:.0f}%)")
        else:
            parts.append(f"{cls_name[0].upper()}:0")
    parts.append(f"conf={conf:.2f}")
    return " ".join(parts)


# ── Replay buffer helper ──
_DEBUG_TRANSITION_COUNT = 0

def _add_transitions(
    obs, next_obs, actions, reward, done, info,
    device, image_keys, lowdim_keys, num_envs, online_rb,
):
    global _DEBUG_TRANSITION_COUNT
    obs_keys = set(image_keys) | set(lowdim_keys)
    for i in range(num_envs):
        if done[i] and "final_obs" in info and info["final_obs"] is not None:
            try:
                next_obs_i = {k: torch.as_tensor(v, device=device) for k, v in info["final_obs"][i].items()}
            except Exception:
                next_obs_i = {k: v[i] for k, v in next_obs.items()}
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}

        curr_obs_i = {k: v[i] for k, v in obs.items() if k in obs_keys}
        next_obs_i = {k: v for k, v in next_obs_i.items() if k in obs_keys}
        to_uint8(curr_obs_i, image_keys)
        to_uint8(next_obs_i, image_keys)

        # Depth dtype/range validation (first 5 transitions only)
        if not hasattr(_add_transitions, '_depth_debug_n'):
            _add_transitions._depth_debug_n = 0
        if _add_transitions._depth_debug_n < 5:
            for dk in image_keys:
                if dk.startswith("observation.depth") and dk in curr_obs_i:
                    v = curr_obs_i[dk]
                    _log(f"[DEPTH BUFFER DEBUG] key={dk} dtype={v.dtype} shape={v.shape} "
                         f"range=[{v.min().item():.4f}, {v.max().item():.4f}] "
                         f"is_float32={v.dtype == torch.float32}")
            _add_transitions._depth_debug_n += 1

        # Debug logging for first few transitions
        if _DEBUG_TRANSITION_COUNT < 20:
            state = curr_obs_i.get("observation.state", None)
            base_act = curr_obs_i.get("observation.base_action", None)
            act = actions[i]
            r = reward[i]
            d = done[i]
            _log(f"[DEBUG buffer] transition #{_DEBUG_TRANSITION_COUNT} env={i}:")
            if state is not None:
                s = state.detach().cpu().numpy() if hasattr(state, 'detach') else state
                _log(f"  state({len(s)}D): {s[:5]}... (eef_pos+quat+grip)")
            if base_act is not None:
                ba = base_act.detach().cpu().numpy() if hasattr(base_act, 'detach') else base_act
                _log(f"  base_action(7D): {ba}")
            obj_st = curr_obs_i.get("observation.object_state", None)
            if obj_st is not None:
                os_v = obj_st.detach().cpu().numpy() if hasattr(obj_st, 'detach') else obj_st
                _log(f"  object_state({len(os_v)}D): {os_v}")
            a = act.detach().cpu().numpy() if hasattr(act, 'detach') else act
            _log(f"  action(stored): {a}")
            rv = r.item() if hasattr(r, 'item') else r
            dv = d.item() if hasattr(d, 'item') else d
            _log(f"  reward={rv:.4f} done={dv}")
            # Check for zeros bug
            if hasattr(act, 'abs'):
                if act.abs().sum().item() < 1e-8:
                    _log(f"  ⚠️ WARNING: stored action is ALL ZEROS!")

        td = TensorDict({
            "obs": TensorDict(curr_obs_i, batch_size=[]),
            "next": TensorDict({
                "obs": TensorDict(next_obs_i, batch_size=[]),
                "done": done[i],
                "reward": reward[i],
            }, batch_size=[]),
            "action": actions[i],
            "_priority": torch.tensor(10.0, dtype=torch.float32, device=device),
        }, batch_size=[]).unsqueeze(0)
        online_rb.add(td)
        _DEBUG_TRANSITION_COUNT += 1


def _default_iface_script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "workspace" / "franka_cogact_motion_generation_abs_window_gr00t_training_only_iface.py"


def _default_isaaclab_sh() -> Path:
    return Path("/workspace/isaaclab/isaaclab.sh")


def _default_iface_step_server_script_path() -> Path:
    return Path(__file__).resolve().parent / "iface_residual_step_server.py"


def _effective_task_description(task_description: str, language_override: str | None) -> str:
    if language_override is not None:
        text = str(language_override).strip()
        if text:
            return text
    return str(task_description)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _to_torch_tree(x, device: torch.device):
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8:
            return torch.as_tensor(x, device=device, dtype=torch.uint8)
        if x.dtype == np.bool_:
            return torch.as_tensor(x, device=device, dtype=torch.bool)
        return torch.as_tensor(x, device=device, dtype=torch.float32)
    if isinstance(x, dict):
        return {k: _to_torch_tree(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_torch_tree(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_torch_tree(v, device) for v in x)
    if isinstance(x, (bool, np.bool_)):
        return torch.as_tensor(x, device=device, dtype=torch.bool)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return torch.as_tensor(x, device=device, dtype=torch.float32)
    return x


class IfaceResidualRemoteEnv:
    def __init__(self, *, cfg, args_cli, device: torch.device):
        self._device = device
        self._conn = None
        self._proc = None

        port = _find_free_port()
        auth = secrets.token_hex(16)
        isaaclab_sh_path = _default_isaaclab_sh()
        server_script = _default_iface_step_server_script_path()
        gcfg = cfg.groot_policy
        ecfg = cfg.isaaclab_env

        cmd = [
            str(isaaclab_sh_path),
            "-p",
            str(server_script),
            "--headless",
            "--enable_cameras",
            "--num_envs",
            str(ecfg.num_envs),
            "--device",
            str(ecfg.device),
            "--task",
            str(ecfg.task),
            "--image_h",
            str(ecfg.image_size_h),
            "--image_w",
            str(ecfg.image_size_w),
            "--csv_base_dir",
            str(ecfg.csv_base_dir),
            "--csv_init_row_index",
            str(ecfg.csv_init_row_index),
            "--max_episode_steps",
            str(ecfg.max_episode_steps),
            "--groot_model_path",
            str(gcfg.model_path),
            "--groot_host",
            str(gcfg.host),
            "--groot_port",
            str(gcfg.port),
            "--groot_embodiment_tag",
            str(gcfg.embodiment_tag),
            "--task_description",
            str(gcfg.task_description),
            "--ipc_host",
            "127.0.0.1",
            "--ipc_port",
            str(port),
            "--ipc_auth",
            auth,
        ]

        if gcfg.language_override is not None:
            cmd.extend(["--language_override", str(gcfg.language_override)])
        if gcfg.policy_device:
            cmd.extend(["--groot_policy_device", str(gcfg.policy_device)])
        if bool(gcfg.strict):
            cmd.append("--groot_policy_strict")

        self._proc = subprocess.Popen(cmd)

        deadline = time.time() + 180.0
        last_err = None
        while time.time() < deadline:
            try:
                self._conn = Client(("127.0.0.1", port), authkey=auth.encode("utf-8"))
                break
            except Exception as exc:
                last_err = exc
                time.sleep(0.5)

        if self._conn is None:
            raise RuntimeError(f"failed to connect iface step server: {last_err}")

        obs, _ = self.reset()
        self.num_envs = int(obs["observation.state"].shape[0])
        self.action_dim = 7
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_envs, self.action_dim),
            dtype=np.float32,
        )
        obs_spaces = {}
        for k, v in obs.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.dtype == torch.uint8:
                obs_spaces[k] = gym.spaces.Box(low=0, high=255, shape=tuple(v.shape), dtype=np.uint8)
            else:
                obs_spaces[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=tuple(v.shape), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _rpc(self, payload: dict):
        if self._conn is None:
            raise RuntimeError("iface connection is not initialized")
        self._conn.send(payload)
        rep = self._conn.recv()
        if not rep.get("ok", False):
            raise RuntimeError(f"iface server error: {rep.get('error', 'unknown')}")
        return rep

    def reset(self, **kwargs):
        rep = self._rpc({"cmd": "reset"})
        obs = _to_torch_tree(rep["obs"], self._device)
        info = _to_torch_tree(rep.get("info", {}), self._device)
        return obs, (info if isinstance(info, dict) else {})

    def step(self, actions: torch.Tensor):
        arr = actions.detach().cpu().numpy().astype(np.float32)
        rep = self._rpc({"cmd": "step", "action": arr})
        obs = _to_torch_tree(rep["obs"], self._device)
        reward = _to_torch_tree(rep["reward"], self._device)
        terminated = _to_torch_tree(rep["terminated"], self._device)
        truncated = _to_torch_tree(rep["truncated"], self._device)
        info = _to_torch_tree(rep.get("info", {}), self._device)
        return obs, reward, terminated, truncated, (info if isinstance(info, dict) else {})

    def close(self):
        try:
            if self._conn is not None:
                try:
                    self._rpc({"cmd": "close"})
                except Exception:
                    pass
                self._conn.close()
        finally:
            self._conn = None
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=10)
                except Exception:
                    self._proc.kill()
                self._proc = None


def _run_iface_online_rollout_ipc(
    *,
    csv_base_dir: str,
    groot_model_path: str,
    max_steps: int,
    policy_device: str | None,
    embodiment_tag: str,
    task_description: str,
    language_override: str | None,
    policy_strict: bool,
    headless: bool,
    shm_size_bytes: int,
    timeout_s: int,
    camera_warmup_steps: int,
    post_reset_settle_steps: int,
    replay_actions_npz: str | None = None,
) -> dict[str, np.ndarray]:
    isaaclab_sh_path = _default_isaaclab_sh()
    iface_script = _default_iface_script_path()
    shm_size_bytes = max(1 << 20, int(shm_size_bytes))
    shm = shared_memory.SharedMemory(create=True, size=shm_size_bytes)

    try:
        cmd = [
            str(isaaclab_sh_path),
            "-p",
            str(iface_script),
            "--enable_cameras",
            "--num_envs",
            "1",
            "--model_path",
            str(Path(groot_model_path).expanduser().resolve()),
            "--embodiment_tag",
            str(embodiment_tag),
            "--task_description",
            str(task_description),
            "--csv_dir",
            str(Path(csv_base_dir).expanduser().resolve()),
            "--max_steps",
            str(max_steps),
            "--max_attempts_per_episode",
            "1",
            "--rl_disable_retries",
            "--camera_warmup_steps",
            str(max(0, int(camera_warmup_steps))),
            "--post_reset_settle_steps",
            str(max(0, int(post_reset_settle_steps))),
            "--online_buffer_shm_name",
            shm.name,
            "--online_buffer_shm_size",
            str(shm_size_bytes),
        ]

        if language_override is not None:
            cmd.extend(["--language_override", str(language_override)])
        if policy_device:
            cmd.extend(["--policy_device", str(policy_device)])
        if bool(policy_strict):
            cmd.append("--policy_strict")
        if replay_actions_npz:
            cmd.extend(["--replay_actions_npz", str(Path(replay_actions_npz).expanduser().resolve())])
        if headless:
            cmd.append("--headless")

        timeout = None if int(timeout_s) <= 0 else int(timeout_s)
        try:
            proc = subprocess.run(cmd, check=False, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"iface warmup rollout timed out after {int(timeout_s)}s "
                f"(max_steps={max_steps}, shm={shm_size_bytes})"
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(f"iface warmup rollout failed with code {proc.returncode}")

        header_size = 8
        payload_len = int.from_bytes(bytes(shm.buf[:header_size]), byteorder="little", signed=False)
        if payload_len <= 0:
            raise RuntimeError("iface warmup rollout shared-memory payload is empty")
        if payload_len + header_size > shm_size_bytes:
            raise RuntimeError(
                f"iface warmup shared-memory payload overflow: payload={payload_len}, shm={shm_size_bytes}"
            )

        payload = bytes(shm.buf[header_size:header_size + payload_len])
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            return {k: np.asarray(data[k]) for k in data.files}
    finally:
        try:
            shm.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def _add_iface_npz_to_online_rb(
    *,
    npz_data: dict[str, np.ndarray],
    image_keys: list[str],
    online_rb,
    device,
    expected_state_dim: int,
    expected_base_action_dim: int,
    expected_action_dim: int,
) -> int:
    data = npz_data
    required_payload_keys = [
        "state",
        "base_action",
        "action",
        "reward",
        "done",
        "front_image",
        "back_image",
        "wrist_image",
    ]
    missing = [k for k in required_payload_keys if k not in data]
    if missing:
        raise ValueError(f"iface online payload missing required keys: {missing}")

    state = np.asarray(data["state"], dtype=np.float32)
    base_action = np.asarray(data["base_action"], dtype=np.float32)
    action = np.asarray(data["action"], dtype=np.float32)
    reward = np.asarray(data["reward"], dtype=np.float32)
    done = np.asarray(data["done"], dtype=np.bool_)
    front_image = np.asarray(data["front_image"], dtype=np.uint8)
    back_image = np.asarray(data["back_image"], dtype=np.uint8)
    wrist_image = np.asarray(data["wrist_image"], dtype=np.uint8)

    for name, arr in (("front_image", front_image), ("back_image", back_image), ("wrist_image", wrist_image)):
        if arr.ndim != 4 or arr.shape[1] != 3:
            raise ValueError(f"{name} must be (T,3,H,W), got shape={arr.shape}")

    lengths = {
        "state": len(state),
        "base_action": len(base_action),
        "action": len(action),
        "reward": len(reward),
        "done": len(done),
        "front_image": len(front_image),
        "back_image": len(back_image),
        "wrist_image": len(wrist_image),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"iface online payload length mismatch: {lengths}")
    T = len(state)
    if T < 2:
        return 0

    def _fit_last_dim(arr_1d: np.ndarray, target_dim: int) -> np.ndarray:
        vec = np.asarray(arr_1d, dtype=np.float32).reshape(-1)
        if vec.shape[0] == target_dim:
            return vec
        if vec.shape[0] > target_dim:
            return vec[:target_dim]
        out = np.zeros((target_dim,), dtype=np.float32)
        out[: vec.shape[0]] = vec
        return out

    added = 0
    for t in range(T - 1):
        curr_obs = {
            "observation.state": torch.tensor(_fit_last_dim(state[t], expected_state_dim), dtype=torch.float32, device=device),
            "observation.base_action": torch.tensor(_fit_last_dim(base_action[t], expected_base_action_dim), dtype=torch.float32, device=device),
        }
        next_obs = {
            "observation.state": torch.tensor(_fit_last_dim(state[t + 1], expected_state_dim), dtype=torch.float32, device=device),
            "observation.base_action": torch.tensor(_fit_last_dim(base_action[t + 1], expected_base_action_dim), dtype=torch.float32, device=device),
        }

        front_curr = torch.tensor(front_image[t], dtype=torch.uint8, device=device)
        front_next = torch.tensor(front_image[t + 1], dtype=torch.uint8, device=device)
        back_curr = torch.tensor(back_image[t], dtype=torch.uint8, device=device)
        back_next = torch.tensor(back_image[t + 1], dtype=torch.uint8, device=device)
        wrist_curr = torch.tensor(wrist_image[t], dtype=torch.uint8, device=device)
        wrist_next = torch.tensor(wrist_image[t + 1], dtype=torch.uint8, device=device)

        for image_key in image_keys:
            if image_key == "observation.images.front":
                curr_obs[image_key] = front_curr
                next_obs[image_key] = front_next
            elif image_key == "observation.images.back":
                curr_obs[image_key] = back_curr
                next_obs[image_key] = back_next
            elif image_key == "observation.images.wrist":
                curr_obs[image_key] = wrist_curr
                next_obs[image_key] = wrist_next
            else:
                raise KeyError(f"Unsupported image key for iface payload mapping: {image_key}")

        td = TensorDict(
            {
                "obs": TensorDict(curr_obs, batch_size=[]),
                "next": TensorDict(
                    {
                        "obs": TensorDict(next_obs, batch_size=[]),
                        "done": torch.tensor(bool(done[t + 1]), dtype=torch.bool, device=device),
                        "reward": torch.tensor(float(reward[t + 1]), dtype=torch.float32, device=device),
                    },
                    batch_size=[],
                ),
                "action": torch.tensor(_fit_last_dim(action[t], expected_action_dim), dtype=torch.float32, device=device),
                "_priority": torch.tensor(10.0, dtype=torch.float32, device=device),
            },
            batch_size=[],
        ).unsqueeze(0)
        online_rb.add(td)
        added += 1

    return added


def _iface_npz_step_to_tensors(
    *,
    npz_data: dict[str, np.ndarray],
    image_keys: list[str],
    device,
    expected_state_dim: int,
    expected_base_action_dim: int,
    expected_action_dim: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    data = npz_data
    required_payload_keys = [
        "state",
        "base_action",
        "action",
        "reward",
        "done",
        "front_image",
        "back_image",
        "wrist_image",
    ]
    missing = [k for k in required_payload_keys if k not in data]
    if missing:
        raise ValueError(f"iface online payload missing required keys: {missing}")

    state = np.asarray(data["state"], dtype=np.float32)
    base_action = np.asarray(data["base_action"], dtype=np.float32)
    action = np.asarray(data["action"], dtype=np.float32)
    reward = np.asarray(data["reward"], dtype=np.float32)
    done = np.asarray(data["done"], dtype=np.bool_)
    front_image = np.asarray(data["front_image"], dtype=np.uint8)
    back_image = np.asarray(data["back_image"], dtype=np.uint8)
    wrist_image = np.asarray(data["wrist_image"], dtype=np.uint8)

    T = min(
        len(state), len(base_action), len(action), len(reward), len(done),
        len(front_image), len(back_image), len(wrist_image),
    )
    if T < 2:
        raise RuntimeError("iface rollout payload requires at least 2 frames for one transition")

    def _fit_last_dim(arr_1d: np.ndarray, target_dim: int) -> np.ndarray:
        vec = np.asarray(arr_1d, dtype=np.float32).reshape(-1)
        if vec.shape[0] == target_dim:
            return vec
        if vec.shape[0] > target_dim:
            return vec[:target_dim]
        out = np.zeros((target_dim,), dtype=np.float32)
        out[: vec.shape[0]] = vec
        return out

    curr_obs = {
        "observation.state": torch.tensor(_fit_last_dim(state[0], expected_state_dim), dtype=torch.float32, device=device).unsqueeze(0),
        "observation.base_action": torch.tensor(_fit_last_dim(base_action[0], expected_base_action_dim), dtype=torch.float32, device=device).unsqueeze(0),
    }
    next_obs = {
        "observation.state": torch.tensor(_fit_last_dim(state[1], expected_state_dim), dtype=torch.float32, device=device).unsqueeze(0),
        "observation.base_action": torch.tensor(_fit_last_dim(base_action[1], expected_base_action_dim), dtype=torch.float32, device=device).unsqueeze(0),
    }

    cam_curr = {
        "observation.images.front": torch.tensor(front_image[0], dtype=torch.uint8, device=device).unsqueeze(0),
        "observation.images.back": torch.tensor(back_image[0], dtype=torch.uint8, device=device).unsqueeze(0),
        "observation.images.wrist": torch.tensor(wrist_image[0], dtype=torch.uint8, device=device).unsqueeze(0),
    }
    cam_next = {
        "observation.images.front": torch.tensor(front_image[1], dtype=torch.uint8, device=device).unsqueeze(0),
        "observation.images.back": torch.tensor(back_image[1], dtype=torch.uint8, device=device).unsqueeze(0),
        "observation.images.wrist": torch.tensor(wrist_image[1], dtype=torch.uint8, device=device).unsqueeze(0),
    }
    for k in image_keys:
        if k in cam_curr:
            curr_obs[k] = cam_curr[k]
            next_obs[k] = cam_next[k]

    action_t = torch.tensor(_fit_last_dim(action[0], expected_action_dim), dtype=torch.float32, device=device).unsqueeze(0)
    reward_t = torch.tensor([float(reward[1])], dtype=torch.float32, device=device)
    done_t = torch.tensor([bool(done[1])], dtype=torch.bool, device=device)
    return curr_obs, next_obs, action_t, reward_t, done_t


def _replace_front_image_with_iface_render(
    *,
    rollout_data: dict[str, np.ndarray],
    csv_base_dir: str,
    groot_model_path: str,
    policy_device: str | None,
    embodiment_tag: str,
    task_description: str,
    language_override: str | None,
    policy_strict: bool,
    headless: bool,
    shm_size_bytes: int,
    timeout_s: int,
    camera_warmup_steps: int,
    post_reset_settle_steps: int,
) -> dict[str, np.ndarray]:
    actions = np.asarray(rollout_data.get("action", []), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) <= 0:
        return rollout_data

    with tempfile.TemporaryDirectory(prefix="resfit_iface_replay_") as tmp_dir:
        replay_npz = Path(tmp_dir) / "replay_actions.npz"
        np.savez_compressed(str(replay_npz), action=actions)

        iface_data = _run_iface_online_rollout_ipc(
            csv_base_dir=csv_base_dir,
            groot_model_path=groot_model_path,
            max_steps=len(actions),
            policy_device=policy_device,
            embodiment_tag=embodiment_tag,
            task_description=task_description,
            language_override=language_override,
            policy_strict=policy_strict,
            headless=headless,
            shm_size_bytes=shm_size_bytes,
            timeout_s=timeout_s,
            camera_warmup_steps=camera_warmup_steps,
            post_reset_settle_steps=post_reset_settle_steps,
            replay_actions_npz=str(replay_npz),
        )

    out = dict(rollout_data)
    for key in ("front_image", "back_image", "wrist_image"):
        iface_img = np.asarray(iface_data.get(key, []), dtype=np.uint8)
        if iface_img.ndim < 4 or len(iface_img) <= 0:
            continue
        in_img = np.asarray(out.get(key, []), dtype=np.uint8)
        if in_img.ndim < 4 or len(in_img) <= 0:
            out[key] = iface_img
            continue
        T = min(len(in_img), len(iface_img))
        merged = in_img.copy()
        merged[:T] = iface_img[:T]
        out[key] = merged
    return out


def _collect_online_rollout_inprocess(
    *,
    env,
    max_steps: int,
    image_keys: list[str],
    device,
) -> dict[str, np.ndarray]:
    max_steps = max(1, int(max_steps))
    required_view_keys = (
        "observation.images.front",
        "observation.images.back",
        "observation.images.wrist",
    )
    missing_required = [k for k in required_view_keys if k not in image_keys]
    if missing_required:
        raise ValueError(f"image_keys missing required camera views: {missing_required}")

    obs, _ = env.reset()
    action_dim = int(env.action_space.shape[-1])
    num_envs = int(getattr(env, "num_envs", 1))

    buffer_timestep: list[int] = []
    buffer_state: list[np.ndarray] = []
    buffer_base_action: list[np.ndarray] = []
    buffer_action: list[np.ndarray] = []
    buffer_reward: list[float] = []
    buffer_done: list[bool] = []
    buffer_front_image: list[np.ndarray] = []
    buffer_back_image: list[np.ndarray] = []
    buffer_wrist_image: list[np.ndarray] = []

    for t in range(max_steps):
        state_t = np.asarray(obs["observation.state"][0].detach().cpu(), dtype=np.float32)
        base_action_t = np.asarray(obs["observation.base_action"][0].detach().cpu(), dtype=np.float32)
        image_now: dict[str, np.ndarray] = {}
        for key in required_view_keys:
            if key not in obs:
                raise KeyError(f"missing required camera observation key: {key}")
            img_t = np.asarray(obs[key][0].detach().cpu())
            if img_t.ndim != 3:
                raise ValueError(f"invalid camera tensor shape for {key}: {img_t.shape}")
            if img_t.dtype != np.uint8:
                img_t = np.clip(img_t, 0, 255).astype(np.uint8)
            image_now[key] = img_t


        residual = torch.zeros((num_envs, action_dim), device=device, dtype=torch.float32)
        next_obs, reward, terminated, truncated, info = env.step(residual)
        done = (terminated | truncated)

        combined = info.get("scaled_action", None)
        if combined is None:
            combined_t = base_action_t
        else:
            combined_t = np.asarray(combined[0].detach().cpu(), dtype=np.float32)

        reward_t = float(reward[0].detach().cpu().item()) if isinstance(reward, torch.Tensor) else float(reward[0])
        done_t = bool(done[0].detach().cpu().item()) if isinstance(done, torch.Tensor) else bool(done[0])

        buffer_timestep.append(int(t))
        buffer_state.append(state_t)
        buffer_base_action.append(base_action_t)
        buffer_action.append(combined_t)
        buffer_reward.append(reward_t)
        buffer_done.append(done_t)
        buffer_front_image.append(image_now["observation.images.front"])
        buffer_back_image.append(image_now["observation.images.back"])
        buffer_wrist_image.append(image_now["observation.images.wrist"])

        obs = next_obs

    if buffer_done:
        buffer_done[-1] = True

    return {
        "timestep": np.asarray(buffer_timestep, dtype=np.int32),
        "state": np.asarray(buffer_state, dtype=np.float32),
        "base_action": np.asarray(buffer_base_action, dtype=np.float32),
        "action": np.asarray(buffer_action, dtype=np.float32),
        "reward": np.asarray(buffer_reward, dtype=np.float32),
        "done": np.asarray(buffer_done, dtype=np.bool_),
        "front_image": np.asarray(buffer_front_image, dtype=np.uint8),
        "back_image": np.asarray(buffer_back_image, dtype=np.uint8),
        "wrist_image": np.asarray(buffer_wrist_image, dtype=np.uint8),
    }


# ── Main ──
def main(cfg: ResidualTD3IsaacLabConfig):
    heartbeat_interval_sec = max(1, int(getattr(cfg, "heartbeat_interval_sec", 20)))
    stack_dump_interval_sec = int(getattr(cfg, "stack_dump_interval_sec", 180))

    # NOTE: We intentionally do NOT call faulthandler.dump_traceback_later()
    # because Isaac Sim's async engine conflicts with it, causing 3-min timeouts.
    # Periodic stack dumps are handled by the heartbeat thread instead.
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass

    _start_phase_heartbeat(heartbeat_interval_sec)
    _set_phase("setup")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Seed
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    _log(f"Seed: {cfg.seed}")

    ecfg = cfg.isaaclab_env
    gcfg = cfg.groot_policy
    acfg = cfg.algo
    if not gcfg.model_path:
        raise ValueError("iface-only residual mode requires --groot_model_path")

    # ── Create env in-process (no IPC — same code path as online iface script) ──
    _set_phase("create_env")
    eval_num_envs = int(getattr(cfg, "eval_num_envs", 10))
    _inline_eval_disabled = getattr(args_cli, "disable_eval", False) or str(os.environ.get("RESFIT_DISABLE_EVAL", "0")).lower() in {"1", "true", "yes", "on"}
    if _inline_eval_disabled:
        scene_num_envs = int(ecfg.num_envs)  # train only — no eval envs in scene
        _log(f"Creating IfaceEnvWrapper: train_envs={ecfg.num_envs} (eval disabled, use async eval process)")
    else:
        scene_num_envs = int(ecfg.num_envs) + eval_num_envs
        _log(f"Creating IfaceEnvWrapper: train_envs={ecfg.num_envs} + eval_envs={eval_num_envs} = {scene_num_envs} total")

    from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=cfg.device)
    _sim = sim_utils.SimulationContext(sim_cfg)

    policy_device = gcfg.policy_device or cfg.device
    env = IfaceEnvWrapper(
        sim=_sim,
        csv_dir=str(ecfg.csv_base_dir),
        groot_model_path=str(gcfg.model_path),
        embodiment_tag=str(gcfg.embodiment_tag),
        policy_device=str(policy_device),
        policy_strict=bool(gcfg.strict),
        task_description=str(gcfg.task_description),
        language_override=gcfg.language_override,
        max_episode_steps=int(ecfg.max_episode_steps),
        num_envs=scene_num_envs,
        success_threshold=float(getattr(ecfg, 'success_threshold', 0.005)),
        cube_perturb_range=float(getattr(ecfg, 'cube_perturb_range', 0.0)),
        cube_perturb_table_path=getattr(ecfg, 'cube_perturb_table_path', None),
        random_cube_perturb=bool(getattr(args_cli, 'random_cube_perturb', False)),
        reward_type=str(getattr(ecfg, 'reward_type', 'dense_clipped')),
        use_depth=bool(getattr(args_cli, 'use_depth', False)) or bool(getattr(args_cli, 'asymmetric_critic', False)),
        depth_norm_path=getattr(args_cli, 'depth_norm_path', None),
        use_state=bool(getattr(args_cli, 'use_state', False)) or bool(getattr(args_cli, 'asymmetric_critic', False)),
        object_state_mode=str(getattr(args_cli, 'object_state_mode', 'raw')),
        contact_binary=bool(getattr(args_cli, 'contact_binary', False)),
    )
    eval_env = env  # same sim context, switch active_env_ids for eval
    _set_phase("env_ready")
    _log(f"IfaceEnvWrapper created: {scene_num_envs} envs in scene.")

    # ── Dimensions ──
    # obs_space shapes have batch dim = scene_num_envs but training uses 1
    image_keys = list(cfg.rl_camera)
    # If depth mode, replace RGB cameras with depth cameras for residual actor
    _use_depth = bool(getattr(args_cli, 'use_depth', False))
    _use_state = bool(getattr(args_cli, 'use_state', False))
    _asymmetric = bool(getattr(args_cli, 'asymmetric_critic', False))
    if _asymmetric:
        # Asymmetric: critic uses depth, actor uses state-only
        # Force both depth and state on
        _use_depth = True
        _use_state = True
        image_keys = ["observation.depth.front", "observation.depth.wrist"]
        _log(f"ASYMMETRIC mode: critic uses depth {image_keys}, actor is state-only")
    elif _use_state:
        image_keys = []  # state-only: no images
        _log(f"State-only mode: no image keys (encoder-free)")
    elif _use_depth:
        image_keys = ["observation.depth.front", "observation.depth.wrist"]
        _log(f"Depth mode: RL cameras = {image_keys}")
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    base_action_dim = env.observation_space["observation.base_action"].shape[1]
    if image_keys:
        img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    else:
        img_c, img_h, img_w = 3, 84, 84  # dummy, not used in state-only mode
    action_dim = env.action_space.shape[1]
    lowdim_keys = ["observation.state", "observation.base_action"]
    # Check if VLM latent is available from env AND explicitly enabled
    _use_vlm = bool(getattr(args_cli, 'use_vlm', False))
    if _use_state or _asymmetric:
        _use_vlm = False  # state-only / asymmetric mode: no VLM
    vlm_latent_dim = 0
    if _use_vlm and "observation.vlm_latent" in env.observation_space.spaces:
        vlm_latent_dim = env.observation_space["observation.vlm_latent"].shape[1]
        lowdim_keys.append("observation.vlm_latent")
        _log(f"VLM latent enabled (--use_vlm): {vlm_latent_dim}D")
    elif "observation.vlm_latent" in env.observation_space.spaces:
        _log(f"VLM latent available but DISABLED (use --use_vlm to enable)")
    else:
        _log(f"VLM latent not available in env")
    # Object state (cube 6D pose) — adds to lowdim
    _use_state_for_obj = bool(getattr(args_cli, 'use_state', False)) or _asymmetric
    object_state_dim = 0
    if _use_state_for_obj and "observation.object_state" in env.observation_space.spaces:
        object_state_dim = env.observation_space["observation.object_state"].shape[1]
        lowdim_keys.append("observation.object_state")
        _log(f"Object state enabled: {object_state_dim}D")
    elif _use_state_for_obj:
        _log(f"WARNING: --use_state requested but observation.object_state not in env")
    # Warmup uses 1 env; main training loop will expand to all train envs
    num_envs = 1  # warmup with single env
    train_env_ids = [0]  # warmup: env 0 only

    _log(f"lowdim_dim={lowdim_dim}, img=({img_c},{img_h},{img_w}), action_dim={action_dim}")
    _log(f"Image expected: (C,H,W)=({img_c},{img_h},{img_w}) — should be (3,84,84)")
    _log(f"Scene: {scene_num_envs} envs total, warmup with 1 env, training with {int(ecfg.num_envs)} envs")
    if not _inline_eval_disabled:
        eval_env_ids = list(range(int(ecfg.num_envs), scene_num_envs))
    else:
        eval_env_ids = []
    _log(f"Warmup env IDs: {train_env_ids}, Eval env IDs: {eval_env_ids}")

    # ── Config dump (matches original resfit) ──
    _log(f"Config:\n{pprint.pformat(asdict(cfg) if is_dataclass(cfg) else {})}")

    # ── QAgent (residual actor + critic) ──
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
        vlm_latent_dim=vlm_latent_dim,
        object_state_dim=object_state_dim,
        asymmetric_critic=_asymmetric,
    )

    # Load frozen phase probe (must be before resume_checkpoint to set correct weights)
    if args_cli.phase_probe_path and agent.vlm_projector is not None:
        agent.load_phase_probe(args_cli.phase_probe_path)

    # Resume from checkpoint (weights only — fresh optimizers for new training)
    if args_cli.resume_checkpoint:
        _log(f"Resuming from checkpoint: {args_cli.resume_checkpoint}")
        ckpt = torch.load(args_cli.resume_checkpoint, map_location=cfg.device)
        agent.load_checkpoint_compat(ckpt)  # loads encoders, actor, critic, targets, vlm_projector
        _log("  Weights loaded (fresh optimizers)")

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_isaaclab_residual_td3_seed{cfg.seed}"
    if getattr(cfg.wandb, "name", None):
        run_name = f"{cfg.wandb.name}__{run_name}"

    # ── Replay buffers ──
    online_batch_size = int(acfg.batch_size * (1 - acfg.offline_fraction))
    offline_batch_size = int(acfg.batch_size * acfg.offline_fraction)

    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=acfg.buffer_size, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(n_steps=acfg.n_step, gamma=acfg.gamma),
        pin_memory=True,
        prefetch=4,
        batch_size=max(online_batch_size, 1),
    )

    def _make_offline_rb(max_size: int):
        return TensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(max_size=max(1, int(max_size)), device="cpu"),
            alpha=0.0, beta=0.0, eps=1e-6,
            priority_key="_priority",
            transform=MultiStepTransform(n_steps=acfg.n_step, gamma=acfg.gamma),
            pin_memory=True,
            prefetch=4,
            batch_size=max(offline_batch_size, 1),
        )

    offline_rb = _make_offline_rb(acfg.buffer_size)

    # ══════════════════════════════════════════════════════════════════
    # Buffer cache setup (shared at repo root, hash-based)
    # Must be defined before offline + online buffer caching sections.
    # ══════════════════════════════════════════════════════════════════
    import hashlib, json as _json

    def _buffer_cache_key(ecfg, acfg, *, extra: dict | None = None):
        """Build a deterministic hash of buffer-relevant config + model dims.

        Includes environment config, algorithm config, AND model architecture
        dimensions so that changing input dims (e.g. adding VLM, changing
        state dim) automatically invalidates the cache.
        """
        key_dict = {
            # ── Environment ──
            "csv_base_dir": str(ecfg.csv_base_dir),
            "success_threshold": float(getattr(ecfg, 'success_threshold', 0.005)),
            "cube_perturb_table_path": str(getattr(ecfg, 'cube_perturb_table_path', '')),
            "cube_perturb_range": float(getattr(ecfg, 'cube_perturb_range', 0.0)),
            "num_envs": int(ecfg.num_envs),
            "max_episode_steps": int(ecfg.max_episode_steps),
            "reward_type": str(getattr(ecfg, 'reward_type', 'dense_clipped')),
            # ── Algorithm ──
            "random_action_noise_scale": float(acfg.random_action_noise_scale),
            "n_step": int(acfg.n_step),
            "gamma": float(acfg.gamma),
            # ── Model architecture (input dims) ──
            "lowdim_dim": int(lowdim_dim),
            "vlm_latent_dim": int(vlm_latent_dim),
            "action_dim": int(action_dim),
            "image_size": [int(img_h), int(img_w)],
            "image_keys": sorted(image_keys),
            "phase_probe_path": str(getattr(args_cli, 'phase_probe_path', '') or ''),
            # ── Mode flags ──
            "use_depth": bool(getattr(args_cli, 'use_depth', False)),
            "use_vlm": bool(getattr(args_cli, 'use_vlm', False)),
            "use_state": bool(getattr(args_cli, 'use_state', False)),
            "object_state_dim": int(object_state_dim),
            "object_state_mode": str(getattr(args_cli, 'object_state_mode', 'raw')),
            "contact_binary": bool(getattr(args_cli, 'contact_binary', False)),
            "asymmetric_critic": bool(getattr(args_cli, 'asymmetric_critic', False)),
        }
        # Include perturbation table content if exists
        _pt = getattr(ecfg, 'cube_perturb_table_path', None)
        if _pt and Path(_pt).exists():
            key_dict["perturb_table_content"] = Path(_pt).read_text()
        # Merge any extra keys (e.g. offline-specific fields)
        if extra:
            key_dict.update(extra)
        key_str = _json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    # Shared cache root at repo level (sibling of outputs/)
    _resrl_root = Path(__file__).resolve().parents[3]  # residual-offpolicy-rl/
    cache_root = _resrl_root / "buffer_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    _log(f"Shared buffer cache dir: {cache_root}")

    # ── Populate offline buffer from CSV episodes (with shared cache) ──
    if cfg.offline_data is not None and acfg.offline_fraction > 0.0:
        offline_root = Path(cfg.offline_data.csv_data_dir)
        is_lerobot_local = (offline_root / "data" / "chunk-000").exists() and (offline_root / "videos" / "chunk-000").exists()

        # Validate offline reward formula matches online
        if is_lerobot_local:
            success_thresh = float(getattr(ecfg, 'success_threshold', 0.005))
            _log(f"Validating offline reward formula (success_threshold={success_thresh})...")
            validate_offline_reward_formula(offline_root, success_threshold=success_thresh)
            _log("Offline reward formula validated ✓")

        def _populate_current_offline_rb():
            if is_lerobot_local:
                _log(f"Populating offline buffer from LeRobot local dataset: {offline_root}")
                return populate_offline_buffer_from_lerobot_local(
                    data_dir=offline_root,
                    rb=offline_rb,
                    image_keys=image_keys,
                    image_size=(ecfg.image_size_h, ecfg.image_size_w),
                    max_episodes=cfg.offline_data.num_episodes,
                    lowdim_keys=lowdim_keys,
                    object_state_mode=str(getattr(args_cli, 'object_state_mode', 'raw')),
                )
            _log("Populating offline buffer from CSV episodes...")
            return populate_offline_buffer_from_csv(
                data_dir=cfg.offline_data.csv_data_dir,
                rb=offline_rb,
                split_file=cfg.offline_data.split_file if cfg.offline_data.split_file else None,
                split=cfg.offline_data.split,
                image_keys=image_keys,
                image_size=(ecfg.image_size_h, ecfg.image_size_w),
                max_episodes=cfg.offline_data.num_episodes,
            )

        # Offline cache hash includes data source + model dims
        _offline_extra = {
            "offline_data_dir": str(cfg.offline_data.csv_data_dir),
            "offline_num_episodes": cfg.offline_data.num_episodes,
            "offline_split": str(getattr(cfg.offline_data, 'split', 'training')),
            "image_size_h": int(ecfg.image_size_h),
            "image_size_w": int(ecfg.image_size_w),
        }
        offline_cache_hash = _buffer_cache_key(ecfg, acfg, extra=_offline_extra)
        offline_cache_dir = cache_root / f"offline_{offline_cache_hash}"
        _loaded_offline_from_cache = False

        if offline_cache_dir.exists() and (offline_cache_dir / "cache_meta.json").exists():
            try:
                from resfit.rl_finetuning.utils.hugging_face import optimized_replay_buffer_loads as _ofl
                ometa = _json.loads((offline_cache_dir / "cache_meta.json").read_text())
                offline_rb = _make_offline_rb(int(ometa.get("buffer_size", acfg.buffer_size)))
                _ofl(offline_rb, offline_cache_dir)
                _loaded_offline_from_cache = True
                _log(f"Loaded offline cache (hash={offline_cache_hash}): {offline_cache_dir} (size={len(offline_rb)})")
                _log(f"  Data: {ometa.get('offline_data_dir','?')}, eps={ometa.get('offline_num_episodes','?')}, dims: lowdim={ometa.get('lowdim_dim','?')} vlm={ometa.get('vlm_latent_dim','?')}")
            except Exception as e:
                _log(f"Failed to load offline cache {offline_cache_hash}: {e}; re-populating...")
                import shutil
                shutil.rmtree(offline_cache_dir, ignore_errors=True)
                _loaded_offline_from_cache = False

        if not _loaded_offline_from_cache:
            n_offline = _populate_current_offline_rb()
            n_step_drop_tolerance = max(0, int(acfg.n_step) - 1)
            expected_min_size = max(0, int(n_offline) - n_step_drop_tolerance)
            if len(offline_rb) < expected_min_size:
                offline_rb = _make_offline_rb(n_offline)
                n_offline = _populate_current_offline_rb()
            _log(f"Offline buffer: {n_offline} transitions, size={len(offline_rb)}")

            # Save offline cache
            try:
                from resfit.rl_finetuning.utils.hugging_face import optimized_replay_buffer_dumps as _ofd
                offline_cache_dir.mkdir(parents=True, exist_ok=True)
                _ofd(offline_rb, offline_cache_dir)
                offline_meta = {
                    "hash": offline_cache_hash,
                    "offline_data_dir": str(cfg.offline_data.csv_data_dir),
                    "offline_num_episodes": cfg.offline_data.num_episodes,
                    "buffer_size": len(offline_rb),
                    "lowdim_dim": int(lowdim_dim),
                    "vlm_latent_dim": int(vlm_latent_dim),
                    "action_dim": int(action_dim),
                    "image_keys": sorted(image_keys),
                    "image_size": [int(ecfg.image_size_h), int(ecfg.image_size_w)],
                    "n_step": int(acfg.n_step),
                    "gamma": float(acfg.gamma),
                }
                (offline_cache_dir / "cache_meta.json").write_text(_json.dumps(offline_meta, indent=2))
                _log(f"Saved offline cache (hash={offline_cache_hash}): {offline_cache_dir}")
            except Exception as e:
                _log(f"Failed to save offline cache: {e}")
    else:
        _log("Skipping offline buffer (offline_fraction=0 or no offline_data config)")

    # ── Debug: Log training configuration summary ──
    _log("════════════════════════════════════════════════════════")
    _log(f"  OFFLINE_FRACTION = {acfg.offline_fraction}")
    _log(f"  online_batch_size = {online_batch_size}")
    _log(f"  offline_batch_size = {offline_batch_size}")
    _log(f"  offline_rb size = {len(offline_rb) if offline_rb is not None and hasattr(offline_rb, '__len__') else 0}")
    _log(f"  online_rb max = {acfg.buffer_size}")
    _log(f"  reward_type = {getattr(ecfg, 'reward_type', 'N/A')}")
    _log(f"  success_threshold = {getattr(ecfg, 'success_threshold', 'N/A')}")
    _log(f"  action_scale = {cfg.agent.actor.action_scale}")
    _log(f"  actor_lr = {cfg.agent.actor_lr}")
    _log(f"  num_train_envs = {num_envs}")
    if cfg.offline_data is not None:
        _log(f"  offline_data_dir = {cfg.offline_data.csv_data_dir}")
    # VLM latent status
    if vlm_latent_dim > 0:
        _log(f"  VLM ENABLED: raw_dim={vlm_latent_dim}, projected=128, in critic_opt")
        _log(f"  vlm_projector params: {sum(p.numel() for p in agent.vlm_projector.parameters())}")
    else:
        _log(f"  VLM DISABLED (vlm_latent_dim=0)")
    _log("════════════════════════════════════════════════════════")

    # ── W&B ──
    _wb = wandb  # local alias so we can disable without UnboundLocalError
    if _wb is not None:
        if is_dataclass(cfg):
            wandb_cfg = asdict(cfg)
        else:
            try:
                wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            except Exception:
                wandb_cfg = {}
        if _wb.run is not None:
            _wb.finish()
        try:
            _wb.init(
                id=cfg.wandb.continue_run_id,
                resume=None if cfg.wandb.continue_run_id is None else "allow",
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=wandb_cfg,
                name=run_name,
                mode=cfg.wandb.mode,
                notes=cfg.wandb.notes,
                group=cfg.wandb.group,
                reinit=True,
            )
        except Exception as e:
            _log(f"[WARN] wandb.init() failed: {e}")
            _log("[WARN] Continuing without wandb logging. Set WANDB_MODE=disabled to suppress.")
            _wb = None

    # ── Output / checkpoint dirs ──
    outputs_dir = Path(getattr(cfg, "output_dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outputs_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Local SSD checkpoint dir for fast eval-polling I/O
    # Periodic checkpoints go here; only best/final are saved to NAS
    _slurm_jid = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    _local_ckpt_dir = Path(f"/tmp/resfit_ckpts_{_slurm_jid}")
    _local_ckpt_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Local checkpoint dir (SSD): {_local_ckpt_dir}")
    _log(f"NAS checkpoint dir: {checkpoint_dir}")

    # ── Save training config to output dir ──
    try:
        import json as _json_mod
        _cfg_dict = {}
        # Collect all CLI args
        _cfg_dict["cli_args"] = {k: str(v) for k, v in vars(args_cli).items()
                                 if not k.startswith("_") and k not in ("func",)}
        # Collect key hyperparameters
        _cfg_dict["algo"] = {
            "total_timesteps": acfg.total_timesteps,
            "learning_starts": acfg.learning_starts,
            "critic_warmup_steps": acfg.critic_warmup_steps,
            "batch_size": acfg.batch_size,
            "buffer_size": acfg.buffer_size,
            "gamma": acfg.gamma,
            "n_step": acfg.n_step,
            "offline_fraction": acfg.offline_fraction,
            "stddev_max": acfg.stddev_max,
            "stddev_min": acfg.stddev_min,
            "stddev_step": acfg.stddev_step,
            "num_updates_per_iteration": acfg.num_updates_per_iteration,
            "update_every_n_steps": acfg.update_every_n_steps,
        }
        _cfg_dict["agent"] = {
            "actor_lr": cfg.agent.actor_lr,
            "critic_lr": cfg.agent.critic_lr,
            "action_scale": cfg.agent.actor.action_scale,
            "action_l2_reg_weight": cfg.agent.actor.action_l2_reg_weight,
            "clip_q_target_to_reward_range": cfg.agent.clip_q_target_to_reward_range,
            "critic_target_tau": cfg.agent.critic_target_tau,
        }
        _cfg_dict["env"] = {
            "num_train_envs": num_envs,
            "max_episode_steps": int(ecfg.max_episode_steps),
            "reward_type": getattr(ecfg, "reward_type", "unknown"),
            "success_threshold": getattr(ecfg, "success_threshold", 0.03),
        }
        _cfg_dict["seed"] = cfg.seed
        _cfg_path = outputs_dir / "training_config.json"
        with open(_cfg_path, "w") as _cf:
            _json_mod.dump(_cfg_dict, _cf, indent=2, default=str)
        _log(f"Training config saved to {_cfg_path}")
    except Exception as _e:
        _log(f"[WARN] Failed to save training config: {_e}")

    # ── Async evaluator (launches separate Isaac Sim for eval) ──
    _debug_first_ep_video_early = bool(getattr(args_cli, "debug_first_episode_video", False))
    async_eval: AsyncEvaluator | None = None
    if _inline_eval_disabled and not _debug_first_ep_video_early:
        async_eval = AsyncEvaluator(cfg)
        async_eval._phase_probe_path = getattr(args_cli, 'phase_probe_path', None)
        async_eval._use_depth = _use_depth
        async_eval._depth_norm_path = getattr(args_cli, 'depth_norm_path', None)
        async_eval._use_vlm = _use_vlm
        async_eval._use_state = _use_state
        async_eval._object_state_mode = str(getattr(args_cli, 'object_state_mode', 'raw'))
        async_eval._contact_binary = bool(getattr(args_cli, 'contact_binary', False))
        async_eval._critic_hidden_dim = getattr(args_cli, 'critic_hidden_dim', None)
        async_eval._actor_hidden_dim = getattr(args_cli, 'actor_hidden_dim', None)
        async_eval._asymmetric_critic = _asymmetric
        async_eval_results_dir = outputs_dir / "async_eval_results"
        # Eval logs to its own wandb run (same project) — sharing a single run causes timeout/conflicts
        _wandb_run_id = None
        async_eval.start(_local_ckpt_dir, async_eval_results_dir, wandb_run_id=_wandb_run_id)
        _log(f"[AsyncEval] Using local SSD checkpoints: {_local_ckpt_dir}")

    # Helper: slice obs to training envs only (env 0)
    def _slice_obs_train(obs_full):
        return {k: v[:num_envs] for k, v in obs_full.items()}

    # Helper: pad action from (1,7) to (scene_num_envs,7) with zeros
    def _pad_action(action):
        if action.shape[0] >= scene_num_envs:
            return action
        pad = torch.zeros((scene_num_envs - action.shape[0], action.shape[1]), device=action.device, dtype=action.dtype)
        return torch.cat([action, pad], dim=0)

    # ══════════════════════════════════════════════════════════════════
    # Online warmup buffer caching (uses _buffer_cache_key defined above)
    # ══════════════════════════════════════════════════════════════════

    # ── Online warmup cache ──
    cache_hash = _buffer_cache_key(ecfg, acfg)
    online_cache_dir = cache_root / f"warmup_{cache_hash}"
    _loaded_online_from_cache = False

    # Check if matching cache exists
    if online_cache_dir.exists() and (online_cache_dir / "cache_meta.json").exists():
        try:
            meta = _json.loads((online_cache_dir / "cache_meta.json").read_text())
            from resfit.rl_finetuning.utils.hugging_face import optimized_replay_buffer_loads
            online_rb.sampler._empty()
            optimized_replay_buffer_loads(online_rb, online_cache_dir)
            _loaded_online_from_cache = True
            _log(f"Loaded warmup cache (hash={cache_hash}): {online_cache_dir} (size={len(online_rb)})")
            _log(f"  Cache config: csv={meta.get('csv_base_dir','?')}, thresh={meta.get('success_threshold','?')}, envs={meta.get('num_envs','?')}")
            _log(f"  Model dims: lowdim={meta.get('lowdim_dim','?')}, vlm={meta.get('vlm_latent_dim','?')}, action={meta.get('action_dim','?')}")
        except Exception as e:
            _log(f"Failed to load cache {cache_hash}: {e}; re-collecting...")
            import shutil
            shutil.rmtree(online_cache_dir, ignore_errors=True)
    else:
        # Scan for existing caches with different hash
        if cache_root.exists():
            for d in cache_root.iterdir():
                if d.is_dir() and d.name.startswith("warmup_") and d.name != f"warmup_{cache_hash}":
                    _log(f"  Found other warmup cache: {d.name} (different config)")

    # ══════════════════════════════════════════════════════════════════
    # Custom env snapshot: deterministic cube poses shared across warmup/train/eval
    # ══════════════════════════════════════════════════════════════════
    custom_env_root = outputs_dir / "custom_env"
    _env_hash = compute_env_hash(
        csv_base_dir=str(ecfg.csv_base_dir),
        success_threshold=float(getattr(ecfg, 'success_threshold', 0.005)),
        cube_perturb_table_path=str(getattr(ecfg, 'cube_perturb_table_path', '') or ''),
        cube_perturb_range=float(getattr(ecfg, 'cube_perturb_range', 0.0)),
        num_envs=scene_num_envs,
        max_episode_steps=int(ecfg.max_episode_steps),
    )
    _perturb_names = []
    _pt_path = getattr(ecfg, 'cube_perturb_table_path', None)
    if _pt_path and Path(_pt_path).exists():
        import json as _jj
        _pt = _jj.loads(Path(_pt_path).read_text())
        _perturb_names = [e.get('name', f"env{e['env_id']}") for e in _pt.get('envs', [])]
    while len(_perturb_names) < scene_num_envs:
        _perturb_names.append(f"env{len(_perturb_names)}")

    _has_env_snapshot = snapshot_exists(custom_env_root, _env_hash)
    _log(f"Env snapshot hash={_env_hash}, exists={_has_env_snapshot}")

    # ══════════════════════════════════════════════════════════════════
    # Debug first episode video flag (needed before warmup)
    # ══════════════════════════════════════════════════════════════════
    _debug_first_ep_video = bool(getattr(args_cli, "debug_first_episode_video", False))

    # ══════════════════════════════════════════════════════════════════
    # Warm-up: fill online buffer with base-policy + noise exploration
    # All envs run 1 episode each simultaneously for diverse warmup data.
    # ══════════════════════════════════════════════════════════════════
    _set_phase("warmup")
    all_scene_env_ids = list(range(scene_num_envs))
    warmup_per_env_frames: dict[int, list[np.ndarray]] = {}

    if _loaded_online_from_cache and len(online_rb) >= acfg.learning_starts:
        _log(f"Warm-up SKIPPED — loaded {len(online_rb)} transitions from cache: {online_cache_dir}")
        # Still need to reset env and apply snapshot
        env.set_active_env_ids(all_scene_env_ids)
        obs_full, _ = env.reset()
        if _has_env_snapshot:
            load_env_snapshot(custom_env_root, _env_hash, env)
            _log(f"Applied env snapshot (hash={_env_hash}) to match warmup env state")
        capture_sanity_frames(env, outputs_dir, "warmup_cached", _perturb_names)
        env.set_active_env_ids([0])
        obs = _slice_obs_train(obs_full)
    else:
        N_warmup = scene_num_envs
        _log(f"Warm-up: {N_warmup} envs running 1 episode each simultaneously (base policy + noise)...")

        succ_thresh = float(getattr(ecfg, 'success_threshold', 0.005))

        # Full reset with all envs active
        env.set_active_env_ids(all_scene_env_ids)
        obs_full, _ = env.reset()

        # Save env snapshot (cube poses after settle) for reproducibility
        if not _has_env_snapshot:
            _snap_config = {
                "csv_base_dir": str(ecfg.csv_base_dir),
                "success_threshold": float(getattr(ecfg, 'success_threshold', 0.005)),
                "cube_perturb_table_path": str(getattr(ecfg, 'cube_perturb_table_path', '') or ''),
                "cube_perturb_range": float(getattr(ecfg, 'cube_perturb_range', 0.0)),
                "num_envs": scene_num_envs,
                "max_episode_steps": int(ecfg.max_episode_steps),
            }
            save_env_snapshot(custom_env_root, _env_hash, env, _snap_config, _perturb_names)
        else:
            load_env_snapshot(custom_env_root, _env_hash, env)
            _log(f"Applied existing env snapshot (hash={_env_hash})")

        # Sanity check: capture first frames of all envs
        capture_sanity_frames(env, outputs_dir, "warmup_start", _perturb_names)

        # Per-env tracking
        env_done = torch.zeros(N_warmup, device=device, dtype=torch.bool)
        env_ep_max_h = torch.zeros(N_warmup, device=device)
        env_ep_reward = torch.zeros(N_warmup, device=device)
        env_ep_steps = torch.zeros(N_warmup, device=device, dtype=torch.long)
        warmup_total_transitions = 0
        for eid in range(N_warmup):
            warmup_per_env_frames[eid] = []

        step = 0
        while not env_done.all():
            # All envs step with noise residual
            noise = (torch.rand((N_warmup, action_dim), device=device) * 2 - 1) * acfg.random_action_noise_scale
            next_obs_full, reward_full, terminated_full, truncated_full, info = env.step(noise)
            done_full = terminated_full | truncated_full
            step += 1

            # Track per-env
            for eid in range(N_warmup):
                if env_done[eid]:
                    continue  # skip already-finished envs

                env_ep_reward[eid] += reward_full[eid].item()
                env_ep_steps[eid] += 1
                cube_z = env.cube.data.root_state_w[eid, 2].item()
                cube_init_z = env._initial_cube_z[eid].item()
                env_ep_max_h[eid] = max(env_ep_max_h[eid].item(), cube_z - cube_init_z)

                # Capture video frame (debug mode: every step with overlay; normal: every 5 steps)
                if _debug_first_ep_video and hasattr(env, "get_debug_frame"):
                    try:
                        warmup_per_env_frames[eid].append(env.get_debug_frame(eid, size=(256, 400)))
                    except Exception:
                        warmup_per_env_frames[eid].append(env.get_frame(eid, camera="front", size=(128, 160)))
                elif hasattr(env, "get_frame") and step % 5 == 0:
                    warmup_per_env_frames[eid].append(env.get_frame(eid, camera="front", size=(128, 160)))

                # Store transition ONLY if base_action is non-zero (GR00T has inferred)
                obs_1 = {k: v[eid:eid+1] for k, v in obs_full.items()}
                next_obs_1 = {k: v[eid:eid+1] for k, v in next_obs_full.items()}
                combined_action = info.get("scaled_action", noise)

                # Check base_action is valid (not all zeros from pre-inference state)
                # Only check dp+drot (first 6 dims), ignore grip (dim 6 is always non-zero: -1)
                base_act = obs_1.get("observation.base_action", None)
                base_is_zero = base_act is not None and base_act[0, :6].abs().sum().item() < 1e-6
                if base_is_zero:
                    if step <= 5:
                        _log(f"[DEBUG warmup] Skipping transition: base_action=0 (env={eid}, step={step})")
                    continue  # Don't store transitions with zero base_action

                # Debug: verify scaled_action ≈ base_action_norm when noise=0
                if warmup_total_transitions < 10:
                    ba = base_act[0].detach().cpu().numpy() if base_act is not None else None
                    sa = combined_action[eid].detach().cpu().numpy() if hasattr(combined_action[eid], 'detach') else combined_action[eid]
                    _log(f"[DEBUG warmup] env={eid} step={step}:")
                    _log(f"  base_action_norm={ba}")
                    _log(f"  scaled_action  ={sa}")
                    if ba is not None:
                        diff = float(np.abs(ba - sa).max())
                        _log(f"  max_diff={diff:.6f} {'OK' if diff < 0.01 else 'MISMATCH!'}")

                _add_transitions(
                    obs=obs_1, next_obs=next_obs_1, actions=combined_action[eid:eid+1],
                    reward=reward_full[eid:eid+1], done=done_full[eid:eid+1],
                    info=info, device=device,
                    image_keys=image_keys, lowdim_keys=lowdim_keys,
                    num_envs=1, online_rb=online_rb,
                )
                warmup_total_transitions += 1

                if done_full[eid]:
                    env_done[eid] = True

            obs_full = next_obs_full

            if step % 200 == 0:
                n_done = env_done.sum().item()
                _log(f"warm-up: step={step} transitions={warmup_total_transitions} "
                     f"envs_done={int(n_done)}/{N_warmup}")

        # Episode summaries + save videos
        warmup_vid_dir = outputs_dir / "warmup_videos"
        warmup_vid_dir.mkdir(parents=True, exist_ok=True)
        warmup_success_count = 0
        for eid in range(N_warmup):
            h = env_ep_max_h[eid].item()
            succ = h >= succ_thresh
            if succ:
                warmup_success_count += 1
            s_tag = "succ" if succ else "fail"
            r = env_ep_reward[eid].item()
            steps = int(env_ep_steps[eid].item())
            _log(f"  env {eid}: {s_tag} h={h:.4f} R={r:.1f} steps={steps}")

            frames = warmup_per_env_frames.get(eid, [])
            if frames:
                pname = _perturb_names[eid] if eid < len(_perturb_names) else f"env{eid}"
                vpath = warmup_vid_dir / f"warmup_env{eid}_{pname}_{s_tag}.mp4"
                w = imageio.get_writer(str(vpath), fps=20)
                for fr in frames:
                    w.append_data(fr)
                w.close()

        sr = warmup_success_count / max(1, N_warmup)
        _log(f"Warm-up done. transitions={warmup_total_transitions}, "
             f"success={warmup_success_count}/{N_warmup} ({sr*100:.0f}%)")

        # If debug_first_episode_video: exit after warmup (base policy eval only)
        if _debug_first_ep_video:
            _log(f"[DEBUG] Base policy eval complete. {warmup_success_count}/{N_warmup} success. Videos saved to {warmup_vid_dir}")
            _log(f"[DEBUG] Exiting early (--debug_first_episode_video mode).")
            if _wb is not None and _wb.run is not None:
                _wb.summary["warmup_success_rate"] = float(sr)
                _wb.summary["warmup_success_count"] = warmup_success_count
                _wb.summary["warmup_num_envs"] = N_warmup
                _wb.finish()
            env.close()
            return

        # Save warmup cache with metadata
        try:
            from resfit.rl_finetuning.utils.hugging_face import optimized_replay_buffer_dumps
            online_cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(online_rb, online_cache_dir)
            # Save cache metadata for future validation
            cache_meta = {
                "hash": cache_hash,
                "csv_base_dir": str(ecfg.csv_base_dir),
                "success_threshold": float(getattr(ecfg, 'success_threshold', 0.005)),
                "cube_perturb_table_path": str(getattr(ecfg, 'cube_perturb_table_path', '')),
                "cube_perturb_range": float(getattr(ecfg, 'cube_perturb_range', 0.0)),
                "num_envs": int(ecfg.num_envs),
                "max_episode_steps": int(ecfg.max_episode_steps),
                "random_action_noise_scale": float(acfg.random_action_noise_scale),
                "n_step": int(acfg.n_step),
                "gamma": float(acfg.gamma),
                "buffer_size": len(online_rb),
                "success_rate": float(sr),
                # Model dims (for human inspection of what's in the cache)
                "lowdim_dim": int(lowdim_dim),
                "vlm_latent_dim": int(vlm_latent_dim),
                "action_dim": int(action_dim),
                "image_keys": sorted(image_keys),
                "image_size": [int(img_h), int(img_w)],
                "phase_probe_path": str(getattr(args_cli, 'phase_probe_path', '') or ''),
            }
            (online_cache_dir / "cache_meta.json").write_text(_json.dumps(cache_meta, indent=2))
            _log(f"Saved warmup cache (hash={cache_hash}): {online_cache_dir}")
        except Exception as e:
            _log(f"Failed to save warmup cache: {e}")

    # ══════════════════════════════════════════════════════════════════
    # Critic warmup (critic-only updates, no actor)
    # ══════════════════════════════════════════════════════════════════
    if acfg.critic_warmup_steps > 0:
        _set_phase("critic_warmup")
        _log(f"Critic warmup: {acfg.critic_warmup_steps} critic-only updates...")
        for i in range(acfg.critic_warmup_steps):
            batch = online_rb.sample(online_batch_size).to(device, non_blocking=True)
            if offline_batch_size > 0 and len(offline_rb) > 0:
                obatch = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
                batch = torch.cat([batch, obatch], dim=0)
            cw_metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)
            if i % 100 == 0:
                cw_loss = cw_metrics.get('train/critic_loss', 0)
                cw_qt = cw_metrics.get('train/critic_qt', 0)
                _log(f"critic warmup: {i}/{acfg.critic_warmup_steps} critic_loss={cw_loss:.4f} critic_qt={cw_qt:.4f}")
        _log("Critic warmup done.")

    # ══════════════════════════════════════════════════════════════════
    # Main training loop
    # (matches original: agent.act() → residual → env.step(residual) →
    #  combined = base + residual → stored in replay buffer)
    # ══════════════════════════════════════════════════════════════════
    _set_phase("training_loop")

    # Expand to all train envs now (warmup used 1 env, training uses all)
    if _inline_eval_disabled:
        # All envs are train envs (no eval envs in scene)
        train_env_ids = list(range(scene_num_envs))
        eval_env_ids = []
    else:
        train_env_ids = list(range(int(ecfg.num_envs)))
        eval_env_ids = list(range(int(ecfg.num_envs), scene_num_envs))
    num_envs = len(train_env_ids)
    _log(f"Training loop: using {num_envs} train envs (IDs: {train_env_ids[:5]}{'...' if num_envs > 5 else ''})")
    _log(f"Update ratio: {acfg.num_updates_per_iteration}*{num_envs}={acfg.num_updates_per_iteration * num_envs} gradient updates per {num_envs} transitions (ratio={acfg.num_updates_per_iteration}:1)")

    env.set_active_env_ids(train_env_ids)
    obs_full, _ = env.reset()

    # Apply same env snapshot to ensure identical cube poses as warmup
    if snapshot_exists(custom_env_root, _env_hash):
        load_env_snapshot(custom_env_root, _env_hash, env)
    capture_sanity_frames(env, outputs_dir, "train_start", _perturb_names[:num_envs])

    obs = _slice_obs_train(obs_full)
    global_step = 0
    episode_count = 0
    best_success = 0.0
    actor_updates = 0
    metrics: dict = {}
    timer = TrainingTimer()
    train_start = time.time()
    training_cum_time = 0.0
    last_eval_video_path = None
    # Per-env episode tracking (for correct episode_return / episode_steps)
    ep_cum_reward = torch.zeros(num_envs, device=device)
    ep_step_counter = torch.zeros(num_envs, device=device, dtype=torch.long)
    disable_eval = str(os.environ.get("RESFIT_DISABLE_EVAL", "0")).lower() in {"1", "true", "yes", "on"}
    if getattr(args_cli, "disable_eval", False):
        disable_eval = True
    if disable_eval:
        _log("Inline eval DISABLED (use async eval process with eval_async_isaaclab.py)")
    debug_zero_residual = bool(getattr(args_cli, "debug_zero_residual", False))

    # Debug first episode video: record env 0's first complete episode with reward overlay
    _debug_first_ep_video = bool(getattr(args_cli, "debug_first_episode_video", False))
    _debug_ep_frames: list[np.ndarray] = []  # frames for env 0
    _debug_ep_log_rows: list[dict] = []  # per-step debug data for CSV
    _debug_ep_done = False  # True once first episode finishes
    if _debug_first_ep_video:
        _log("DEBUG: Will record first training episode (env 0) with reward overlay")

    _log(f"Starting training for {acfg.total_timesteps} steps...")
    if debug_zero_residual:
        _log("DEBUG: residual action forced to zeros")

    while global_step <= acfg.total_timesteps:
        iter_start = time.time()

        # ── (1) Collect: agent predicts residual, env combines with base ──
        with timer.time("env_step"):
            stddev = utils.schedule(acfg.stddev_schedule, global_step)

            with torch.no_grad(), utils.eval_mode(agent):
                residual_action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)

            # Debug: log agent input/output for first few steps
            if global_step < 5:
                _log(f"[DEBUG agent] step={global_step}:")
                _log(f"  obs.state shape={obs['observation.state'].shape} val={obs['observation.state'][0,:5].detach().cpu().numpy()}")
                _log(f"  obs.base_action shape={obs['observation.base_action'].shape} val={obs['observation.base_action'][0].detach().cpu().numpy()}")
                _log(f"  residual_action shape={residual_action.shape} val={residual_action[0].detach().cpu().numpy()}")
                _log(f"  residual L1={residual_action.abs().mean().item():.6f}")

            if debug_zero_residual:
                residual_action = torch.zeros_like(residual_action)

            next_obs_full, reward_full, terminated_full, truncated_full, info = env.step(_pad_action(residual_action))
            reward = reward_full[:num_envs]
            terminated = terminated_full[:num_envs]
            truncated = truncated_full[:num_envs]
            next_obs = _slice_obs_train(next_obs_full)
            done = terminated | truncated

        # ── Debug first episode video + CSV log: capture env 0 frame with reward overlay ──
        if _debug_first_ep_video and not _debug_ep_done:
            try:
                frame = env.get_debug_frame(env_id=0, size=(256, 400))
                _debug_ep_frames.append(frame)
            except Exception as e:
                if len(_debug_ep_frames) == 0:
                    _log(f"[DEBUG video] get_debug_frame failed: {e}")

            # Collect per-step debug data for CSV log
            _dbg = getattr(env, '_last_step_debug', None)
            if _dbg is not None:
                row = {
                    "step": int(_dbg["step"][0].item()) if hasattr(_dbg["step"], '__getitem__') else int(_dbg["step"]),
                    "contact_force": float(_dbg["contact_force"][0].item()),
                    "has_contact": float(_dbg["has_contact"][0].item()),
                    "cube_height": float(_dbg["cube_height"][0].item()),
                    "finger_cube_dist": float(_dbg["finger_cube_dist"][0].item()),
                    "reward": float(_dbg["reward"][0].item()),
                    "success": float(_dbg["success"][0].item()),
                    "sustained_contact": float(_dbg.get("sustained_contact", torch.zeros(1))[0].item()),
                    "contact_lost": float(_dbg.get("contact_lost_during_lift", torch.zeros(1))[0].item()),
                }
                # Reward components
                for key in ["reward_distance", "reward_contact", "reward_height", "reward_success",
                            "grasp_gate", "both_close", "gripper_closing",
                            "left_finger_cube_dist", "right_finger_cube_dist", "finger_joint_pos"]:
                    if key in _dbg:
                        row[key] = float(_dbg[key][0].item())
                _debug_ep_log_rows.append(row)

            # Check if env 0 is done → save video + CSV and exit
            if done[0]:
                _debug_ep_done = True

                # Save CSV log
                _debug_csv_path = outputs_dir / "debug_first_episode_log.csv"
                try:
                    import csv
                    if _debug_ep_log_rows:
                        with open(str(_debug_csv_path), "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=_debug_ep_log_rows[0].keys())
                            writer.writeheader()
                            writer.writerows(_debug_ep_log_rows)
                        _log(f"[DEBUG csv] Saved {len(_debug_ep_log_rows)} rows: {_debug_csv_path}")
                except Exception as e:
                    _log(f"[DEBUG csv] Failed to save: {e}")

                # Save video
                _debug_vid_path = outputs_dir / "debug_first_episode_reward.mp4"
                try:
                    w = imageio.get_writer(str(_debug_vid_path), fps=20)
                    for fr in _debug_ep_frames:
                        w.append_data(fr)
                    w.close()
                    _log(f"[DEBUG video] Saved first episode ({len(_debug_ep_frames)} frames): {_debug_vid_path}")
                    if _wb is not None and _wb.run is not None:
                        _wb.log({"debug/first_episode_video": _wb.Video(str(_debug_vid_path), format="mp4")}, step=global_step)
                        _wb.finish()
                except Exception as e:
                    _log(f"[DEBUG video] Failed to save: {e}")
                _log("[DEBUG video] First episode recorded. Exiting early.")
                env.close()
                return

        # Episode bookkeeping — cumulative return + step count per env
        ep_cum_reward += reward
        ep_step_counter += 1
        if done.any():
            done_mask = done.bool()
            n_done = done_mask.sum().item()
            episode_count += n_done
            if _wb is not None and _wb.run is not None:
                ep_return = float(ep_cum_reward[done_mask].mean().item())
                ep_steps = float(ep_step_counter[done_mask].float().mean().item())
                _wb.log({
                    "training/episode_return": ep_return,
                    "training/episode_steps": ep_steps,
                    "training/episode_count": episode_count,
                }, step=global_step)
            # Reset accumulators for finished envs
            ep_cum_reward[done_mask] = 0.0
            ep_step_counter[done_mask] = 0

            # Auto-reset training envs that are done
            done_train_ids = [train_env_ids[i] for i in range(num_envs) if done[i]]
            if done_train_ids:
                env.reset_envs(done_train_ids)
                next_obs_full = env._build_obs()
                next_obs = _slice_obs_train(next_obs_full)

        # Store combined action (base + residual) from env wrapper
        combined_action = info.get("scaled_action", residual_action)
        if combined_action.shape[0] > num_envs:
            combined_action = combined_action[:num_envs]

        # Skip transitions where base_action is zero (GR00T not yet inferred)
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        base_act_check = obs.get("observation.base_action", None)
        if base_act_check is not None:
            # Check dp+drot (first 6 dims), ignore grip (dim 6 is always -1 for zero base)
            base_nonzero = base_act_check[:, :6].abs().sum(dim=-1) > 1e-6
            valid_mask = base_nonzero
            n_skipped = (~valid_mask).sum().item()
            if n_skipped > 0 and global_step < 500:
                _log(f"[DEBUG train] Skipping {n_skipped}/{num_envs} transitions (base_action=0)")

        # Periodic debug logging — every 1000 steps throughout entire training
        # (not just first 10000) + strict assertion on base+res=combined
        if global_step % 1000 == 0:
            for eid in range(min(num_envs, 2)):  # log first 2 envs
                ba = obs["observation.base_action"][eid].detach().cpu().numpy()
                sa = combined_action[eid].detach().cpu().numpy()
                ra = residual_action[eid].detach().cpu().numpy() if residual_action is not None else np.zeros(7)
                st = obs["observation.state"][eid].detach().cpu().numpy()
                rw = reward[eid].item()
                _log(f"[DEBUG train step={global_step} env={eid}]")
                _log(f"  state({st.shape[0]}D): [{st[0]:.3f}, {st[1]:.3f}, {st[2]:.3f}, {st[3]:.3f}, ...] (eef_pos+quat)")
                _log(f"  base_action: [{ba[0]:.4f}, {ba[1]:.4f}, {ba[2]:.4f}, {ba[3]:.4f}, {ba[4]:.4f}, {ba[5]:.4f}, {ba[6]:.4f}]")
                _log(f"  residual:    [{ra[0]:.4f}, {ra[1]:.4f}, {ra[2]:.4f}, {ra[3]:.4f}, {ra[4]:.4f}, {ra[5]:.4f}, {ra[6]:.4f}]")
                _log(f"  combined:    [{sa[0]:.4f}, {sa[1]:.4f}, {sa[2]:.4f}, {sa[3]:.4f}, {sa[4]:.4f}, {sa[5]:.4f}, {sa[6]:.4f}]")
                # Strict check: combined must exactly equal clamp(base + residual, -1, 1)
                expected_combined = np.clip(ba + ra, -1.0, 1.0)
                exact_match = np.array_equal(expected_combined, sa)
                diff = np.abs(expected_combined - sa).max()
                base_is_zero = np.abs(ba[:6]).sum() < 1e-6
                if not base_is_zero:
                    assert exact_match, (
                        f"FATAL: base+res != combined at step={global_step} env={eid}! "
                        f"diff={diff:.10f}\n"
                        f"  base={ba}\n  res={ra}\n  expected={expected_combined}\n  actual={sa}"
                    )
                _log(f"  base+res=combined: {'EXACT' if exact_match else 'SKIP(base=0)'}")
                _log(f"  reward={rw:.4f} done={done[eid].item()}")
                # Contact force debug
                debug_info = getattr(env, '_last_step_debug', None)
                if debug_info is not None:
                    cf = debug_info['contact_force'][eid].item()
                    hc = debug_info['has_contact'][eid].item()
                    ch = debug_info['cube_height'][eid].item()
                    _log(f"  contact_force={cf:.3f}N has_contact={'ON' if hc > 0.5 else 'OFF'} cube_h={ch*100:.2f}cm")
                # VLM latent debug
                if "observation.vlm_latent" in obs:
                    vlm = obs["observation.vlm_latent"][eid].detach().cpu()
                    vlm_nz = (vlm.abs() > 1e-8).sum().item()
                    vlm_norm = vlm.norm().item()
                    _log(f"  vlm_latent: dim={vlm.shape[0]} nonzero={vlm_nz}/{vlm.shape[0]} norm={vlm_norm:.2f}")
                    # Phase probe debug: show predicted vs GT phase
                    if agent.vlm_projector is not None:
                        import torch as _t
                        with _t.no_grad():
                            logits = agent.vlm_projector(vlm.unsqueeze(0).to(device))
                            probs = _t.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                        phase_names = ["approach", "grasp", "lift"]
                        pred_phase = phase_names[probs.argmax()]
                        # GT phase from env debug info
                        gt_phase = "?"
                        if debug_info is not None:
                            ch_m = debug_info['cube_height'][eid].item()
                            fc_d = debug_info.get('finger_cube_dist', {})
                            fc_val = fc_d[eid].item() if hasattr(fc_d, '__getitem__') and eid < len(fc_d) else 1.0
                            hc_val = debug_info['has_contact'][eid].item() if 'has_contact' in debug_info else 0.0
                            if ch_m >= 0.005:
                                gt_phase = "lift"
                            elif fc_val < 0.05 or hc_val > 0.5:
                                gt_phase = "grasp"
                            else:
                                gt_phase = "approach"
                        match_str = "✓" if pred_phase == gt_phase else "✗"
                        _log(f"  phase_probe: [{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}] "
                             f"pred={pred_phase} gt={gt_phase} {match_str}")

        # ── Strict assertions on EVERY step (not just debug logs) ──
        # Check base+res=combined for ALL valid transitions using exact match
        if valid_mask.any():
            ba_all = obs["observation.base_action"][:num_envs].detach()  # (N, 7)
            ra_all = residual_action[:num_envs].detach() if residual_action is not None else torch.zeros(num_envs, 7, device=device)
            sa_all = combined_action[:num_envs].detach()  # (N, 7)
            expected_all = torch.clamp(ba_all + ra_all, -1.0, 1.0)
            for eid in range(num_envs):
                if not valid_mask[eid]:
                    continue
                if not torch.equal(expected_all[eid], sa_all[eid]):
                    diff_v = (expected_all[eid] - sa_all[eid]).abs().max().item()
                    assert False, (
                        f"FATAL: base+res != combined at step={global_step} env={eid}! "
                        f"max_diff={diff_v:.10f}\n"
                        f"  base={ba_all[eid].cpu().numpy()}\n"
                        f"  res={ra_all[eid].cpu().numpy()}\n"
                        f"  expected={expected_all[eid].cpu().numpy()}\n"
                        f"  actual={sa_all[eid].cpu().numpy()}"
                    )

        # ── Strict obs shape/content assertions on EVERY step ──
        assert obs["observation.state"].shape == (num_envs, 10), (
            f"obs.state shape {obs['observation.state'].shape} != ({num_envs}, 10)"
        )
        assert next_obs["observation.state"].shape == (num_envs, 10), (
            f"next_obs.state shape {next_obs['observation.state'].shape} != ({num_envs}, 10)"
        )
        assert not torch.isnan(obs["observation.state"]).any(), f"NaN in obs.state at step={global_step}"
        assert not torch.isnan(next_obs["observation.state"]).any(), f"NaN in next_obs.state at step={global_step}"
        assert not torch.isnan(reward).any(), f"NaN in reward at step={global_step}"

        # ── Asymmetric critic: verify depth + object_state in online obs ──
        if _asymmetric and global_step < 5:
            for _ck in image_keys:
                assert _ck in obs, f"ASYM online: missing {_ck} in obs at step={global_step}"
                _cv = obs[_ck]
                assert _cv.dim() == 4 and _cv.shape[0] == num_envs, f"ASYM online: {_ck} shape {_cv.shape}"
                assert (_cv.abs() > 1e-6).any(), f"ASYM online: {_ck} all zeros at step={global_step}"
            if "observation.object_state" in obs:
                _ov = obs["observation.object_state"]
                assert _ov.shape == (num_envs, object_state_dim), \
                    f"ASYM online: object_state shape {_ov.shape} != ({num_envs}, {object_state_dim})"
                assert (_ov.abs() > 1e-8).any(), f"ASYM online: object_state all zeros at step={global_step}"
            elif object_state_dim > 0:
                _log(f"WARNING: asymmetric but object_state missing from online obs at step={global_step}")
            _log(f"[ASYM ONLINE] step={global_step} depth_ok={all(k in obs for k in image_keys)} "
                 f"obj_state={'observation.object_state' in obs}({object_state_dim}D)")

        # Store valid transitions only
        if valid_mask.all():
            _add_transitions(
                obs=obs, next_obs=next_obs, actions=combined_action,
                reward=reward, done=done, info=info, device=device,
                image_keys=image_keys, lowdim_keys=lowdim_keys,
                num_envs=num_envs, online_rb=online_rb,
            )
        else:
            for eid in range(num_envs):
                if valid_mask[eid]:
                    obs_1 = {k: v[eid:eid+1] for k, v in obs.items()}
                    next_obs_1 = {k: v[eid:eid+1] for k, v in next_obs.items()}
                    _add_transitions(
                        obs=obs_1, next_obs=next_obs_1, actions=combined_action[eid:eid+1],
                        reward=reward[eid:eid+1], done=done[eid:eid+1],
                        info=info, device=device,
                        image_keys=image_keys, lowdim_keys=lowdim_keys,
                        num_envs=1, online_rb=online_rb,
                    )
        obs = next_obs

        # ── (2) Periodic evaluation with agent ──
        eval_interval = max(1, int(cfg.eval_interval_every_steps))
        should_eval = (global_step % eval_interval == 0) and (bool(cfg.eval_first) or global_step > 0)

        # ── (2a) Async eval path: trigger + collect (never blocks) ──
        if disable_eval and async_eval is not None:
            if should_eval:
                async_eval.trigger_eval(agent, global_step, _local_ckpt_dir)
            # Non-blocking: check for completed eval results
            async_result = async_eval.collect_results()
            if async_result is not None:
                current_success = float(async_result.get("eval/success_rate", 0.0))
                if current_success > best_success:
                    _log(f"[AsyncEval] New best: {current_success:.4f} (prev {best_success:.4f})")
                    best_success = current_success
                    _save_checkpoint(agent, checkpoint_dir / "best_agent.pt")
                result_step = async_result.get("step", global_step)
                _log(f"[AsyncEval] step={result_step} success={current_success:.3f} return={async_result.get('eval/mean_return', 0):.3f}")
                # NOTE: eval subprocess already logs to wandb directly (same run_id),
                # so we do NOT log eval metrics from train side to avoid duplicates/step conflicts.

        # ── (2b) Inline eval path (original, blocks training) ──
        elif should_eval and not disable_eval:
            _set_phase("evaluation")
            n_eval = len(eval_env_ids)
            _log(f"Eval: {n_eval} envs, step={global_step}...")
            # Reset ONLY eval envs (1..9), training env 0 is untouched
            env.reset_envs(eval_env_ids)
            env.set_active_env_ids(eval_env_ids)
            with timer.time("evaluation"):
                eval_metrics, eval_video_path = _run_eval_with_agent(
                    eval_env=eval_env,
                    agent=agent,
                    num_episodes=n_eval,
                    eval_env_ids=eval_env_ids,
                    device=device,
                    global_step=global_step,
                    outputs_dir=outputs_dir,
                    run_name=run_name,
                    save_video=bool(getattr(cfg, "save_video", False)),
                    image_keys=image_keys,
                    debug_zero_residual=debug_zero_residual,
                )
            current_success = float(eval_metrics.get("eval/success_rate", 0.0))
            if eval_video_path is not None:
                last_eval_video_path = eval_video_path
            if current_success > best_success:
                _log(f"New best: {current_success:.4f} (prev {best_success:.4f}). Saving checkpoint.")
                best_success = current_success
                _save_checkpoint(agent, checkpoint_dir / "best_agent.pt")
            succ_len = eval_metrics.get('eval/mean_successful_episode_length', 0)
            _log(f"Eval: success={current_success:.3f} return={eval_metrics.get('eval/mean_return', 0):.3f} succ_len={succ_len:.0f}")
            if _wb is not None and _wb.run is not None:
                log_dict = {k: v for k, v in eval_metrics.items() if not k.startswith("_")}
                if last_eval_video_path is not None:
                    log_dict["eval/video"] = _wb.Video(str(last_eval_video_path), format="mp4")
                _wb.log(log_dict, step=global_step)
            # Switch back to training mode — NO reset of env 0
            env.set_active_env_ids(train_env_ids)
            _set_phase("training_loop")

        global_step += num_envs

        # ── (3) Gradient updates ──
        # Scale updates_per_iteration by num_envs to maintain update-to-data ratio
        # Original (1 env): 4 updates per 1 transition = 4:1 ratio
        # Multi-env: 4*num_envs updates per num_envs transitions = 4:1 ratio
        scaled_updates = acfg.num_updates_per_iteration * num_envs
        if global_step % acfg.update_every_n_steps == 0 or global_step == num_envs:
            actor_cadence = max(1, scaled_updates // (acfg.actor_updates_per_iteration * num_envs))
            for i in range(scaled_updates):
                with timer.time("batch_sampling"):
                    batch = online_rb.sample(online_batch_size).to(device, non_blocking=True)
                    if offline_batch_size > 0 and len(offline_rb) > 0:
                        obatch = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
                        batch = torch.cat([batch, obatch], dim=0)

                update_actor = (i + 1) % actor_cadence == 0
                if update_actor:
                    actor_lr_warmup_steps = int(getattr(acfg, "actor_lr_warmup_steps", 0))
                    if actor_lr_warmup_steps > 0:
                        warmup_progress = min(1.0, actor_updates / max(1, actor_lr_warmup_steps))
                        for pg in agent.actor_opt.param_groups:
                            pg["lr"] = float(cfg.agent.actor_lr) * warmup_progress
                    actor_updates += 1

                # Action scale annealing: gradually increase from 0.01 to target
                _anneal_steps = int(getattr(args_cli, 'action_scale_anneal_steps', 0) or 0)
                if _anneal_steps > 0:
                    _target_scale = float(args_cli.action_scale) if args_cli.action_scale is not None else cfg.agent.actor.action_scale
                    _anneal_progress = min(1.0, global_step / max(1, _anneal_steps))
                    _current_scale = 0.01 + (float(_target_scale) - 0.01) * _anneal_progress
                    agent.actor.cfg.action_scale = _current_scale
                    if hasattr(agent, 'actor_target'):
                        agent.actor_target.cfg.action_scale = _current_scale

                with timer.time("gradient_update"):
                    metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

                metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
                metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()

        training_cum_time += time.time() - iter_start

        # ── (4) Periodic checkpoint ──
        ckpt_interval = int(getattr(cfg, "checkpoint_interval", 50_000))
        eval_ckpt_interval = max(1, int(cfg.eval_interval_every_steps))
        # Save at both checkpoint_interval AND eval_interval (for async eval process)
        if global_step > 0 and (
            (ckpt_interval > 0 and global_step % ckpt_interval == 0) or
            (eval_ckpt_interval > 0 and global_step % eval_ckpt_interval == 0)
        ):
            # Save to local SSD for fast async eval polling
            _save_checkpoint(agent, _local_ckpt_dir / f"agent_step{global_step}.pt")
            # Also save to NAS only if performance threshold met
            _min_save = float(getattr(args_cli, 'min_save_success_rate', 0.0))
            if _min_save <= 0 or best_success > _min_save:
                import shutil
                shutil.copy2(
                    str(_local_ckpt_dir / f"agent_step{global_step}.pt"),
                    str(checkpoint_dir / f"agent_step{global_step}.pt"),
                )

        # ── (5) Logging ──
        wandb_log_every = max(1, int(getattr(cfg, "wandb_log_every_steps", 10)))
        should_log_console = (global_step % 500 == 0)
        should_log_wandb = (_wb is not None and _wb.run is not None and global_step % wandb_log_every == 0)

        if should_log_console or should_log_wandb:
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0
            ts = timer.get_timing_stats()

            if should_log_wandb:
                log_dict = {
                    "training/SPS": sps,
                    "training/global_step": global_step,
                    "buffer/online_size": len(online_rb),
                    "buffer/offline_size": len(offline_rb) if offline_rb is not None and hasattr(offline_rb, '__len__') else 0,
                    "buffer/offline_fraction": acfg.offline_fraction,
                    "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
                    "timing/training_total_time": time.time() - train_start,
                    "timing/aggregate_steps_per_second": global_step / max(1, time.time() - train_start),
                }
                # Contact force / reward debug from env wrapper
                debug_info = getattr(env, '_last_step_debug', None)
                if debug_info is not None:
                    log_dict["env/mean_contact_force"] = float(debug_info['contact_force'].mean().item())
                    log_dict["env/contact_rate"] = float(debug_info['has_contact'].mean().item())
                    log_dict["env/mean_cube_height"] = float(debug_info['cube_height'].mean().item())
                    log_dict["env/mean_reward"] = float(debug_info['reward'].mean().item())
                    log_dict["env/mean_finger_cube_dist"] = float(debug_info['finger_cube_dist'].mean().item())
                # VLM latent wandb logging
                if vlm_latent_dim > 0 and "observation.vlm_latent" in obs:
                    vlm_env = obs["observation.vlm_latent"][:num_envs].detach()
                    log_dict["vlm/mean_norm"] = float(vlm_env.norm(dim=-1).mean().item())
                    log_dict["vlm/nonzero_rate"] = float((vlm_env.abs() > 1e-8).any(dim=-1).float().mean().item())
                    # VLM projector weight norm (if learnable)
                    if hasattr(agent, 'vlm_projector') and agent.vlm_projector is not None:
                        proj_w = next(agent.vlm_projector.parameters())
                        log_dict["vlm/projector_weight_norm"] = float(proj_w.norm().item())
                        if proj_w.grad is not None:
                            log_dict["vlm/projector_grad_norm"] = float(proj_w.grad.norm().item())
                # Phase probe accuracy metrics (all envs)
                _phase_debug = getattr(env, '_last_step_debug', None)
                if agent.vlm_projector is not None and _phase_debug is not None:
                    _ps = _compute_phase_probe_stats(agent, obs, _phase_debug, num_envs, device)
                    if _ps is not None:
                        log_dict.update(_ps)
                log_dict.update(ts)
                filtered = {k: v for k, v in metrics.items() if not k.startswith("_")}
                log_dict.update(filtered)
                if "_actions" in metrics:
                    actions_np = metrics["_actions"].numpy().reshape(-1)
                    log_dict["train/residual_l1_magnitude"] = float(np.mean(np.abs(actions_np)))
                    log_dict["train/residual_l2_magnitude"] = float(np.mean(actions_np ** 2))
                    log_dict["histograms/residual_actions"] = _wb.Histogram(actions_np)
                if "_target_q" in metrics:
                    log_dict["histograms/critic_qt"] = _wb.Histogram(metrics["_target_q"].numpy().reshape(-1))
                _wb.log(log_dict, step=global_step)

            if should_log_console:
                env_pct = ts.get("timing/env_step_percentage", 0)
                grad_pct = ts.get("timing/gradient_update_percentage", 0)
                batch_pct = ts.get("timing/batch_sampling_percentage", 0)
                eval_pct = ts.get("timing/evaluation_percentage", 0)
                msg = (
                    f"[{global_step}] eps={int(episode_count)} SPS={sps} "
                    f"critic={metrics.get('train/critic_loss', 0):.4f} "
                    f"| env={env_pct:.1f}% grad={grad_pct:.1f}% batch={batch_pct:.1f}% eval={eval_pct:.1f}%"
                )
                if "train/actor_loss_base" in metrics:
                    msg += f" actor={metrics['train/actor_loss_base']:.4f}"
                if "train/actor_grad_norm" in metrics:
                    msg += f" grad_norm={metrics['train/actor_grad_norm']:.4f}"
                if "train/actor_l2_penalty" in metrics:
                    msg += f" l2={metrics['train/actor_l2_penalty']:.6f}"
                if "_actions" in metrics:
                    actions_np = metrics['_actions'].numpy()
                    msg += f" res_l1={float(np.mean(np.abs(actions_np))):.4f} res_l2={float(np.mean(actions_np**2)):.4f}"
                # VLM summary in console log
                if vlm_latent_dim > 0 and "observation.vlm_latent" in obs:
                    vlm_env = obs["observation.vlm_latent"][:num_envs].detach()
                    msg += f" vlm_norm={vlm_env.norm(dim=-1).mean().item():.1f}"
                    if hasattr(agent, 'vlm_projector') and agent.vlm_projector is not None:
                        pw = next(agent.vlm_projector.parameters())
                        msg += f" proj_w={pw.norm().item():.2f}"
                _log(msg)
                # Phase probe accuracy summary (all envs, every 500 steps)
                if agent.vlm_projector is not None and "observation.vlm_latent" in obs:
                    _phase_debug = getattr(env, '_last_step_debug', None)
                    _ps = _compute_phase_probe_stats(agent, obs, _phase_debug, num_envs, device)
                    if _ps is not None:
                        _log(f"  [PhaseProbe] {_format_phase_probe_summary(_ps, num_envs)}")
                        # Detailed per-env breakdown at key steps:
                        # step 0, first 5 intervals, then every 5000 steps
                        _do_detail = (global_step == 0
                                      or global_step <= 2500
                                      or global_step % 5000 == 0
                                      or global_step >= acfg.total_timesteps - num_envs)
                        if _do_detail:
                            vlm_raw = obs["observation.vlm_latent"][:num_envs].detach()
                            with torch.no_grad():
                                logits = agent.vlm_projector(vlm_raw.to(device))
                                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                            for eid in range(min(num_envs, 5)):  # first 5 envs
                                gt_id = _compute_gt_phase_from_debug(_phase_debug, eid)
                                pred_id = int(probs[eid].argmax())
                                match = "OK" if pred_id == gt_id else "MISS"
                                ch = _phase_debug['cube_height'][eid].item()
                                fd = _phase_debug['finger_cube_dist'][eid].item()
                                cf = _phase_debug['contact_force'][eid].item()
                                _log(f"    env{eid}: pred={_PHASE_NAMES[pred_id]}({probs[eid][pred_id]:.2f}) "
                                     f"gt={_PHASE_NAMES[gt_id]} [{match}] "
                                     f"probs=[{probs[eid][0]:.3f},{probs[eid][1]:.3f},{probs[eid][2]:.3f}] "
                                     f"h={ch*100:.1f}cm dist={fd*100:.1f}cm cf={cf:.2f}N")

        now = time.time()
        if now - train_start > 0 and global_step % max(1, heartbeat_interval_sec * 50) < num_envs:
            _log(f"heartbeat train: step={global_step}/{acfg.total_timesteps} rb={len(online_rb)} eps={int(episode_count)}")

    total_time = time.time() - train_start
    _log(f"Training finished in {total_time:.1f}s ({global_step} steps, {episode_count} episodes)")

    # Final checkpoint
    _save_checkpoint(agent, checkpoint_dir / "final_agent.pt")

    # Wait for async evaluator to finish all pending evals, then shut down
    if async_eval is not None:
        _log("[AsyncEval] Training done. Waiting for eval process to finish remaining checkpoints...")
        _wait_timeout = 3600  # max 1 hour wait
        _wait_start = time.time()
        while async_eval.is_alive() and (time.time() - _wait_start) < _wait_timeout:
            # Collect any results that come in while waiting
            result = async_eval.collect_results()
            if result is not None:
                step = result.get("step", 0)
                succ = result.get("eval/success_rate", 0.0)
                ret = result.get("eval/mean_return", 0.0)
                _log(f"[AsyncEval] step={step} success={succ:.3f} return={ret:.3f}")
                if succ > best_success:
                    best_success = succ
            time.sleep(15)
        elapsed_wait = time.time() - _wait_start
        if async_eval.is_alive():
            _log(f"[AsyncEval] Timed out after {elapsed_wait:.0f}s, force-shutting down.")
            async_eval.shutdown()
        else:
            # Collect any final results
            result = async_eval.collect_results()
            if result is not None:
                step = result.get("step", 0)
                succ = result.get("eval/success_rate", 0.0)
                _log(f"[AsyncEval] Final result: step={step} success={succ:.3f}")
                if succ > best_success:
                    best_success = succ
            _log(f"[AsyncEval] Eval process finished naturally after {elapsed_wait:.0f}s wait.")

    # Wandb summary
    if _wb is not None and _wb.run is not None:
        _wb.summary["environment/horizon"] = int(ecfg.max_episode_steps)
        _wb.summary["best_success_rate"] = best_success
        _wb.summary["total_time_sec"] = total_time
        _wb.finish()

    env.close()


def _save_checkpoint(agent: QAgent, path: Path):
    """Save agent state dict to disk (atomic write to avoid race with eval reader)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save({
        "encoders": agent.encoders.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "encoder_opt": agent.encoder_opt.state_dict(),
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "vlm_projector": agent.vlm_projector.state_dict() if agent.vlm_projector is not None else None,
    }, str(tmp_path))
    tmp_path.rename(path)  # atomic on same filesystem
    _log(f"Checkpoint saved: {path}")


def _run_eval_with_agent(
    *,
    eval_env,
    agent: QAgent,
    num_episodes: int,
    eval_env_ids: list[int] | None = None,
    device: torch.device,
    global_step: int,
    outputs_dir: Path,
    run_name: str,
    save_video: bool,
    image_keys: list[str],
    debug_zero_residual: bool = False,
) -> tuple[dict[str, float], Path | None]:
    """Batched evaluation on eval envs only (does NOT reset training env).

    eval_env_ids: which env indices are used for eval (e.g. [1,2,...,9]).
    The function assumes reset_envs(eval_env_ids) was already called.
    """
    N_scene = getattr(eval_env, "num_envs", 1)
    if eval_env_ids is None:
        eval_env_ids = list(range(N_scene))
    N = len(eval_env_ids)
    max_ep_steps = int(getattr(eval_env, "max_episode_steps", 2000))
    steps_per_action = getattr(eval_env, "steps_per_action", 5)

    # Build obs for eval envs (don't reset — already done by caller)
    obs = eval_env._build_obs()  # full (N_scene, ...) obs

    # Per-env tracking
    ep_rewards = torch.zeros(N, device=device)
    ep_ever_success = torch.zeros(N, device=device)
    ep_first_success_step = torch.full((N,), float(max_ep_steps), device=device)
    ep_steps = 0

    # Video frame buffers (128x160 = multiple of 16)
    frames_per_env: list[list[np.ndarray]] = [[] for _ in range(N)] if save_video else []

    while ep_steps < max_ep_steps:
        if debug_zero_residual:
            residual_action = torch.zeros((N_scene, 7), device=device, dtype=torch.float32)
        else:
            with torch.no_grad(), utils.eval_mode(agent):
                # Agent sees only eval env obs; pad to full scene size
                eval_obs = {k: v[eval_env_ids] for k, v in obs.items()}
                residual_per_eval = agent.act(eval_obs, eval_mode=True, stddev=0.0, cpu=False)
                residual_action = torch.zeros((N_scene, 7), device=device, dtype=torch.float32)
                for i, eid in enumerate(eval_env_ids):
                    residual_action[eid] = residual_per_eval[i]

        next_obs, reward, terminated, truncated, info = eval_env.step(residual_action)

        # Accumulate for eval envs only
        eval_reward = reward[eval_env_ids].to(device)
        ep_rewards += eval_reward

        # Success = cube lifted >= success_threshold
        cube_z = eval_env.cube.data.root_state_w[eval_env_ids, 2].to(device)
        cube_init_z = eval_env._initial_cube_z[eval_env_ids].to(device)
        _succ_thresh = getattr(eval_env, 'success_threshold', 0.005)
        cube_lifted = ((cube_z - cube_init_z) >= _succ_thresh).float()
        just_succeeded = (cube_lifted > 0) & (ep_ever_success == 0)
        ep_first_success_step[just_succeeded] = ep_steps
        ep_ever_success = torch.max(ep_ever_success, cube_lifted)
        ep_steps += 1

        # Capture frames at control-tick intervals
        if save_video and (ep_steps % steps_per_action == 0):
            if hasattr(eval_env, "get_frame"):
                for i, eid in enumerate(eval_env_ids):
                    frames_per_env[i].append(eval_env.get_frame(eid, camera="front", size=(128, 160)))

        obs = next_obs
        done = terminated | truncated
        if done[eval_env_ids].all():
            break

    # Compute metrics
    successes = ep_ever_success.cpu().numpy()
    returns = ep_rewards.cpu().numpy()

    # Mean successful episode length (steps until first success)
    succ_mask = successes > 0
    if succ_mask.any():
        succ_lengths = ep_first_success_step.cpu().numpy()[succ_mask]
        mean_succ_len = float(np.mean(succ_lengths))
    else:
        mean_succ_len = 0.0

    metrics = {
        "eval/success_rate": float(np.mean(successes)),
        "eval/mean_return": float(np.mean(returns)),
        "eval/mean_successful_episode_length": mean_succ_len,
        "eval/num_envs": N,
        "eval/episode_steps": ep_steps,
    }

    # Save videos
    video_path: Path | None = None
    if save_video and frames_per_env:
        eval_vid_dir = outputs_dir / "eval_videos"
        eval_vid_dir.mkdir(parents=True, exist_ok=True)
        # Per-env videos (save at most 4)
        for i in range(min(N, 4)):
            if frames_per_env[i]:
                p = eval_vid_dir / f"eval_step{global_step}_env{eval_env_ids[i]}.mp4"
                w = imageio.get_writer(str(p), fps=20)
                for fr in frames_per_env[i]:
                    w.append_data(fr)
                w.close()
        # Concat video (side-by-side, grid layout for >5 envs)
        min_len = min(len(f) for f in frames_per_env) if frames_per_env else 0
        if min_len > 0:
            cols = min(N, 5)
            rows = (N + cols - 1) // cols
            concat_path = eval_vid_dir / f"eval_step{global_step}_concat_{N}envs.mp4"
            w = imageio.get_writer(str(concat_path), fps=20)
            for fi in range(min_len):
                grid_rows = []
                for r in range(rows):
                    row_frames = []
                    for c in range(cols):
                        idx = r * cols + c
                        if idx < N:
                            row_frames.append(frames_per_env[idx][fi])
                        else:
                            row_frames.append(np.zeros_like(frames_per_env[0][fi]))
                    grid_rows.append(np.concatenate(row_frames, axis=1))
                grid = np.concatenate(grid_rows, axis=0)
                w.append_data(grid)
            w.close()
            video_path = concat_path
            _log(f"Eval video: {concat_path} ({min_len} frames, {N} envs, {rows}x{cols} grid)")

    return metrics, video_path


if __name__ == "__main__":
    cfg = ResidualTD3IsaacLabConfig()
    cfg.isaaclab_env.num_envs = args_cli.num_envs
    cfg.isaaclab_env.device = args_cli.device
    cfg.num_envs = args_cli.num_envs
    cfg.algo.total_timesteps = args_cli.total_timesteps
    if args_cli.learning_starts is not None:
        cfg.algo.learning_starts = args_cli.learning_starts
    if args_cli.critic_warmup_steps is not None:
        cfg.algo.critic_warmup_steps = args_cli.critic_warmup_steps
    if args_cli.update_every_n_steps is not None:
        cfg.algo.update_every_n_steps = args_cli.update_every_n_steps
    if args_cli.num_updates_per_iteration is not None:
        cfg.algo.num_updates_per_iteration = args_cli.num_updates_per_iteration
    if args_cli.offline_fraction is not None:
        cfg.algo.offline_fraction = args_cli.offline_fraction
    if args_cli.batch_size is not None:
        cfg.algo.batch_size = args_cli.batch_size
    if args_cli.buffer_size is not None:
        cfg.algo.buffer_size = args_cli.buffer_size
    if args_cli.gamma is not None:
        cfg.algo.gamma = args_cli.gamma
    if args_cli.n_step is not None:
        cfg.algo.n_step = args_cli.n_step
    if args_cli.stddev_max is not None:
        cfg.algo.stddev_max = args_cli.stddev_max
    if args_cli.stddev_min is not None:
        cfg.algo.stddev_min = args_cli.stddev_min
    if args_cli.random_action_noise_scale is not None:
        cfg.algo.random_action_noise_scale = args_cli.random_action_noise_scale
    if args_cli.actor_lr is not None:
        cfg.agent.actor_lr = args_cli.actor_lr
    if args_cli.action_scale is not None:
        cfg.agent.actor.action_scale = args_cli.action_scale
    cfg.groot_policy.host = args_cli.gr00t_host
    cfg.groot_policy.port = args_cli.gr00t_port
    cfg.groot_policy.model_path = args_cli.groot_model_path
    cfg.groot_policy.embodiment_tag = args_cli.groot_embodiment_tag
    cfg.groot_policy.policy_device = args_cli.groot_policy_device
    if args_cli.groot_policy_strict:
        cfg.groot_policy.strict = True
    if args_cli.task_description is not None:
        cfg.groot_policy.task_description = args_cli.task_description
    if args_cli.language_override is not None:
        cfg.groot_policy.language_override = args_cli.language_override
    cfg.groot_policy.task_description = _effective_task_description(
        task_description=str(cfg.groot_policy.task_description),
        language_override=cfg.groot_policy.language_override,
    )
    if args_cli.csv_base_dir is not None:
        cfg.isaaclab_env.csv_base_dir = args_cli.csv_base_dir
    if args_cli.csv_init_row_index is not None:
        cfg.isaaclab_env.csv_init_row_index = args_cli.csv_init_row_index
    if args_cli.max_episode_steps is not None:
        cfg.isaaclab_env.max_episode_steps = max(1, int(args_cli.max_episode_steps))
    if args_cli.success_threshold is not None:
        cfg.isaaclab_env.success_threshold = float(args_cli.success_threshold)
    if args_cli.cube_perturb_range is not None:
        cfg.isaaclab_env.cube_perturb_range = float(args_cli.cube_perturb_range)
    if args_cli.cube_perturb_table_path is not None:
        cfg.isaaclab_env.cube_perturb_table_path = args_cli.cube_perturb_table_path
    if args_cli.reward_type is not None:
        cfg.isaaclab_env.reward_type = args_cli.reward_type
    if args_cli.no_clip_q:
        cfg.agent.clip_q_target_to_reward_range = False
    if args_cli.actor_lr_warmup_steps is not None:
        cfg.algo.actor_lr_warmup_steps = args_cli.actor_lr_warmup_steps
    if args_cli.critic_hidden_dim is not None:
        cfg.agent.critic.hidden_dim = args_cli.critic_hidden_dim
    if args_cli.actor_hidden_dim is not None:
        cfg.agent.actor.hidden_dim = args_cli.actor_hidden_dim
    if args_cli.vlm_projected_dim is not None:
        cfg.agent.vlm_projected_dim = args_cli.vlm_projected_dim
    if args_cli.offline_data_dir is not None:
        if cfg.offline_data is None:
            from resfit.rl_finetuning.config.residual_td3_isaaclab import IsaacLabOfflineDataConfig
            cfg.offline_data = IsaacLabOfflineDataConfig()
        cfg.offline_data.csv_data_dir = args_cli.offline_data_dir
    cfg.seed = args_cli.seed
    cfg.device = args_cli.device
    cfg.wandb.mode = args_cli.wandb_mode
    if args_cli.wandb_project is not None:
        cfg.wandb.project = args_cli.wandb_project
    if args_cli.wandb_entity is not None:
        cfg.wandb.entity = args_cli.wandb_entity
    if args_cli.wandb_name is not None:
        cfg.wandb.name = args_cli.wandb_name
    if args_cli.wandb_group is not None:
        cfg.wandb.group = args_cli.wandb_group
    if args_cli.wandb_notes is not None:
        cfg.wandb.notes = args_cli.wandb_notes
    if args_cli.wandb_continue_run_id is not None:
        cfg.wandb.continue_run_id = args_cli.wandb_continue_run_id
    cfg.wandb_log_every_steps = max(1, int(args_cli.wandb_log_every_steps))
    if args_cli.eval_interval_every_steps is not None:
        cfg.eval_interval_every_steps = max(1, int(args_cli.eval_interval_every_steps))
    if args_cli.eval_num_episodes is not None:
        cfg.eval_episodes = max(1, int(args_cli.eval_num_episodes))
    if args_cli.eval_num_envs is not None:
        cfg.eval_num_envs = max(1, int(args_cli.eval_num_envs))
    if args_cli.eval_first:
        cfg.eval_first = True
    if args_cli.eval_use_iface_runner is not None:
        cfg.eval_use_iface_runner = bool(int(args_cli.eval_use_iface_runner))
    if args_cli.save_video:
        cfg.save_video = True
    if args_cli.eval_save_video:
        cfg.eval_save_video = True
    if args_cli.output_dir is not None:
        cfg.output_dir = args_cli.output_dir
    cfg.heartbeat_interval_sec = args_cli.heartbeat_interval_sec
    cfg.stack_dump_interval_sec = args_cli.stack_dump_interval_sec

    # Isaac Sim's simulation_app.close() deadlocks with wandb's asyncio
    # threads during async engine shutdown (3-minute timeout then crash).
    # ALWAYS force-exit — whether main() succeeds or crashes.
    try:
        main(cfg)
        _log("Training complete. Force-exiting to avoid Isaac Sim shutdown deadlock.")
        os._exit(0)
    except SystemExit as e:
        os._exit(e.code if e.code is not None else 0)
    except Exception:
        import traceback
        traceback.print_exc()
        _log("Training CRASHED. Force-exiting to avoid Isaac Sim shutdown deadlock.")
        os._exit(1)

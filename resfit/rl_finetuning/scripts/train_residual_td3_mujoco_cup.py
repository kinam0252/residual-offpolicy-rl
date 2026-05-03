"""
Residual TD3 training on MuJoCo Stand Cup + GR00T.

Standalone script using ``MuJoCoVecEnvCup`` + ``MuJoCoResidualWrapperCup``
for Stand Cup task.  Gripper uses raw metres [0, 0.04], NOT normalised.

Launch:
    source ~/.venvs/groot/bin/activate
    MUJOCO_GL=egl LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-} \
    python resfit/rl_finetuning/scripts/train_residual_td3_mujoco_cup.py \
        --groot_checkpoint ~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-200000 \
        --num_envs 27 --total_timesteps 500000
"""

from __future__ import annotations

import argparse
import os
import pprint
import random
import sys
import time
import subprocess
import json as _json_mod
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── Prevent deepspeed import crash when nvcc is missing ──
import importlib
import types as _types
if "deepspeed" not in sys.modules:
    _ds_mock = _types.ModuleType("deepspeed")
    _ds_mock.__version__ = "0.0.0"
    _ds_mock.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds_mock.__path__ = []
    _ds_mock.__file__ = __file__
    _ds_zero = _types.ModuleType("deepspeed.zero")
    _ds_zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _ds_zero.Init = lambda *a, **kw: (lambda f: f)
    _ds_mock.zero = _ds_zero
    sys.modules["deepspeed"] = _ds_mock
    sys.modules["deepspeed.zero"] = _ds_zero

# ── Patch huggingface_hub repo_id validator for local paths ──
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig_validate = _hf_val.validate_repo_id
    def _patched_validate_repo_id(repo_id: str) -> None:
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")):
            return
        return _orig_validate(repo_id)
    _hf_val.validate_repo_id = _patched_validate_repo_id
except Exception:
    pass

# Ensure resfit is importable
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
from tensordict import TensorDict
torch.backends.cuda.enable_cudnn_sdp(False)  # Fix cuDNN Frontend error on B200
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.normalization import ActionScaler
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup as MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup as MuJoCoVecEnv

try:
    import wandb
except ImportError:
    wandb = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def _log(msg: str) -> None:
    print(f"[mujoco-cup-td3] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
# Convert 8D combined action to 7D replay action for ActionScaler mode
# ══════════════════════════════════════════════════════════════════

def _to_replay_action_7d(combined_8d: torch.Tensor, residual_7d: torch.Tensor,
                         action_scaler) -> torch.Tensor:
    """Convert 8D combined (pos3+quat4+grip1) to 7D replay (norm_pos3+euler3+norm_grip1).

    The replay buffer stores actions in the same format as offline data:
      [normalized_pos(3), euler_residual(3), normalized_grip(1)]
    """
    n = combined_8d.shape[0]
    pg = torch.cat([combined_8d[:, :3], combined_8d[:, 7:8]], dim=-1)  # (N, 4)
    pg_norm = action_scaler.scale(pg)  # (N, 4) normalized
    out = torch.zeros(n, 7, device=combined_8d.device, dtype=combined_8d.dtype)
    out[:, :3] = pg_norm[:, :3]       # normalized pos
    out[:, 3:6] = residual_7d[:, 3:6] # raw euler residual
    out[:, 6] = pg_norm[:, 3]         # normalized grip
    return out


# ══════════════════════════════════════════════════════════════════
# ActionScaler stats computation (cup: pos+grip from 7D base_action)
# ══════════════════════════════════════════════════════════════════

def _compute_action_stats_from_offline(data_dir: str) -> dict:
    """Compute per-dim min/max for 4D pos+grip actions from offline npz."""
    npz_files = sorted(Path(data_dir).rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    all_pg = []
    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)
        ba = data["obs_base_action"].astype(np.float32)  # (N, 7)
        pg = np.concatenate([ba[:, :3], ba[:, 6:7]], axis=-1)  # (N, 4)
        all_pg.append(pg)

    all_pg = np.concatenate(all_pg, axis=0)
    stats = {
        "min": all_pg.min(axis=0).tolist(),
        "max": all_pg.max(axis=0).tolist(),
    }
    _log(f"Action stats (pos+grip, {len(all_pg)} samples):")
    labels = ["pos_x", "pos_y", "pos_z", "grip"]
    for i, lb in enumerate(labels):
        _log(f"  {lb}: [{stats['min'][i]:.4f}, {stats['max'][i]:.4f}]")
    return stats


def _load_offline_data(
    offline_rb,
    data_dir: str,
    image_keys: list,
    lowdim_keys: list,
    device: str = "cpu",
    reward_relabel: str = "none",
    action_scaler=None,
) -> None:
    """Load offline .npz files into the replay buffer (cup version).

    Args:
        reward_relabel: "none" = use saved reward (clamped to [0,1]),
                        "sparse" = relabel using 'success' field (1.0 at terminal step).
        action_scaler: If provided, normalizes base_action pos+grip and uses as replay action.
    """
    from pathlib import Path as _P

    npz_files = sorted(_P(data_dir).rglob("*.npz"))
    if not npz_files:
        _log(f"WARNING: No .npz files found in {data_dir}")
        return

    import random as _rng
    _rng.shuffle(npz_files)

    max_transitions = offline_rb.storage.max_size
    _log(f"  [offline] reward_relabel={reward_relabel}, data_dir={data_dir}, files={len(npz_files)}")
    _log(f"  [offline] buffer_size={max_transitions}, will stop after buffer is full")

    total_loaded = 0
    _relabel_success_count = 0
    _relabel_total_count = 0

    for npz_path in npz_files:
        if total_loaded >= max_transitions:
            _log(f"  [offline] Buffer full ({total_loaded} >= {max_transitions}), stopping early")
            break
        data = np.load(str(npz_path), allow_pickle=True)
        n = len(data["obs_state"])

        ba_t = torch.from_numpy(data["obs_base_action"].astype(np.float32))

        # Normalize pos+grip if ActionScaler is provided
        if action_scaler is not None:
            pg = torch.cat([ba_t[:, :3], ba_t[:, 6:7]], dim=-1)  # (N, 4)
            pg_norm = action_scaler.scale(pg)
            ba_t = ba_t.clone()
            ba_t[:, :3] = pg_norm[:, :3]
            ba_t[:, 6] = pg_norm[:, 3]
            ba_t[:, 3:6] = 0.0

        obs_td = {
            "observation.state": torch.from_numpy(data["obs_state"].astype(np.float32)),
            "observation.base_action": ba_t,
        }
        if "observation.object_state" in lowdim_keys:
            if "obs_object_state" in data:
                obs_td["observation.object_state"] = torch.from_numpy(
                    data["obs_object_state"].astype(np.float32))
            else:
                # Construct from cup features if available
                cup_pos = torch.from_numpy(data["cup_pos"].astype(np.float32))  # (N, 3)
                cup_quat = torch.from_numpy(data["cup_quat"].astype(np.float32))  # (N, 4)
                uprightness = torch.from_numpy(data["uprightness"].astype(np.float32))  # (N,)
                obs_td["observation.object_state"] = torch.cat(
                    [cup_pos, cup_quat, uprightness.unsqueeze(-1)], dim=-1)  # (N, 8)

        # Build next_obs: shift by 1
        done_t = torch.from_numpy(data["done"].astype(bool))
        next_obs_td = {}
        for k, v in obs_td.items():
            shifted = torch.roll(v, -1, dims=0)
            shifted[-1] = v[-1]
            boundary = done_t.view(-1, *([1] * (v.ndim - 1))).expand_as(v)
            next_obs_td[k] = torch.where(boundary, v, shifted)

        # ── Compute reward ──
        if reward_relabel == "sparse":
            reward_t = torch.zeros(n, dtype=torch.float32)
            _relabel_total_count += n
            if "success" in data and bool(data["success"].flat[0]):
                term_arr = data.get("terminated", data.get("done"))
                term_idxs = np.where(term_arr)[0]
                if len(term_idxs) > 0:
                    reward_t[term_idxs[-1]] = 1.0
                    _relabel_success_count += 1
        else:
            # Use saved reward (dense = uprightness, clamped to [0,1])
            reward_t = torch.from_numpy(data["reward"].astype(np.float32)).clamp(0.0, 1.0)

        # Action for replay: normalized base_action (zero residual in offline demos)
        if action_scaler is not None:
            action_t = ba_t.clone()
            action_t[:, 3:6] = 0.0
        else:
            action_t = torch.from_numpy(data["action"].astype(np.float32))

        # Batch insert into replay buffer
        for i in range(n):
            td = TensorDict({
                "obs": TensorDict({k: v[i] for k, v in obs_td.items()}, batch_size=[]),
                "next": TensorDict({
                    "obs": TensorDict({k: v[i] for k, v in next_obs_td.items()}, batch_size=[]),
                    "done": done_t[i],
                    "reward": reward_t[i],
                }, batch_size=[]),
                "action": action_t[i],
            }, batch_size=[]).unsqueeze(0)
            offline_rb.add(td)
        total_loaded += n

        if total_loaded % 50000 < n:
            _log(f"  ... {total_loaded} transitions loaded so far")

    _log(f"  Loaded {total_loaded} transitions from {len(npz_files)} files")
    if reward_relabel == "sparse":
        sr_pct = _relabel_success_count / max(_relabel_total_count, 1) * 100
        _log(f"  [offline] SPARSE RELABEL: {_relabel_success_count} terminal-success "
             f"out of {_relabel_total_count} ({sr_pct:.4f}%)")
    # Log reward stats
    if total_loaded > 0:
        sample_size = min(1000, len(offline_rb))
        sample = offline_rb.sample(sample_size)
        r = sample["next", "reward"]
        _log(f"  [offline] Reward stats (sample {sample_size}): "
             f"min={r.min().item():.4f}, max={r.max().item():.4f}, "
             f"mean={r.mean().item():.4f}, nonzero={int((r > 0).sum().item())}/{sample_size}")


def _add_transitions(
    obs, next_obs, actions, reward, done,
    device, image_keys, lowdim_keys, num_envs, online_rb,
):
    obs_keys = set(image_keys) | set(lowdim_keys)
    rgb_keys = [k for k in image_keys if "depth" not in k]
    for i in range(num_envs):
        curr_obs_i = {k: v[i] for k, v in obs.items() if k in obs_keys}
        next_obs_i = {k: v[i] for k, v in next_obs.items() if k in obs_keys}
        to_uint8(curr_obs_i, rgb_keys)
        to_uint8(next_obs_i, rgb_keys)

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


# ── Evaluation ──

class AsyncEvaluator:
    """Manages a background eval subprocess with its own MuJoCo env + GR00T."""

    def __init__(self, args, checkpoint_dir: Path, results_dir: Path, action_scaler=None):
        self._args = args
        self._checkpoint_dir = checkpoint_dir
        self._results_dir = results_dir
        self._action_scaler = action_scaler
        self._proc = None
        self._best_success = 0.0
        results_dir.mkdir(parents=True, exist_ok=True)

    def start(self, wandb_run_id: str | None = None):
        """Launch the async eval subprocess."""
        eval_script = str(Path(__file__).parent / "eval_async_mujoco_cup.py")
        a = self._args

        cmd = [
            sys.executable, eval_script,
            "--checkpoint_dir", str(self._checkpoint_dir),
            "--output_dir", str(self._results_dir),
            "--groot_checkpoint", a.groot_checkpoint,
            "--groot_embodiment_tag", a.groot_embodiment_tag,
            "--task_description", a.task_description,
            "--open_loop_horizon", str(a.open_loop_horizon),
            "--residual_pos_scale", str(a.residual_pos_scale),
            "--residual_rot_scale", str(a.residual_rot_scale),
            "--residual_grip_scale", str(a.residual_grip_scale),
            "--ema_alpha", str(a.ema_alpha),
            "--action_scale", str(a.action_scale),
            "--actor_hidden_dim", str(a.actor_hidden_dim),
            "--critic_hidden_dim", str(a.critic_hidden_dim),
            "--max_episode_steps", str(a.max_episode_steps),
            "--reward_type", a.reward_type,
            "--device", a.device,
            "--eval_num_episodes", str(a.eval_num_episodes),
            "--max_eval_envs", str(a.eval_num_envs),
            "--poll_interval_sec", "10",
        ]
        if a.cup_positions_file:
            cmd += ["--cup_positions_file", a.cup_positions_file]
        if a.groot_policy_device:
            cmd += ["--groot_policy_device", a.groot_policy_device]
        if a.save_video:
            cmd += ["--save_video"]
        if a.asymmetric_critic:
            cmd += ["--asymmetric_critic"]
        if a.torch_compile:
            cmd += ["--torch_compile"]
        # ActionScaler forwarding
        if a.use_action_scaler and self._action_scaler is not None:
            cmd += ["--use_action_scaler"]
            cmd += ["--action_scaler_min"] + [str(x) for x in self._action_scaler.action_min.tolist()]
            cmd += ["--action_scaler_max"] + [str(x) for x in self._action_scaler.action_max.tolist()]
            if a.no_action_clamp:
                cmd += ["--no_action_clamp"]
        # W&B
        if a.wandb_mode != "disabled" and wandb_run_id:
            cmd += [
                "--wandb_mode", a.wandb_mode,
                "--wandb_project", a.wandb_project,
                "--wandb_run_id", wandb_run_id,
                "--wandb_name", f"{a.wandb_name or 'train'}_eval",
            ]
            if a.wandb_entity:
                cmd += ["--wandb_entity", a.wandb_entity]

        log_path = self._results_dir / "async_eval.log"
        self._proc = subprocess.Popen(
            cmd,
            stdout=open(str(log_path), "w"),
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        _log(f"[AsyncEval-Cup] Started PID={self._proc.pid}, log={log_path}")

    def trigger_eval(self, agent, global_step: int):
        """Save agent weights for the eval process to pick up."""
        ckpt_path = self._checkpoint_dir / f"eval_request_step{global_step}.pt"
        tmp_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"model": agent.state_dict(), "args": vars(self._args) if hasattr(self._args, "__dict__") else self._args}, tmp_path)
        tmp_path.rename(ckpt_path)
        _log(f"[AsyncEval] Triggered eval at step {global_step}")

    def collect_results(self) -> dict | None:
        """Non-blocking: check for completed eval JSON results."""
        result_files = sorted(self._results_dir.glob("eval_step*.json"))
        if not result_files:
            return None
        latest = result_files[-1]
        try:
            with open(latest) as f:
                data = _json_mod.load(f)
            for rf in result_files:
                try:
                    rf.rename(rf.with_suffix(".done"))
                except Exception:
                    pass
            step = data.get("eval/step", 0)
            sr = data.get("eval/success_rate", 0.0)
            _log(f"[AsyncEval] Got results: step={step} SR={sr:.2%}")
            if sr > self._best_success:
                self._best_success = sr
            return data
        except Exception:
            return None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def shutdown(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            _log("[AsyncEval] Subprocess terminated")


# ── Resume checkpoint ──

def _save_resume_checkpoint(agent, checkpoint_dir, global_step, best_success, args, wandb_run_id=None):
    """Save a full resume checkpoint (atomic write)."""
    ckpt_data = {
        "encoders": agent.encoders.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "encoder_opt": agent.encoder_opt.state_dict(),
        "global_step": global_step,
        "best_success": best_success,
        "wandb_run_id": wandb_run_id,
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }
    for sched_name in ("actor_scheduler", "critic_scheduler", "encoder_scheduler"):
        sched = getattr(agent, sched_name, None)
        if sched is not None:
            ckpt_data[sched_name] = sched.state_dict()
    path = Path(checkpoint_dir) / "resume_latest.pt"
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(ckpt_data, str(tmp_path))
    tmp_path.rename(path)
    _log(f"Resume checkpoint saved: {path} (step={global_step})")


def _load_resume_checkpoint(agent, ckpt_path, device):
    """Load a resume checkpoint and restore full training state."""
    _log(f"Loading resume checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.encoders.load_state_dict(ckpt["encoders"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt["critic_target"])
    agent.actor_target.load_state_dict(ckpt["actor_target"])
    agent.actor_opt.load_state_dict(ckpt["actor_opt"])
    agent.critic_opt.load_state_dict(ckpt["critic_opt"])
    agent.encoder_opt.load_state_dict(ckpt["encoder_opt"])
    for sched_name in ("actor_scheduler", "critic_scheduler", "encoder_scheduler"):
        if sched_name in ckpt:
            sched = getattr(agent, sched_name, None)
            if sched is not None:
                sched.load_state_dict(ckpt[sched_name])
    global_step = ckpt.get("global_step", 0)
    best_success = ckpt.get("best_success", 0.0)
    wandb_run_id = ckpt.get("wandb_run_id", None)
    _log(f"  Restored: step={global_step}, best_sr={best_success:.2%}, wandb_id={wandb_run_id}")
    return global_step, best_success, wandb_run_id


def parse_args():
    p = argparse.ArgumentParser(description="Residual TD3 on MuJoCo Stand Cup + GR00T")
    # Environment
    p.add_argument("--num_envs", type=int, default=27)
    p.add_argument("--cup_positions_file", type=str, default=None,
                   help="JSON with per-episode cup positions (configs/cup_positions.json)")
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--reward_type", type=str, default="dense", choices=["sparse", "dense", "dense_bonus"])
    # GR00T
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--groot_policy_device", type=str, default=None)
    p.add_argument("--task_description", type=str,
                   default="Pick up the cup and stand it upright.")
    p.add_argument("--open_loop_horizon", type=int, default=16)
    # Residual scales
    p.add_argument("--residual_pos_scale", type=float, default=0.02)
    p.add_argument("--residual_rot_scale", type=float, default=0.05)
    p.add_argument("--residual_grip_scale", type=float, default=0.004)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument("--torch_compile", action="store_true", default=False,
                   help="Apply torch.compile to GR00T model for inference acceleration")
    # Algorithm
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--learning_starts", type=int, default=1_000)
    p.add_argument("--critic_warmup_steps", type=int, default=2_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=500_000)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--random_action_noise_scale", type=float, default=0.05)
    p.add_argument("--stddev_max", type=float, default=0.05)
    p.add_argument("--stddev_min", type=float, default=0.05)
    p.add_argument("--action_scale", type=float, default=0.1)
    p.add_argument("--action_l2_reg", type=float, default=1.0)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--num_updates_per_iteration", type=int, default=4)
    p.add_argument("--update_every_n_steps", type=int, default=1)
    # Asymmetric critic
    p.add_argument("--asymmetric_critic", action="store_true")
    p.add_argument("--actor_hidden_dim", type=int, default=256)
    p.add_argument("--critic_hidden_dim", type=int, default=256)
    # Offline data
    p.add_argument("--offline_data_dir", type=str, default=None)
    p.add_argument("--offline_fraction", type=float, default=0.75)
    p.add_argument("--offline_reward_relabel", type=str, default="none", choices=["none", "sparse"])
    p.add_argument("--offline_pretrain_only", action="store_true")
    p.add_argument("--target_tau", type=float, default=0.005)
    p.add_argument("--critic_grad_clip_norm", type=float, default=None)
    # ActionScaler
    p.add_argument("--use_action_scaler", action="store_true")
    p.add_argument("--no_action_clamp", action="store_true")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default="outputs/cup_rl")
    p.add_argument("--eval_interval", type=int, default=2_000)
    p.add_argument("--eval_num_episodes", type=int, default=1)
    p.add_argument("--eval_num_envs", type=int, default=27)
    p.add_argument("--debug_zero_residual", action="store_true")
    p.add_argument("--save_video", action="store_true")
    # Checkpoint & Resume
    p.add_argument("--checkpoint_interval", type=int, default=10000)
    p.add_argument("--resume_checkpoint", type=str, default=None)
    # W&B
    p.add_argument("--no_warmup", action="store_true")
    p.add_argument("--async_eval", action="store_true")
    p.add_argument("--wandb_mode", type=str, default="offline")
    p.add_argument("--wandb_project", type=str, default="mujoco-franka-cup-residual-td3")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--no_offline_cache", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _log(f"[CONFIG] reward_type={args.reward_type}, gamma={args.gamma}, action_scale={args.action_scale}")
    _log(f"[CONFIG] offline_fraction={args.offline_fraction}, offline_reward_relabel={args.offline_reward_relabel}")
    _log(f"[CONFIG] action_l2_reg={args.action_l2_reg}, target_tau={args.target_tau}")
    _log(f"[CONFIG] actor_lr={args.actor_lr}, critic_lr={args.critic_lr}, n_step={args.n_step}")

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Output dirs
    outputs_dir = Path(args.output_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outputs_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Seed: {args.seed}")
    _log(f"Device: {device}")
    _log(f"Output: {outputs_dir}")

    # ── Load cup positions ──
    # Default cup_positions_path: MSRA/stand_cup/cup_positions.json (used by MuJoCoVecEnvCup)
    cup_positions_path = args.cup_positions_file  # None → uses default in MuJoCoVecEnvCup
    
    # Determine available episodes
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import load_cup_positions
    _cup_data = load_cup_positions(cup_positions_path)
    _available_eps = len(_cup_data)
    
    episode_ids = list(range(min(args.num_envs, _available_eps)))
    if args.num_envs > _available_eps:
        _log(f"WARNING: num_envs={args.num_envs} > positions={_available_eps}, clamping")
        args.num_envs = len(episode_ids)
    
    _log(f"Cup positions: {len(episode_ids)} episodes available")

    # ── Create MuJoCo environment ──
    _log("Creating MuJoCo Stand Cup environment...")
    mujoco_env = MuJoCoVecEnv(
        num_envs=args.num_envs,
        episode_ids=episode_ids,
        cup_positions_path=cup_positions_path,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        device=args.device,
    )
    _log(f"MuJoCo Cup env: {args.num_envs} envs")

    # ── Build ActionScaler ──
    _action_scaler = None
    if args.use_action_scaler:
        if not args.offline_data_dir:
            raise ValueError("--use_action_scaler requires --offline_data_dir")
        _log("Computing action stats from offline data for ActionScaler...")
        _action_stats = _compute_action_stats_from_offline(args.offline_data_dir)
        _action_scaler = ActionScaler(
            action_min=torch.tensor(_action_stats["min"], dtype=torch.float32),
            action_max=torch.tensor(_action_stats["max"], dtype=torch.float32),
            action_scale=args.action_scale,
            device="cpu",
            no_clamp=args.no_action_clamp,
        )
        _log(f"ActionScaler created (action_scale={args.action_scale}, no_clamp={args.no_action_clamp})")

    # Wrap with residual + GR00T
    _log("Creating residual wrapper with GR00T policy...")
    policy_device = args.groot_policy_device or args.device
    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=policy_device,
        task_description=args.task_description,
        open_loop_horizon=args.open_loop_horizon,
        residual_pos_scale=args.residual_pos_scale,
        residual_rot_scale=args.residual_rot_scale,
        residual_grip_scale=args.residual_grip_scale,
        ema_alpha=args.ema_alpha,
        torch_compile=args.torch_compile,
    )
    _log("Environment ready.")

    # ── Dimensions (state-only) ──
    num_envs = args.num_envs
    image_keys = []
    object_state_dim = env.observation_space["observation.object_state"].shape[1]  # 8
    lowdim_keys = ["observation.state", "observation.base_action", "observation.object_state"]

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim  # 7

    _log(f"State-only mode (no images for actor/critic)")
    _log(f"lowdim_dim={lowdim_dim}, object_state_dim={object_state_dim}, action_dim={action_dim}")

    # ── QAgent (state-only) ──
    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor_lr = args.actor_lr
    cfg.agent.critic_lr = args.critic_lr
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.action_l2_reg_weight = args.action_l2_reg
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim
    if args.target_tau is not None:
        cfg.agent.critic_target_tau = args.target_tau
    if args.critic_grad_clip_norm is not None:
        cfg.agent.critic_grad_clip_norm = args.critic_grad_clip_norm

    agent = QAgent(
        obs_shape=(3, 84, 84),  # dummy, not used in state-only
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=[],  # state-only
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=False,
    )

    # ── Replay buffer ──
    online_batch_size = int(args.batch_size * (1 - args.offline_fraction)) if args.offline_fraction > 0 else args.batch_size
    offline_batch_size = args.batch_size - online_batch_size

    online_rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=args.buffer_size, device="cpu"),
        transform=MultiStepTransform(n_steps=args.n_step, gamma=args.gamma),
        pin_memory=True,
        prefetch=4,
        batch_size=max(online_batch_size, 1),
    )

    # Offline buffer
    offline_rb = None
    if args.offline_fraction > 0 and not args.offline_data_dir:
        raise ValueError(
            f"--offline_fraction={args.offline_fraction} > 0 but --offline_data_dir is not set."
        )
    if args.offline_data_dir and args.offline_fraction > 0:
        offline_rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=args.buffer_size, device="cpu"),
            transform=MultiStepTransform(n_steps=args.n_step, gamma=args.gamma),
            pin_memory=True,
            prefetch=4,
            batch_size=max(offline_batch_size, 1),
        )
        _log(f"Offline buffer created (dir={args.offline_data_dir}, fraction={args.offline_fraction})")

        # ── Buffer cache ──
        import hashlib, json as _json
        _cache_key_dict = {
            "offline_data_dir": str(Path(args.offline_data_dir).resolve()),
            "n_step": args.n_step,
            "gamma": args.gamma,
            "buffer_size": args.buffer_size,
            "image_keys": sorted(image_keys),
            "lowdim_keys": sorted(lowdim_keys),
            "reward_relabel": args.offline_reward_relabel,
            "use_action_scaler": args.use_action_scaler,
            "action_scale": args.action_scale if args.use_action_scaler else None,
        }
        _cache_hash = hashlib.sha256(_json.dumps(_cache_key_dict, sort_keys=True).encode()).hexdigest()[:16]
        _cache_root = Path(__file__).resolve().parents[3] / "buffer_cache"
        _cache_dir = _cache_root / f"offline_cup_{_cache_hash}"
        _loaded_from_cache = False

        if not args.no_offline_cache and _cache_dir.exists() and (_cache_dir / "cache_meta.json").exists():
            try:
                _t0 = time.time()
                offline_rb.loads(str(_cache_dir))
                _log(f"Loaded offline cache (hash={_cache_hash}) in {time.time()-_t0:.1f}s: {len(offline_rb)} transitions")
                _loaded_from_cache = True
            except Exception as e:
                _log(f"Cache load failed ({e}), re-populating...")
                import shutil
                shutil.rmtree(_cache_dir, ignore_errors=True)

        if not _loaded_from_cache:
            _load_offline_data(
                offline_rb=offline_rb,
                data_dir=args.offline_data_dir,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                device="cpu",
                reward_relabel=args.offline_reward_relabel,
                action_scaler=_action_scaler,
            )
            _log(f"Offline buffer populated: {len(offline_rb)} transitions")
            if not args.no_offline_cache:
                try:
                    _cache_dir.mkdir(parents=True, exist_ok=True)
                    _t0 = time.time()
                    offline_rb.dumps(str(_cache_dir))
                    _meta = {**_cache_key_dict, "num_transitions": len(offline_rb)}
                    (_cache_dir / "cache_meta.json").write_text(_json.dumps(_meta, indent=2))
                    _log(f"Saved offline cache (hash={_cache_hash}) in {time.time()-_t0:.1f}s")
                except Exception as e:
                    _log(f"Warning: failed to save cache: {e}")

    # ── Resume checkpoint ──
    _resume_step = 0
    _resume_best = 0.0
    _resume_wandb_id = None
    if args.resume_checkpoint:
        _resume_step, _resume_best, _resume_wandb_id = _load_resume_checkpoint(
            agent, args.resume_checkpoint, device,
        )

    # ── W&B ──
    _wb = wandb
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mujoco_cup_td3_seed{args.seed}"
    if args.wandb_name:
        run_name = f"{args.wandb_name}__{run_name}"
    if _wb is not None:
        try:
            _wb_kwargs = dict(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                mode=args.wandb_mode,
                config=vars(args),
                reinit=True,
            )
            if _resume_wandb_id:
                _wb_kwargs["id"] = _resume_wandb_id
                _wb_kwargs["resume"] = "must"
            _wb.init(**_wb_kwargs)
            _wb.define_metric("eval/*", step_metric="eval/step")
        except Exception as e:
            _log(f"[WARN] wandb.init() failed: {e}")
            _wb = None

    # ── Async eval setup ──
    _async_eval = None
    if args.async_eval:
        _async_eval_dir = outputs_dir / "async_eval_results"
        _async_eval = AsyncEvaluator(args, checkpoint_dir, _async_eval_dir, action_scaler=_action_scaler)
        _wandb_run_id = _wb.run.id if (_wb is not None and _wb.run is not None) else None
        _async_eval.start(wandb_run_id=_wandb_run_id)
        _log("Async eval subprocess started")

    # ══════════════════════════════════════════════════════════════════
    # Warm-up: fill buffer with base-policy + noise exploration
    # ══════════════════════════════════════════════════════════════════
    if args.no_warmup:
        _log("Skipping warmup (--no_warmup)")
        args.learning_starts = 0
        args.critic_warmup_steps = 0
    _log(f"Warm-up: collecting {args.learning_starts} transitions (base policy + noise)...")
    obs, _ = env.reset()
    warmup_transitions = 0
    warmup_ep_count = 0

    while warmup_transitions < args.learning_starts:
        noise = (torch.rand((num_envs, action_dim), device=device) * 2 - 1) * args.random_action_noise_scale
        next_obs, reward, terminated, truncated, info = env.step(noise)
        done = terminated | truncated

        _replay_action = _to_replay_action_7d(info["scaled_action"], noise, _action_scaler) if _action_scaler is not None else noise
        _add_transitions(
            obs=obs, next_obs=next_obs, actions=_replay_action,
            reward=reward, done=done, device=device,
            image_keys=image_keys, lowdim_keys=lowdim_keys,
            num_envs=num_envs, online_rb=online_rb,
        )
        warmup_transitions += num_envs

        if done.any():
            warmup_ep_count += done.sum().item()

        obs = next_obs

        if warmup_transitions % 200 == 0:
            _log(f"  warmup: {warmup_transitions}/{args.learning_starts} transitions, {warmup_ep_count} episodes")

    _log(f"Warm-up done: {warmup_transitions} transitions, {warmup_ep_count} episodes")

    # ══════════════════════════════════════════════════════════════════
    # Critic warmup (critic-only updates)
    # ══════════════════════════════════════════════════════════════════
    if args.critic_warmup_steps > 0:
        _log(f"Critic warmup: {args.critic_warmup_steps} updates...")
        for i in range(args.critic_warmup_steps):
            batch = online_rb.sample(args.batch_size).to(device, non_blocking=True)
            if batch.ndim > 1:
                batch = batch.squeeze(1)
            metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)
            if i % 200 == 0:
                cl = metrics.get("train/critic_loss", 0)
                _log(f"  critic warmup: {i}/{args.critic_warmup_steps} loss={cl:.4f}")
        _log("Critic warmup done.")

    # ══════════════════════════════════════════════════════════════════
    # Main training loop
    # ══════════════════════════════════════════════════════════════════
    _log(f"Training for {args.total_timesteps} steps...")
    obs, _ = env.reset()
    global_step = _resume_step
    episode_count = 0
    best_success = _resume_best
    if global_step > 0:
        _log(f"Resuming from step {global_step} (best_sr={best_success:.2%})")
    train_start = time.time()
    _timers = {"env_step": [], "agent_act": [], "grad_update": [], "replay_sample": [], "buffer_store": []}
    ep_cum_reward = torch.zeros(num_envs, device=device)
    ep_step_counter = torch.zeros(num_envs, device=device, dtype=torch.long)

    while global_step <= args.total_timesteps:
        # ── (1) Collect transition ──
        frac = min(global_step / max(args.total_timesteps, 1), 1.0)
        stddev = args.stddev_max + (args.stddev_min - args.stddev_max) * frac

        _t0 = time.perf_counter()
        with torch.no_grad(), utils.eval_mode(agent):
            residual_action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)
        _timers["agent_act"].append(time.perf_counter() - _t0)

        if args.debug_zero_residual:
            residual_action = torch.zeros_like(residual_action)

        _t0 = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env.step(residual_action)
        _timers["env_step"].append(time.perf_counter() - _t0)
        done = terminated | truncated

        # Episode bookkeeping
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

            ep_cum_reward[done_mask] = 0.0
            ep_step_counter[done_mask] = 0

        # Store transition
        _replay_action = _to_replay_action_7d(info["scaled_action"], residual_action, _action_scaler) if _action_scaler is not None else residual_action
        _t0 = time.perf_counter()
        _add_transitions(
            obs=obs, next_obs=next_obs, actions=_replay_action,
            reward=reward, done=done, device=device,
            image_keys=image_keys, lowdim_keys=lowdim_keys,
            num_envs=num_envs, online_rb=online_rb,
        )
        _timers["buffer_store"].append(time.perf_counter() - _t0)
        obs = next_obs

        # ── (2) Update agent ──
        if len(online_rb) >= max(online_batch_size, 1) and global_step % args.update_every_n_steps == 0:
            _t0 = time.perf_counter()
            for _ui in range(args.num_updates_per_iteration):
                batch = online_rb.sample(max(online_batch_size, 1)).to(device, non_blocking=True)
                if batch.ndim > 1:
                    batch = batch.squeeze(1)
                _use_offline = (offline_rb is not None and len(offline_rb) > 0
                                and offline_batch_size > 0
                                and not (args.offline_pretrain_only and global_step >= args.critic_warmup_steps))
                if _use_offline:
                    obatch = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
                    if obatch.ndim > 1:
                        obatch = obatch.squeeze(1)
                    if "_priority" in batch.keys():
                        batch = batch.exclude("_priority")
                    if "_priority" in obatch.keys():
                        obatch = obatch.exclude("_priority")
                    batch = torch.cat([batch, obatch], dim=0)
                update_actor = global_step >= args.critic_warmup_steps
                metrics = agent.update(
                    batch, stddev=stddev,
                    update_actor=update_actor,
                    bc_batch=None, ref_agent=agent,
                )

            _timers["grad_update"].append(time.perf_counter() - _t0)

            # Log
            if global_step % 100 == 0 and _wb is not None and _wb.run is not None:
                log_dict = {f"train/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                log_dict["training/global_step"] = global_step
                log_dict["training/buffer_size"] = len(online_rb)
                _wb.log(log_dict, step=global_step)

        # ── (3) Periodic logging ──
        if global_step % 500 == 0:
            ba = obs["observation.base_action"][0].detach().cpu().numpy()
            ra = residual_action[0].detach().cpu().numpy() if residual_action is not None else np.zeros(7)
            rw = reward[0].item()
            elapsed = time.time() - train_start
            steps_per_sec = max(1, global_step) / max(1, elapsed)
            eta_h = (args.total_timesteps - global_step) / max(0.01, steps_per_sec) / 3600

            _timing_strs = []
            _timing_wb = {}
            for _tk, _tv in _timers.items():
                if _tv:
                    _avg_ms = sum(_tv[-500:]) / len(_tv[-500:]) * 1000
                    _timing_strs.append(f"{_tk}={_avg_ms:.1f}ms")
                    _timing_wb[f"timing/{_tk}_avg_ms"] = _avg_ms
            _timing_line = " ".join(_timing_strs)

            _log(f"step={global_step}/{args.total_timesteps} reward={rw:.4f} eps={episode_count} "
                 f"speed={steps_per_sec:.1f}step/s ETA={eta_h:.1f}h "
                 f"res_mean={ra[:6].mean():.5f}")
            if _timing_line:
                _log(f"  timing: {_timing_line}")

            if _wb is not None and _wb.run is not None and _timing_wb:
                _wb.log(_timing_wb, step=global_step)

        # ── (4) Evaluation ──
        if global_step % args.eval_interval == 0:
            if _async_eval is not None:
                _async_eval.trigger_eval(agent, global_step)

        # ── Collect async eval results ──
        if _async_eval is not None:
            async_result = _async_eval.collect_results()
            if async_result is not None:
                sr = async_result.get("eval/success_rate", 0.0)
                if _wb is not None and _wb.run is not None:
                    _wb.log(async_result)
                if sr >= best_success:
                    best_success = sr
                    ckpt_path = checkpoint_dir / "best.pt"
                    torch.save({"model": agent.state_dict(), "args": vars(args)}, ckpt_path)
                    _log(f"  Saved best checkpoint: {ckpt_path}")

        # ── Periodic resume checkpoint ──
        if args.checkpoint_interval > 0 and global_step > 0 and global_step % args.checkpoint_interval == 0:
            _wandb_id = _wb.run.id if (_wb is not None and _wb.run is not None) else None
            _save_resume_checkpoint(agent, checkpoint_dir, global_step, best_success, args, _wandb_id)

        global_step += 1

    # ── Final ──
    if _async_eval is not None:
        _async_eval.trigger_eval(agent, global_step)
        _log("Final eval triggered (async). Waiting up to 300s for result...")
        for _wait in range(60):
            time.sleep(5)
            result = _async_eval.collect_results()
            if result is not None:
                sr = result.get("eval/success_rate", 0.0)
                _log(f"Final async eval: SR={sr:.2%}")
                if _wb is not None and _wb.run is not None:
                    _wb.log(result, step=global_step)
                break
        _async_eval.shutdown()

    # Save final checkpoint
    final_ckpt = checkpoint_dir / f"final_step{global_step}.pt"
    torch.save({"model": agent.state_dict(), "args": vars(args)}, final_ckpt)
    _log(f"Final checkpoint: {final_ckpt}")

    if _wb is not None and _wb.run is not None:
        _wb.finish()

    env.close()
    _log("Training complete.")


if __name__ == "__main__":
    main()

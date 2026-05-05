"""
Unified Residual TD3 training on MuJoCo + GR00T.

Supports all tasks via --task flag: cup, pnp, lift, stack.
Includes: ActionScaler, chunk_sync, replay buffer fix, buffer cache.

Launch:
    python resfit/rl_finetuning/scripts/train_residual_td3_unified.py \
        --task cup --groot_checkpoint ~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-100000 \
        --use_action_scaler --chunk_sync --num_envs 27 --total_timesteps 500000
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

# ── Prevent deepspeed import crash ──
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

# ── Patch huggingface_hub repo_id validator ──
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
torch.backends.cuda.enable_cudnn_sdp(False)
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.normalization import ActionScaler
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified
from resfit.rl_finetuning.configs.task_configs import get_task_config, TaskConfig

try:
    import wandb
except ImportError:
    wandb = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def _log(msg: str) -> None:
    print(f"[unified-td3] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
# ActionScaler helpers
# ══════════════════════════════════════════════════════════════════

def _to_replay_action_7d(obs_base_action: torch.Tensor, residual_7d: torch.Tensor) -> torch.Tensor:
    """Compute buffer action = clamp(base_norm + residual, -1, 1).
    
    This matches exactly what critic target and actor loss compute,
    ensuring the critic trains on the same distribution it evaluates.
    
    Args:
        obs_base_action: normalized 7D base action from obs [norm_pos3, 0_euler3, norm_grip1]
        residual_7d: actor output (or noise) in [-1,1] range
    Returns:
        7D combined action for replay buffer
    """
    return torch.clamp(obs_base_action + residual_7d, -1.0, 1.0)


def _compute_action_stats_from_offline(data_dir: str) -> dict:
    """Compute per-dim min/max for 4D pos+grip actions from offline npz."""
    npz_files = sorted(Path(data_dir).rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    all_pg = []
    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)
        ba = data["obs_base_action"].astype(np.float32)
        pg = np.concatenate([ba[:, :3], ba[:, 6:7]], axis=-1)
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


# ══════════════════════════════════════════════════════════════════
# Offline data loading
# ══════════════════════════════════════════════════════════════════

def _load_offline_data(
    offline_rb,
    data_dir: str,
    image_keys: list,
    lowdim_keys: list,
    device: str = "cpu",
    reward_relabel: str = "none",
    action_scaler=None,
    reward_config_path: str | None = None,
) -> None:
    """Load offline .npz files into the replay buffer."""
    npz_files = sorted(Path(data_dir).rglob("*.npz"))
    if not npz_files:
        _log(f"WARNING: No .npz files found in {data_dir}")
        return

    import random as _rng
    _rng.shuffle(npz_files)

    # Load reward config for feature-based computation (stack)
    _reward_cfg = None
    if reward_relabel == "computed" or reward_config_path:
        from resfit.rl_finetuning.rewards.stack_reward import (
            compute_reward_from_features,
            load_reward_config,
        )
        _reward_cfg = load_reward_config(reward_config_path)

    max_transitions = offline_rb.storage.max_size
    _log(f"  [offline] reward_relabel={reward_relabel}, data_dir={data_dir}, files={len(npz_files)}")
    _log(f"  [offline] buffer_size={max_transitions}")
    if reward_config_path:
        _log(f"  [offline] reward_config={reward_config_path}")

    total_loaded = 0
    _relabel_success_count = 0
    _relabel_total_count = 0

    for npz_path in npz_files:
        if total_loaded >= max_transitions:
            _log(f"  [offline] Buffer full ({total_loaded}), stopping")
            break
        data = np.load(str(npz_path), allow_pickle=True)
        n = len(data["obs_state"])

        ba_t = torch.from_numpy(data["obs_base_action"].astype(np.float32))

        # Normalize pos+grip if ActionScaler is provided
        if action_scaler is not None:
            pg = torch.cat([ba_t[:, :3], ba_t[:, 6:7]], dim=-1)
            pg_norm = action_scaler.scale(pg)
            ba_t = ba_t.clone()
            ba_t[:, :3] = pg_norm[:, :3]
            ba_t[:, 6] = pg_norm[:, 3]
            ba_t[:, 3:6] = 0.0

        obs_td = {
            "observation.state": torch.from_numpy(data["obs_state"].astype(np.float32)),
            "observation.base_action": ba_t,
        }
        # Object state (task-specific)
        if "observation.object_state" in lowdim_keys:
            if "obs_object_state" in data:
                obs_td["observation.object_state"] = torch.from_numpy(
                    data["obs_object_state"].astype(np.float32))
            elif "cup_pos" in data:
                # Cup: construct from cup features
                cup_pos = torch.from_numpy(data["cup_pos"].astype(np.float32))
                cup_quat = torch.from_numpy(data["cup_quat"].astype(np.float32))
                uprightness = torch.from_numpy(data["uprightness"].astype(np.float32))
                obs_td["observation.object_state"] = torch.cat(
                    [cup_pos, cup_quat, uprightness.unsqueeze(-1)], dim=-1)

        # Build next_obs
        done_t = torch.from_numpy(data["done"].astype(bool))
        next_obs_td = {}
        for k, v in obs_td.items():
            shifted = torch.roll(v, -1, dims=0)
            shifted[-1] = v[-1]
            boundary = done_t.view(-1, *([1] * (v.ndim - 1))).expand_as(v)
            next_obs_td[k] = torch.where(boundary, v, shifted)

        # Reward
        has_saved_reward = "reward" in data
        use_features = (
            reward_relabel == "computed"
            or (not has_saved_reward and reward_relabel != "sparse")
        )

        if reward_relabel == "sparse":
            reward_t = torch.zeros(n, dtype=torch.float32)
            _relabel_total_count += n
            if "success" in data and bool(data["success"].flat[0]):
                term_arr = data.get("terminated", data.get("done"))
                term_idxs = np.where(term_arr)[0]
                if len(term_idxs) > 0:
                    reward_t[term_idxs[-1]] = 1.0
                    _relabel_success_count += 1
        elif use_features and _reward_cfg is not None:
            # Compute reward from raw features (e.g. stack task)
            feature_keys = ["tcp_white_dist", "white_to_green_top_3d", "white_green_xy_dist",
                            "white_z", "green_z", "grasped", "white_green_contact"]
            missing = [k for k in feature_keys if k not in data]
            if missing:
                _log(f"    WARNING: {npz_path.name} missing features {missing}, skipping")
                continue
            features = {k: data[k] for k in feature_keys}
            features["done"] = data["done"].astype(bool)
            reward_np = compute_reward_from_features(features, config=_reward_cfg, use_next_step=True)
            reward_t = torch.from_numpy(reward_np).clamp(0.0, 1.0)
        else:
            reward_t = torch.from_numpy(data["reward"].astype(np.float32)).clamp(0.0, 1.0)

        # Action for replay: offline has no residual, so action = base_norm (= base + 0)
        # This is consistent with online: action = clamp(base_norm + residual)
        if action_scaler is not None:
            action_t = ba_t.clone()
            action_t[:, 3:6] = 0.0  # euler=0 since offline has no rotation residual
        else:
            action_t = torch.from_numpy(data["action"].astype(np.float32))

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
            _log(f"  ... {total_loaded} transitions loaded")

    _log(f"  Loaded {total_loaded} transitions from {len(npz_files)} files")
    if reward_relabel == "sparse":
        sr_pct = _relabel_success_count / max(_relabel_total_count, 1) * 100
        _log(f"  [offline] SPARSE RELABEL: {_relabel_success_count} success ({sr_pct:.2f}%)")
    if total_loaded > 0:
        sample_size = min(1000, len(offline_rb))
        sample = offline_rb.sample(sample_size)
        r = sample["next", "reward"]
        _log(f"  [offline] Reward stats: min={r.min().item():.4f}, max={r.max().item():.4f}, mean={r.mean().item():.4f}")


def _add_transitions(
    obs, next_obs, actions, reward, done,
    device, image_keys, lowdim_keys, num_envs, online_rb,
    store_mask=None,
):
    obs_keys = set(image_keys) | set(lowdim_keys)
    rgb_keys = [k for k in image_keys if "depth" not in k]
    for i in range(num_envs):
        if store_mask is not None and not store_mask[i]:
            continue
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


# ══════════════════════════════════════════════════════════════════
# Async Evaluator
# ══════════════════════════════════════════════════════════════════

class AsyncEvaluator:
    """Background eval subprocess."""

    def __init__(self, args, checkpoint_dir: Path, results_dir: Path, action_scaler=None):
        self._args = args
        self._checkpoint_dir = checkpoint_dir
        self._results_dir = results_dir
        self._action_scaler = action_scaler
        self._proc = None
        self._best_success = 0.0
        results_dir.mkdir(parents=True, exist_ok=True)

    def start(self, wandb_run_id: str | None = None):
        eval_script = str(Path(__file__).parent / "eval_async_unified.py")
        a = self._args

        cmd = [
            sys.executable, eval_script,
            "--task", a.task,
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
            "--eval_num_envs", str(a.eval_num_envs),
            "--poll_interval_sec", "10",
        ]
        if a.groot_policy_device:
            cmd += ["--groot_policy_device", a.groot_policy_device]
        if a.save_video:
            cmd += ["--save_video"]
        if a.torch_compile:
            cmd += ["--torch_compile"]
        if a.chunk_sync:
            cmd += ["--chunk_sync"]
        # Task-specific eval args
        if hasattr(a, 'cup_positions_file') and a.cup_positions_file:
            cmd += ["--cup_positions_file", a.cup_positions_file]
        if hasattr(a, 'eval_positions_file') and a.eval_positions_file:
            cmd += ["--episode_positions_file", a.eval_positions_file]  # eval uses its own positions
        elif hasattr(a, 'episode_positions_file') and a.episode_positions_file:
            cmd += ["--episode_positions_file", a.episode_positions_file]
        if hasattr(a, 'scene_xml') and a.scene_xml:
            cmd += ["--scene_xml", a.scene_xml]
        if hasattr(a, 'success_threshold') and a.success_threshold is not None:
            cmd += ["--success_threshold", str(a.success_threshold)]
        # ActionScaler
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
        _log(f"[AsyncEval] Started PID={self._proc.pid}, log={log_path}")

    def trigger_eval(self, agent, global_step: int):
        ckpt_path = self._checkpoint_dir / f"eval_request_step{global_step}.pt"
        tmp_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"model": agent.state_dict(), "args": vars(self._args)}, tmp_path)
        tmp_path.rename(ckpt_path)
        _log(f"[AsyncEval] Triggered eval at step {global_step}")

    def collect_results(self) -> dict | None:
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
            _log(f"[AsyncEval] Results: step={step} SR={sr:.2%}")
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


# ══════════════════════════════════════════════════════════════════
# Resume checkpoint
# ══════════════════════════════════════════════════════════════════

def _save_resume_checkpoint(agent, checkpoint_dir, global_step, best_success, args, wandb_run_id=None):
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
    _log(f"  Restored: step={global_step}, best_sr={best_success:.2%}")
    return global_step, best_success, wandb_run_id


# ══════════════════════════════════════════════════════════════════
# Task-specific environment creation
# ══════════════════════════════════════════════════════════════════

def _create_env(args, task_cfg: TaskConfig):
    """Create the task-specific VecEnv based on --task."""
    # Dynamic import of VecEnv class
    import importlib
    mod = importlib.import_module(task_cfg.vec_env_module)
    VecEnvClass = getattr(mod, task_cfg.vec_env_class)

    if task_cfg.name == "cup":
        from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import load_cup_positions
        cup_data = load_cup_positions(getattr(args, 'cup_positions_file', None))
        episode_ids = list(range(min(args.num_envs, len(cup_data))))
        if args.num_envs > len(cup_data):
            _log(f"WARNING: num_envs={args.num_envs} > positions={len(cup_data)}, clamping")
            args.num_envs = len(episode_ids)
        return VecEnvClass(
            num_envs=args.num_envs,
            episode_ids=episode_ids,
            cup_positions_path=getattr(args, 'cup_positions_file', None),
            max_episode_steps=args.max_episode_steps,
            reward_type=args.reward_type,
            device=args.device,
        )

    elif task_cfg.name == "pnp":
        # Parse positions
        cube_positions = [args.cube_pos] * args.num_envs
        if args.perturb_table:
            import json as _json
            with open(args.perturb_table) as f:
                ptable = _json.load(f)
            base_pos = ptable["base_pos"]
            envs_cfg = ptable["envs"]
            cube_positions = [
                [base_pos[0] + e["dx"], base_pos[1] + e["dy"], base_pos[2]]
                for e in envs_cfg
            ]
            args.num_envs = len(cube_positions)

        random_cube_range = None
        if args.random_cube_range:
            import json as _json2
            _parsed = _json2.loads(args.random_cube_range)
            def _cvt(r):
                if r is None: return None
                return {"dx": (r["dx"][0]/100.0, r["dx"][1]/100.0),
                        "dy": (r["dy"][0]/100.0, r["dy"][1]/100.0),
                        "yaw": tuple(r.get("yaw", [0,0]))}
            if isinstance(_parsed, list):
                random_cube_range = [_cvt(item) for item in _parsed]
            else:
                random_cube_range = _cvt(_parsed)

        random_bowl_range = None
        if args.random_bowl_range:
            import json as _json3
            _parsed = _json3.loads(args.random_bowl_range)
            def _cvt_bowl(r):
                if r is None: return None
                return {"dx": (r["dx"][0]/100.0, r["dx"][1]/100.0),
                        "dy": (r["dy"][0]/100.0, r["dy"][1]/100.0)}
            if isinstance(_parsed, list):
                random_bowl_range = [_cvt_bowl(item) for item in _parsed]
            else:
                random_bowl_range = _cvt_bowl(_parsed)

        # Auto-detect camera calibration
        _calib_path = getattr(args, 'calib_path', None)
        _use_calibrated_wrist = getattr(args, 'use_calibrated_wrist', False)
        if _calib_path is None:
            ckpt_lower = args.groot_checkpoint.lower()
            if "66ep" in ckpt_lower or "100ep" in ckpt_lower:
                _calib_66ep = str(Path(__file__).resolve().parents[4] / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml")
                if Path(_calib_66ep).exists():
                    _calib_path = _calib_66ep
                    _use_calibrated_wrist = True

        return VecEnvClass(
            num_envs=args.num_envs,
            cube_positions=cube_positions,
            cube_yaw_deg=getattr(args, 'cube_yaw', 0.0),
            random_cube_range=random_cube_range,
            random_bowl_range=random_bowl_range,
            scene_xml=getattr(args, 'scene_xml', None),
            calib_path=_calib_path,
            use_calibrated_wrist=_use_calibrated_wrist,
            max_episode_steps=args.max_episode_steps,
            success_threshold=getattr(args, 'success_threshold', task_cfg.success_threshold),
            reward_type=args.reward_type,
            device=args.device,
            rl_img_size=84,
            bowl_positions=[args.bowl_pos],
            episode_positions_file=getattr(args, 'episode_positions_file', None),
        )

    elif task_cfg.name == "lift":
        cube_positions = [args.cube_pos] * args.num_envs
        random_cube_range = None
        if args.random_cube_range:
            import json as _json4
            _parsed = _json4.loads(args.random_cube_range)
            def _cvt(r):
                if r is None: return None
                return {"dx": (r["dx"][0]/100.0, r["dx"][1]/100.0),
                        "dy": (r["dy"][0]/100.0, r["dy"][1]/100.0),
                        "yaw": tuple(r.get("yaw", [0,0]))}
            random_cube_range = _cvt(_parsed) if not isinstance(_parsed, list) else [_cvt(x) for x in _parsed]

        return VecEnvClass(
            num_envs=args.num_envs,
            cube_positions=cube_positions,
            cube_yaw_deg=getattr(args, 'cube_yaw', 0.0),
            random_cube_range=random_cube_range,
            scene_xml=getattr(args, 'scene_xml', None),
            max_episode_steps=args.max_episode_steps,
            reward_type=args.reward_type,
            device=args.device,
        )

    elif task_cfg.name == "stack":
        # Stack has white + green cube positions from JSON file
        white_positions = None
        green_positions = None
        pos_file = getattr(args, 'episode_positions_file', None)
        if pos_file:
            import json as _json
            with open(pos_file) as _f:
                _pos_data = _json.load(_f)
            white_positions = [p["white_cube_pos"] for p in _pos_data]
            green_positions = [p["green_cube_pos"] for p in _pos_data]
            if args.num_envs != len(white_positions):
                _log(f"WARNING: num_envs={args.num_envs} != positions={len(white_positions)}, using positions count")
                args.num_envs = len(white_positions)

        # Eval positions (separate file for async eval)
        eval_pos_file = getattr(args, 'eval_positions_file', None)

        return VecEnvClass(
            num_envs=args.num_envs,
            scene_xml=getattr(args, 'scene_xml', None),
            max_episode_steps=args.max_episode_steps,
            reward_type=args.reward_type,
            device=args.device,
            white_cube_positions=white_positions,
            green_cube_positions=green_positions,
        )

    elif task_cfg.name == "drawer":
        return VecEnvClass(
            num_envs=args.num_envs,
            active_drawers=getattr(args, 'active_drawers', None),
            scene_xml=getattr(args, 'scene_xml', None),
            max_episode_steps=args.max_episode_steps,
            reward_type=args.reward_type,
            device=args.device,
            rl_img_size=84,
        )

    else:
        raise ValueError(f"Unknown task: {task_cfg.name}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Unified Residual TD3 on MuJoCo + GR00T")

    # Config file (overrides defaults, CLI overrides config)
    p.add_argument("--config", type=str, default=None,
                   help="Path to task JSON config (e.g. configs/tasks/pnp.json). "
                        "Values in JSON become defaults; explicit CLI args override them.")

    # Task selection
    p.add_argument("--task", type=str, default=None, choices=["cup", "pnp", "lift", "stack", "drawer"],
                   help="Task to train on")

    # Environment (common)
    p.add_argument("--num_envs", type=int, default=None, help="Override task default num_envs")
    p.add_argument("--max_episode_steps", type=int, default=None)
    p.add_argument("--reward_type", type=str, default=None)

    # GR00T
    p.add_argument("--groot_checkpoint", type=str, default=None)
    p.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--groot_policy_device", type=str, default=None)
    p.add_argument("--task_description", type=str, default=None)
    p.add_argument("--open_loop_horizon", type=int, default=16)

    # Residual scales
    p.add_argument("--residual_pos_scale", type=float, default=None)
    p.add_argument("--residual_rot_scale", type=float, default=None)
    p.add_argument("--residual_grip_scale", type=float, default=None)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument("--torch_compile", action="store_true", default=False)
    p.add_argument("--chunk_sync", action="store_true", default=False)

    # Algorithm
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--learning_starts", type=int, default=1_000)
    p.add_argument("--critic_warmup_steps", type=int, default=2_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--random_action_noise_scale", type=float, default=0.05)
    p.add_argument("--stddev_max", type=float, default=0.05)
    p.add_argument("--stddev_min", type=float, default=0.05)
    p.add_argument("--action_scale", type=float, default=None)
    p.add_argument("--action_l2_reg", type=float, default=None)
    p.add_argument("--actor_lr", type=float, default=None)
    p.add_argument("--critic_lr", type=float, default=None)
    p.add_argument("--num_updates_per_iteration", type=int, default=4)
    p.add_argument("--update_every_n_steps", type=int, default=1)

    # Network
    p.add_argument("--asymmetric_critic", action="store_true")
    p.add_argument("--actor_hidden_dim", type=int, default=None)
    p.add_argument("--critic_hidden_dim", type=int, default=None)

    # Offline data
    p.add_argument("--offline_data_dir", type=str, default=None)
    p.add_argument("--offline_fraction", type=float, default=None)
    p.add_argument("--offline_reward_relabel", type=str, default="none", choices=["none", "sparse", "computed"])
    p.add_argument("--reward_config", type=str, default=None,
                   help="Path to reward YAML config for computed reward relabeling (e.g. configs/reward_stack.yaml)")
    p.add_argument("--offline_pretrain_only", action="store_true")
    p.add_argument("--target_tau", type=float, default=None)
    p.add_argument("--critic_grad_clip_norm", type=float, default=None)

    # ActionScaler
    p.add_argument("--use_action_scaler", action="store_true")
    p.add_argument("--no_action_clamp", action="store_true")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--eval_interval", type=int, default=2_000)
    p.add_argument("--eval_num_episodes", type=int, default=1)
    p.add_argument("--eval_num_envs", type=int, default=None)
    p.add_argument("--debug_zero_residual", action="store_true")
    p.add_argument("--save_video", action="store_true")

    # Checkpoint & Resume
    p.add_argument("--checkpoint_interval", type=int, default=10000)
    p.add_argument("--resume_checkpoint", type=str, default=None)

    # W&B
    p.add_argument("--no_warmup", action="store_true")
    p.add_argument("--async_eval", action="store_true")
    p.add_argument("--wandb_mode", type=str, default="offline")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--no_offline_cache", action="store_true")

    # Positions (unified: single file, split into train/eval)
    p.add_argument("--positions_file", type=str, default=None,
                   help="JSON positions file. First train_positions entries for train, last eval_positions for eval.")
    p.add_argument("--train_positions", type=int, default=None,
                   help="Number of positions from positions_file to use for training envs")
    p.add_argument("--eval_positions", type=int, default=None,
                   help="Number of positions from end of positions_file to use for eval envs")

    # Task-specific (PnP) — legacy, prefer --positions_file
    p.add_argument("--cube_pos", type=float, nargs=3, default=[0.45, -0.05, 0.02])
    p.add_argument("--bowl_pos", type=float, nargs=3, default=[0.42, 0.03, 0.0])
    p.add_argument("--episode_positions_file", type=str, default=None)
    p.add_argument("--eval_positions_file", type=str, default=None)
    p.add_argument("--perturb_table", type=str, default=None)
    p.add_argument("--random_cube_range", type=str, default=None)
    p.add_argument("--random_bowl_range", type=str, default=None)
    p.add_argument("--cube_yaw", type=float, default=0.0)
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--calib_path", type=str, default=None)
    p.add_argument("--use_calibrated_wrist", action="store_true", default=False)
    p.add_argument("--success_threshold", type=float, default=None)

    # Task-specific (Cup)
    p.add_argument("--cup_positions_file", type=str, default=None)

    # First pass: check if --config was provided
    args, remaining = p.parse_known_args()

    # Load JSON config and set as defaults
    if args.config:
        import json as _json
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = Path(__file__).resolve().parents[3] / args.config
        with open(config_path) as f:
            cfg = _json.load(f)
        # Expand ~ in path fields
        for k in ("groot_checkpoint", "offline_data_dir", "scene_xml", "reward_config", "positions_file"):
            if k in cfg and cfg[k] is not None:
                cfg[k] = str(Path(cfg[k]).expanduser())
        # Set JSON values as defaults (CLI args will override)
        p.set_defaults(**{k: v for k, v in cfg.items() if v is not None})

    # Re-parse with updated defaults
    args = p.parse_args()

    # Validate task is set
    if args.task is None:
        p.error("--task is required (either via CLI or --config JSON)")
    # Validate groot_checkpoint
    if args.groot_checkpoint is None:
        p.error("--groot_checkpoint is required (either via CLI or --config JSON)")

    # Unified positions_file → episode_positions_file / eval_positions_file mapping
    if args.positions_file and not args.episode_positions_file:
        args.episode_positions_file = args.positions_file
    if args.positions_file and not args.eval_positions_file:
        args.eval_positions_file = args.positions_file

    return args


def _apply_task_defaults(args, task_cfg: TaskConfig):
    """Fill in None values from task config defaults."""
    defaults = {
        "num_envs": task_cfg.default_num_envs,
        "max_episode_steps": task_cfg.default_max_episode_steps,
        "reward_type": task_cfg.default_reward_type,
        "total_timesteps": task_cfg.default_total_timesteps,
        "buffer_size": task_cfg.default_buffer_size,
        "gamma": task_cfg.default_gamma,
        "action_scale": task_cfg.default_action_scale,
        "action_l2_reg": task_cfg.default_action_l2_reg,
        "actor_lr": task_cfg.default_actor_lr,
        "critic_lr": task_cfg.default_critic_lr,
        "actor_hidden_dim": task_cfg.default_actor_hidden_dim,
        "critic_hidden_dim": task_cfg.default_critic_hidden_dim,
        "offline_fraction": task_cfg.default_offline_fraction,
        "wandb_project": task_cfg.wandb_project,
        "task_description": task_cfg.task_description,
        "residual_pos_scale": task_cfg.residual_pos_scale,
        "residual_rot_scale": task_cfg.residual_rot_scale,
        "residual_grip_scale": task_cfg.residual_grip_scale,
        "output_dir": f"outputs/{task_cfg.name}_rl",
        "eval_num_envs": task_cfg.default_num_envs,
        "success_threshold": task_cfg.success_threshold,
    }
    for key, default_val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_val)

    # Scene XML: expand ~ and apply task default
    if not args.scene_xml and task_cfg.default_scene_xml:
        args.scene_xml = str(Path(task_cfg.default_scene_xml).expanduser())

    # If train_positions is set, override num_envs
    if getattr(args, 'train_positions', None) is not None:
        args.num_envs = args.train_positions
    # If eval_positions is set, override eval_num_envs
    if getattr(args, 'eval_positions', None) is not None:
        args.eval_num_envs = args.eval_positions


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    task_cfg = get_task_config(args.task)
    _apply_task_defaults(args, task_cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _log(f"[CONFIG] task={args.task}, reward={args.reward_type}, gamma={args.gamma}, action_scale={args.action_scale}")
    _log(f"[CONFIG] offline_fraction={args.offline_fraction}, action_l2_reg={args.action_l2_reg}")
    _log(f"[CONFIG] chunk_sync={args.chunk_sync}, use_action_scaler={args.use_action_scaler}")

    # normalize_base_action: when use_action_scaler is True, obs.base_action is normalized
    # This flag is saved in checkpoint for eval/deploy compatibility checking
    args.normalize_base_action = args.use_action_scaler
    _log(f"[CONFIG] normalize_base_action={args.normalize_base_action}")

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

    _log(f"Seed: {args.seed}, Device: {device}, Output: {outputs_dir}")

    # ── Create environment ──
    _log(f"Creating {args.task} environment...")
    mujoco_env = _create_env(args, task_cfg)
    num_envs = args.num_envs
    _log(f"Env created: {num_envs} envs")

    # ── Build ActionScaler ──
    _action_scaler = None
    if args.use_action_scaler:
        if not args.offline_data_dir:
            raise ValueError("--use_action_scaler requires --offline_data_dir")
        _log("Computing action stats for ActionScaler...")
        _action_stats = _compute_action_stats_from_offline(args.offline_data_dir)
        _action_scaler = ActionScaler(
            action_min=torch.tensor(_action_stats["min"], dtype=torch.float32),
            action_max=torch.tensor(_action_stats["max"], dtype=torch.float32),
            action_scale=args.action_scale,
            device="cpu",
            no_clamp=args.no_action_clamp,
        )
        _log(f"ActionScaler created (scale={args.action_scale}, no_clamp={args.no_action_clamp})")
    else:
        raise ValueError(
            "--use_action_scaler is required for unified training. "
            "Without it, obs.base_action normalization is inconsistent between "
            "offline (normalized) and online (raw), causing critic/actor mismatch."
        )

    # ── Wrap with residual + GR00T ──
    _log("Creating unified residual wrapper...")
    policy_device = args.groot_policy_device or args.device
    env = MuJoCoResidualWrapperUnified(
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
        chunk_sync=args.chunk_sync,
        grip_min=task_cfg.grip_min,
        grip_max=task_cfg.grip_max,
        use_gripper_latch=task_cfg.use_gripper_latch,
        camera_keys=task_cfg.camera_keys,
        action_scaler=_action_scaler,
    )
    _log("Environment ready.")

    # ── Dimensions ──
    image_keys = []
    object_state_dim = task_cfg.object_state_dim
    lowdim_keys = ["observation.state", "observation.base_action"]
    if object_state_dim > 0 and "observation.object_state" in env.observation_space.spaces:
        lowdim_keys.append("observation.object_state")

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    _log(f"State-only: lowdim={lowdim_dim}, object_state={object_state_dim}, action={action_dim}")

    # ── QAgent ──
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
        obs_shape=(3, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=[],
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=False,
    )

    # ── Replay buffers ──
    online_batch_size = int(args.batch_size * (1 - args.offline_fraction)) if args.offline_fraction > 0 else args.batch_size
    offline_batch_size = args.batch_size - online_batch_size

    online_rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=args.buffer_size, device="cpu"),
        transform=MultiStepTransform(n_steps=args.n_step, gamma=args.gamma),
        pin_memory=True,
        prefetch=4,
        batch_size=max(online_batch_size, 1),
    )

    offline_rb = None
    if args.offline_data_dir and args.offline_fraction > 0:
        offline_rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=args.buffer_size, device="cpu"),
            transform=MultiStepTransform(n_steps=args.n_step, gamma=args.gamma),
            pin_memory=True,
            prefetch=4,
            batch_size=max(offline_batch_size, 1),
        )
        _log(f"Offline buffer (dir={args.offline_data_dir}, fraction={args.offline_fraction})")

        # Buffer cache
        import hashlib, json as _json
        _cache_key = {
            "offline_data_dir": str(Path(args.offline_data_dir).resolve()),
            "n_step": args.n_step, "gamma": args.gamma,
            "buffer_size": args.buffer_size,
            "lowdim_keys": sorted(lowdim_keys),
            "reward_relabel": args.offline_reward_relabel,
            "reward_config": getattr(args, 'reward_config', None),
            "use_action_scaler": args.use_action_scaler,
            "action_scale": args.action_scale if args.use_action_scaler else None,
            "normalize_base_action": getattr(args, 'normalize_base_action', False),
        }
        _cache_hash = hashlib.sha256(_json.dumps(_cache_key, sort_keys=True).encode()).hexdigest()[:16]
        _cache_dir = Path(__file__).resolve().parents[3] / "buffer_cache" / f"offline_{args.task}_{_cache_hash}"
        _loaded = False

        if not args.no_offline_cache and _cache_dir.exists() and (_cache_dir / "cache_meta.json").exists():
            try:
                _t0 = time.time()
                offline_rb.loads(str(_cache_dir))
                _log(f"Loaded cache in {time.time()-_t0:.1f}s: {len(offline_rb)} transitions")
                _loaded = True
            except Exception as e:
                _log(f"Cache failed ({e}), repopulating...")
                import shutil
                shutil.rmtree(_cache_dir, ignore_errors=True)

        if not _loaded:
            _load_offline_data(
                offline_rb=offline_rb, data_dir=args.offline_data_dir,
                image_keys=image_keys, lowdim_keys=lowdim_keys,
                device="cpu", reward_relabel=args.offline_reward_relabel,
                action_scaler=_action_scaler,
                reward_config_path=getattr(args, 'reward_config', None),
            )
            _log(f"Offline buffer: {len(offline_rb)} transitions")
            if not args.no_offline_cache:
                try:
                    _cache_dir.mkdir(parents=True, exist_ok=True)
                    _t0 = time.time()
                    offline_rb.dumps(str(_cache_dir))
                    (_cache_dir / "cache_meta.json").write_text(
                        _json.dumps({**_cache_key, "n": len(offline_rb)}, indent=2))
                    _log(f"Saved cache in {time.time()-_t0:.1f}s")
                except Exception as e:
                    _log(f"Cache save failed: {e}")

    # ── Resume ──
    _resume_step = 0
    _resume_best = 0.0
    _resume_wandb_id = None
    if args.resume_checkpoint:
        _resume_step, _resume_best, _resume_wandb_id = _load_resume_checkpoint(
            agent, args.resume_checkpoint, device)

    # ── W&B ──
    _wb = wandb
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.task}_td3_seed{args.seed}"
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

    # ── Async eval ──
    _async_eval = None
    if args.async_eval:
        _async_eval_dir = outputs_dir / "async_eval_results"
        _async_eval = AsyncEvaluator(args, checkpoint_dir, _async_eval_dir, action_scaler=_action_scaler)
        _wandb_run_id = _wb.run.id if (_wb is not None and _wb.run is not None) else None
        _async_eval.start(wandb_run_id=_wandb_run_id)
        _log("Async eval started")

    # ══════════════════════════════════════════════════════════════
    # Warm-up
    # ══════════════════════════════════════════════════════════════
    if args.no_warmup:
        args.learning_starts = 0
        args.critic_warmup_steps = 0
    _log(f"Warm-up: {args.learning_starts} transitions...")
    obs, _ = env.reset()
    warmup_transitions = 0

    while warmup_transitions < args.learning_starts:
        noise = (torch.rand((num_envs, action_dim), device=device) * 2 - 1) * args.random_action_noise_scale
        next_obs, reward, terminated, truncated, info = env.step(noise)
        done = terminated | truncated

        _replay_action = _to_replay_action_7d(obs["observation.base_action"], noise)
        reward_clamped = reward.clamp(0.0, 1.0)

        # Skip stale chunk_sync transitions; use terminated for Q-bootstrap
        if args.chunk_sync and done.any():
            store_mask = ~done.bool()
        else:
            store_mask = None

        _add_transitions(
            obs=obs, next_obs=next_obs, actions=_replay_action,
            reward=reward_clamped, done=terminated, device=device,
            image_keys=image_keys, lowdim_keys=lowdim_keys,
            num_envs=num_envs, online_rb=online_rb,
            store_mask=store_mask,
        )
        warmup_transitions += num_envs
        obs = next_obs

    _log(f"Warm-up done: {warmup_transitions} transitions")

    # ── Critic warmup ──
    if args.critic_warmup_steps > 0:
        _log(f"Critic warmup: {args.critic_warmup_steps} updates...")
        for i in range(args.critic_warmup_steps):
            batch = online_rb.sample(args.batch_size).to(device, non_blocking=True)
            if batch.ndim > 1:
                batch = batch.squeeze(1)
            metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)
            if i % 200 == 0:
                _log(f"  critic warmup: {i}/{args.critic_warmup_steps} loss={metrics.get('train/critic_loss', 0):.4f}")
        _log("Critic warmup done.")

    # ══════════════════════════════════════════════════════════════
    # Main training loop
    # ══════════════════════════════════════════════════════════════
    _log(f"Training {args.total_timesteps} steps...")
    obs, _ = env.reset()
    global_step = _resume_step
    episode_count = 0
    best_success = _resume_best
    train_start = time.time()
    _timers = {"env_step": [], "agent_act": [], "grad_update": [], "buffer_store": []}
    ep_cum_reward = torch.zeros(num_envs, device=device)
    ep_step_counter = torch.zeros(num_envs, device=device, dtype=torch.long)

    while global_step <= args.total_timesteps:
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

        ep_cum_reward += reward
        ep_step_counter += 1
        if done.any():
            done_mask = done.bool()
            episode_count += done_mask.sum().item()
            if _wb is not None and _wb.run is not None:
                _wb.log({
                    "training/episode_return": float(ep_cum_reward[done_mask].mean().item()),
                    "training/episode_steps": float(ep_step_counter[done_mask].float().mean().item()),
                    "training/episode_count": episode_count,
                }, step=global_step)
            ep_cum_reward[done_mask] = 0.0
            ep_step_counter[done_mask] = 0

        # Store transition
        # Use terminated (not done) for Q-bootstrap: truncated episodes should still bootstrap
        _replay_action = _to_replay_action_7d(obs["observation.base_action"], residual_action)
        # Clamp online reward to [0,1] to match offline (prevents Q-value instability)
        reward_clamped = reward.clamp(0.0, 1.0)

        # Skip transitions where chunk_sync has stale base actions (env just reset)
        if args.chunk_sync and done.any():
            store_mask = ~done.bool()
        else:
            store_mask = torch.ones(num_envs, dtype=torch.bool, device=device)

        _t0 = time.perf_counter()
        if store_mask.any():
            _add_transitions(
                obs=obs, next_obs=next_obs, actions=_replay_action,
                reward=reward_clamped, done=terminated, device=device,
                image_keys=image_keys, lowdim_keys=lowdim_keys,
                num_envs=num_envs, online_rb=online_rb,
                store_mask=store_mask,
            )
        _timers["buffer_store"].append(time.perf_counter() - _t0)
        obs = next_obs

        # ── Update agent ──
        if len(online_rb) >= max(online_batch_size, 1) and global_step % args.update_every_n_steps == 0:
            _t0 = time.perf_counter()
            for _ in range(args.num_updates_per_iteration):
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
                metrics = agent.update(batch, stddev=stddev, update_actor=update_actor, bc_batch=None, ref_agent=agent)
            _timers["grad_update"].append(time.perf_counter() - _t0)

            if global_step % 100 == 0 and _wb is not None and _wb.run is not None:
                log_dict = {f"train/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                log_dict["training/global_step"] = global_step
                _wb.log(log_dict, step=global_step)

        # ── Periodic logging ──
        if global_step % 500 == 0:
            ra = residual_action[0].detach().cpu().numpy() if residual_action is not None else np.zeros(7)
            elapsed = time.time() - train_start
            sps = max(1, global_step) / max(1, elapsed)
            eta_h = (args.total_timesteps - global_step) / max(0.01, sps) / 3600

            _timing = " ".join(f"{k}={sum(v[-500:])/len(v[-500:])*1000:.1f}ms" for k, v in _timers.items() if v)
            _log(f"step={global_step}/{args.total_timesteps} rw={reward[0].item():.4f} eps={episode_count} "
                 f"speed={sps:.1f}sps ETA={eta_h:.1f}h res={ra[:6].mean():.5f}")
            if _timing:
                _log(f"  {_timing}")

        # ── Eval ──
        if global_step % args.eval_interval == 0 and _async_eval is not None:
            _async_eval.trigger_eval(agent, global_step)

        if _async_eval is not None:
            result = _async_eval.collect_results()
            if result is not None:
                sr = result.get("eval/success_rate", 0.0)
                if _wb is not None and _wb.run is not None:
                    _wb.log(result)
                eval_step = result.get("eval/step", 0)
                eval_ckpt = checkpoint_dir / f"eval_request_step{eval_step}.pt"
                if sr >= best_success:
                    best_success = sr
                    # Load weights from the eval request checkpoint (not current agent)
                    if eval_ckpt.exists():
                        _best_data = torch.load(eval_ckpt, map_location="cpu", weights_only=False)
                        torch.save({"model": _best_data["model"], "args": vars(args),
                                    "eval_step": eval_step, "success_rate": sr},
                                   checkpoint_dir / "best.pt")
                    else:
                        # Fallback: use current weights (shouldn't happen normally)
                        torch.save({"model": agent.state_dict(), "args": vars(args)},
                                   checkpoint_dir / "best.pt")
                # Cleanup: delete eval request files older than this one
                for old_req in checkpoint_dir.glob("eval_request_step*.pt"):
                    old_step = int(old_req.stem.split("step")[1])
                    if old_step < eval_step:
                        try:
                            old_req.unlink()
                        except Exception:
                            pass

        # ── Checkpoint ──
        if args.checkpoint_interval > 0 and global_step > 0 and global_step % args.checkpoint_interval == 0:
            _wandb_id = _wb.run.id if (_wb is not None and _wb.run is not None) else None
            _save_resume_checkpoint(agent, checkpoint_dir, global_step, best_success, args, _wandb_id)

        global_step += 1

    # ── Final ──
    if _async_eval is not None:
        _async_eval.trigger_eval(agent, global_step)
        _log("Final eval triggered, waiting...")
        for _ in range(60):
            time.sleep(5)
            result = _async_eval.collect_results()
            if result:
                _log(f"Final SR={result.get('eval/success_rate', 0.0):.2%}")
                if _wb is not None and _wb.run is not None:
                    _wb.log(result, step=global_step)
                break
        _async_eval.shutdown()

    torch.save({"model": agent.state_dict(), "args": vars(args)},
               checkpoint_dir / f"final_step{global_step}.pt")
    _log(f"Final checkpoint saved.")

    if _wb is not None and _wb.run is not None:
        _wb.finish()
    env.close()
    _log("Training complete.")


if __name__ == "__main__":
    main()

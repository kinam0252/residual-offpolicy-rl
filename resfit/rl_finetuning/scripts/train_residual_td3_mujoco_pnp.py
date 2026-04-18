"""
Residual TD3 training on MuJoCo + GR00T.

Standalone script (no Isaac Sim dependency) that uses ``MuJoCoVecEnv`` +
``MuJoCoResidualWrapper`` for environment interaction, and reuses the
same QAgent/TD3, replay buffers, and critic warmup logic from the
IsaacLab version.

Launch:
    source ~/.venvs/groot/bin/activate
    MUJOCO_GL=egl LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-} \\
    python resfit/rl_finetuning/scripts/train_residual_td3_mujoco.py \\
        --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2/checkpoint-30000 \\
        --num_envs 1 --total_timesteps 50000
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
# transformers.modeling_utils does `import deepspeed` which fails without CUDA toolkit.
# We don't use deepspeed for inference, so insert a mock module.
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
# Newer huggingface_hub rejects absolute paths in validate_repo_id() even for local use.
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig_validate = _hf_val.validate_repo_id
    def _patched_validate_repo_id(repo_id: str) -> None:
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")):
            return  # skip validation for local paths
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
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP as MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP as MuJoCoVecEnv

try:
    import wandb
except ImportError:
    wandb = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def _log(msg: str) -> None:
    print(f"[mujoco-pnp-td3] {msg}", flush=True)



def _load_offline_data(
    offline_rb,
    data_dir: str,
    image_keys: list,
    lowdim_keys: list,
    device: str = "cpu",
) -> None:
    """Load offline .npz files into the replay buffer (vectorised)."""
    from pathlib import Path as _P
    npz_files = sorted(_P(data_dir).rglob("*.npz"))
    if not npz_files:
        _log(f"WARNING: No .npz files found in {data_dir}")
        return

    total_loaded = 0
    for npz_path in npz_files:
        _log(f"  Loading {npz_path.name}...")
        data = np.load(str(npz_path), allow_pickle=True)
        n = len(data["obs_state"])
        has_depth = "obs_depth_front" in data

        # Build obs tensors (all at once)
        obs_td = {
            "observation.state": torch.from_numpy(data["obs_state"].astype(np.float32)),
            "observation.base_action": torch.from_numpy(data["obs_base_action"].astype(np.float32)),
        }
        if "observation.object_state" in lowdim_keys:
            obs_td["observation.object_state"] = torch.from_numpy(
                data["obs_object_state"].astype(np.float32))
        if has_depth:
            for cam_key, npz_key in [
                ("observation.depth.front", "obs_depth_front"),
                ("observation.depth.wrist", "obs_depth_wrist"),
            ]:
                if cam_key in image_keys:
                    obs_td[cam_key] = torch.from_numpy(data[npz_key].astype(np.float32))
        else:
            for cam_key in image_keys:
                if "depth" in cam_key:
                    obs_td[cam_key] = torch.zeros(n, 1, 84, 84, dtype=torch.float32)

        # Fill missing RGB image keys with zeros (offline data may only have depth)
        for cam_key in image_keys:
            if cam_key not in obs_td:
                if "images" in cam_key:
                    obs_td[cam_key] = torch.zeros(n, 3, 84, 84, dtype=torch.float32)
                elif "depth" in cam_key:
                    obs_td[cam_key] = torch.zeros(n, 1, 84, 84, dtype=torch.float32)

        # Build next_obs: shift by 1, copy current obs at episode boundaries
        done_t = torch.from_numpy(data["done"].astype(bool))
        next_obs_td = {}
        for k, v in obs_td.items():
            shifted = torch.roll(v, -1, dims=0)
            shifted[-1] = v[-1]
            boundary = done_t.view(-1, *([1] * (v.ndim - 1))).expand_as(v)
            next_obs_td[k] = torch.where(boundary, v, shifted)

        reward_t = torch.from_numpy(data["reward"].astype(np.float32))
        action_t = torch.from_numpy(data["action"].astype(np.float32))
        # Add one-by-one but with pre-converted tensors (much faster than np indexing)
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
        _log(f"    {npz_path.name}: {n} transitions loaded")

    _log(f"  Loaded {total_loaded} transitions from {len(npz_files)} files")


def _add_transitions(
    obs, next_obs, actions, reward, done,
    device, image_keys, lowdim_keys, num_envs, online_rb,
):
    obs_keys = set(image_keys) | set(lowdim_keys)
    # Only convert RGB images to uint8, not depth (float32)
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
    """Manages a background eval subprocess with its own MuJoCo env + GR00T.

    Lifecycle:
      1. __init__: stores config
      2. start(): launches subprocess
      3. trigger_eval(agent, step): saves weights, eval picks them up
      4. collect_results(): non-blocking check for JSON results
      5. shutdown(): kills subprocess
    """

    def __init__(self, args, checkpoint_dir: Path, results_dir: Path):
        self._args = args
        self._checkpoint_dir = checkpoint_dir
        self._results_dir = results_dir
        self._proc = None
        self._best_success = 0.0
        results_dir.mkdir(parents=True, exist_ok=True)

    def start(self, wandb_run_id: str | None = None):
        """Launch the async eval subprocess."""
        eval_script = str(Path(__file__).parent / "eval_async_mujoco_pnp.py")
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
            "--success_threshold", str(a.success_threshold),
            "--reward_type", a.reward_type,
            "--device", a.device,
            "--eval_num_episodes", str(a.eval_num_episodes),
            "--poll_interval_sec", "10",
        ]
        if a.groot_policy_device:
            cmd += ["--groot_policy_device", a.groot_policy_device]
        if a.save_video:
            cmd += ["--save_video"]
        if a.asymmetric_critic:
            cmd += ["--asymmetric_critic"]
        # Use eval_perturb_table for fixed eval positions if set
        if a.eval_perturb_table:
            cmd += ["--perturb_table", a.eval_perturb_table]
        elif a.perturb_table:
            cmd += ["--perturb_table", a.perturb_table]
        else:
            cmd += ["--cube_pos"] + [str(x) for x in a.cube_pos]
            cmd += ["--num_envs", str(a.num_envs)]
        if a.scene_xml:
            cmd += ["--scene_xml", a.scene_xml]
        if a.calib_path:
            cmd += ["--calib_path", a.calib_path]
        if a.use_calibrated_wrist:
            cmd += ["--use_calibrated_wrist"]
        if not a.eval_perturb_table and a.random_cube_range:
            cmd += ["--random_cube_range", a.random_cube_range]
        if hasattr(a, "episode_positions_file") and a.episode_positions_file:
            cmd += ["--episode_positions_file", a.episode_positions_file]
        if hasattr(a, "bowl_pos"):
            cmd += ["--bowl_pos"] + [str(x) for x in a.bowl_pos]
        # W&B: log to same run as training
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
        """Save agent weights for the eval process to pick up. Non-blocking."""
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
            # Rename to .done
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


# ── CLI ──



def _save_resume_checkpoint(agent, checkpoint_dir, global_step, best_success, args, wandb_run_id=None):
    """Save a full resume checkpoint (atomic write)."""
    ckpt_data = {
        # Agent network weights
        "encoders": agent.encoders.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        # Optimizer states
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "encoder_opt": agent.encoder_opt.state_dict(),
        # Training state
        "global_step": global_step,
        "best_success": best_success,
        "wandb_run_id": wandb_run_id,
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }
    # Scheduler states (if they exist)
    for sched_name in ("actor_scheduler", "critic_scheduler", "encoder_scheduler"):
        sched = getattr(agent, sched_name, None)
        if sched is not None:
            ckpt_data[sched_name] = sched.state_dict()
    # Atomic write
    path = Path(checkpoint_dir) / "resume_latest.pt"
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(ckpt_data, str(tmp_path))
    tmp_path.rename(path)
    _log(f"Resume checkpoint saved: {path} (step={global_step})")


def _load_resume_checkpoint(agent, ckpt_path, device):
    """Load a resume checkpoint and restore full training state.
    
    Returns:
        (global_step, best_success, wandb_run_id)
    """
    _log(f"Loading resume checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Restore network weights
    agent.encoders.load_state_dict(ckpt["encoders"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.critic_target.load_state_dict(ckpt["critic_target"])
    agent.actor_target.load_state_dict(ckpt["actor_target"])
    # Restore optimizer states
    agent.actor_opt.load_state_dict(ckpt["actor_opt"])
    agent.critic_opt.load_state_dict(ckpt["critic_opt"])
    agent.encoder_opt.load_state_dict(ckpt["encoder_opt"])
    # Restore scheduler states
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
    p = argparse.ArgumentParser(description="Residual TD3 on MuJoCo + GR00T")
    # Environment
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--cube_pos", type=float, nargs=3, default=[0.45, -0.05, 0.02])
    p.add_argument("--bowl_pos", type=float, nargs=3, default=[0.42, 0.03, 0.0], help="Bowl xyz position")
    p.add_argument("--episode_positions_file", type=str, default=None, help="JSON with per-episode cube/bowl positions")
    p.add_argument("--cube_yaw", type=float, default=0.0)
    p.add_argument("--perturb_table", type=str, default=None,
                   help="Path to JSON cube perturbation table (overrides --cube_pos and --num_envs)")
    p.add_argument("--random_cube_range", type=str, default=None,
                   help="JSON string: {dx: [lo,hi], dy: [lo,hi], yaw: [lo,hi]} in cm/degrees")
    p.add_argument("--eval_perturb_table", type=str, default=None,
                   help="Fixed perturb table for eval (overrides random_cube_range in eval only)")
    p.add_argument("--curriculum_stages", type=str, default=None,
                   help="JSON list of curriculum stages: [{step:N, range:{dx,dy,yaw}}, ...]")

    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--calib_path", type=str, default=None)
    p.add_argument("--use_calibrated_wrist", action="store_true", default=False,
                   help="Use calibrated wrist cam from yaml (66ep/100ep). Default: hardcoded 15deg tilt (32ep).")
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--success_threshold", type=float, default=0.03)
    p.add_argument("--reward_type", type=str, default="dense_clipped", choices=["sparse", "dense", "dense_clipped", "dense_v2"])
    # GR00T
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--groot_policy_device", type=str, default=None)
    p.add_argument("--task_description", type=str, default="Pick up the red cube and place it onto the plate.")
    p.add_argument("--open_loop_horizon", type=int, default=16)
    # Residual scales
    p.add_argument("--residual_pos_scale", type=float, default=0.02)
    p.add_argument("--residual_rot_scale", type=float, default=0.05)
    p.add_argument("--residual_grip_scale", type=float, default=0.1)
    p.add_argument("--ema_alpha", type=float, default=0.0, help="EMA smoothing for GR00T base action (0=off, 0.9=heavy)")
    # Algorithm
    p.add_argument("--total_timesteps", type=int, default=50_000)
    p.add_argument("--learning_starts", type=int, default=1_000)
    p.add_argument("--critic_warmup_steps", type=int, default=1_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=200_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--random_action_noise_scale", type=float, default=0.0)
    p.add_argument("--stddev_max", type=float, default=0.05)
    p.add_argument("--stddev_min", type=float, default=0.05)
    p.add_argument("--action_scale", type=float, default=0.1)
    p.add_argument("--action_l2_reg", type=float, default=10.0)
    p.add_argument("--actor_lr", type=float, default=1e-5)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--num_updates_per_iteration", type=int, default=4)
    p.add_argument("--update_every_n_steps", type=int, default=1)
    # Asymmetric critic
    p.add_argument("--asymmetric_critic", action="store_true")
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--critic_hidden_dim", type=int, default=1024)
    # Offline data
    p.add_argument("--offline_data_dir", type=str, default=None)
    p.add_argument("--offline_fraction", type=float, default=0.5)
    p.add_argument("--offline_pretrain_only", action="store_true",
                   help="Use offline data only for critic warmup, then switch to pure online")
    p.add_argument("--target_tau", type=float, default=None,
                   help="Soft target update rate (overrides default 0.01)")
    p.add_argument("--critic_grad_clip_norm", type=float, default=None,
                   help="Critic gradient clip norm (overrides default 1.0)")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default="outputs/mujoco_td3")
    p.add_argument("--eval_interval", type=int, default=1_000)
    p.add_argument("--eval_num_episodes", type=int, default=2)
    p.add_argument("--debug_zero_residual", action="store_true")
    p.add_argument("--save_video", action="store_true")
    # Checkpoint & Resume
    p.add_argument("--checkpoint_interval", type=int, default=5000, help="Save resume checkpoint every N steps (0=disabled)")
    p.add_argument("--resume_checkpoint", type=str, default=None, help="Path to resume_latest.pt to continue training")
    # W&B
    p.add_argument("--no_warmup", action="store_true", help="Skip warmup and critic warmup (debug)")
    p.add_argument("--async_eval", action="store_true", help="Use async eval subprocess")
    p.add_argument("--wandb_mode", type=str, default="offline")
    p.add_argument("--wandb_project", type=str, default="mujoco-franka-residual-td3")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    # ── Create MuJoCo environment ──
    _log("Creating MuJoCo environment...")

    # Load perturbation table if provided
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
        _log(f"Loaded perturbation table: {args.perturb_table} ({len(cube_positions)} positions)")
        for i, (cp, ec) in enumerate(zip(cube_positions, envs_cfg)):
            _log(f"  env {i}: {ec['name']} cube=({cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f})")
    else:
        cube_positions = [args.cube_pos] * args.num_envs

    # Parse random cube range if provided
    random_cube_range = None
    if args.random_cube_range:
        import json as _json2
        _parsed_rcr = _json2.loads(args.random_cube_range)
        def _cvt_one(r):
            if r is None: return None
            return {"dx": (r["dx"][0]/100.0, r["dx"][1]/100.0), "dy": (r["dy"][0]/100.0, r["dy"][1]/100.0), "yaw": tuple(r.get("yaw", [0,0]))}
        if isinstance(_parsed_rcr, list):
            random_cube_range = [_cvt_one(item) for item in _parsed_rcr]
            _log(f"Random cube range (list, {len(random_cube_range)} levels): {args.random_cube_range}")
        else:
            random_cube_range = _cvt_one(_parsed_rcr)
            _log(f"Random cube range: dx={_parsed_rcr['dx']}cm dy={_parsed_rcr['dy']}cm yaw={_parsed_rcr.get('yaw', [0,0])}°")


    # Parse curriculum stages if provided
    _curriculum_stages = None
    if args.curriculum_stages:
        import json as _json3
        _curriculum_stages = _json3.loads(args.curriculum_stages)
        # Convert cm to m for each stage
        for _cs in _curriculum_stages:
            _r = _cs["range"]
            _cs["_parsed"] = {
                "dx": (_r["dx"][0]/100.0, _r["dx"][1]/100.0),
                "dy": (_r["dy"][0]/100.0, _r["dy"][1]/100.0),
                "yaw": tuple(_r.get("yaw", [0, 0])),
            }
        _curriculum_stages.sort(key=lambda x: x["step"])
        _log(f"Curriculum: {len(_curriculum_stages)} stages")
        for _cs in _curriculum_stages:
            _log(f"  step>={_cs['step']}: {_cs['range']}")

    # ── Auto-detect camera calibration from GR00T checkpoint ──
    _calib_path = args.calib_path
    _use_calibrated_wrist = args.use_calibrated_wrist
    if _calib_path is None:
        ckpt_lower = args.groot_checkpoint.lower()
        if "66ep" in ckpt_lower or "100ep" in ckpt_lower:
            _calib_66ep = str(Path(__file__).resolve().parents[4] / "Mujoco_Franka" / "config" / "camera_info_66ep.yaml")
            if Path(_calib_66ep).exists():
                _calib_path = _calib_66ep
                _use_calibrated_wrist = True
                _log(f"Auto-detected 66ep/100ep checkpoint → using {_calib_66ep} + calibrated wrist")
            else:
                _log(f"WARNING: 66ep/100ep checkpoint detected but {_calib_66ep} not found, using default calib")

    mujoco_env = MuJoCoVecEnv(
        num_envs=args.num_envs,
        cube_positions=cube_positions,
        cube_yaw_deg=args.cube_yaw,
        random_cube_range=random_cube_range,
        scene_xml=args.scene_xml,
        calib_path=_calib_path,
        use_calibrated_wrist=_use_calibrated_wrist,
        max_episode_steps=args.max_episode_steps,
        success_threshold=args.success_threshold,
        reward_type=args.reward_type,
        device=args.device,
        rl_img_size=84,
        bowl_positions=[args.bowl_pos],
        episode_positions_file=args.episode_positions_file,
    )
    _log(f"MuJoCo PnP env: {args.num_envs} envs, cube_pos={args.cube_pos}, bowl_pos={args.bowl_pos}")

    # ── Wrap with residual + GR00T ──
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
    )
    _log("Environment ready.")

    # ── Dimensions ──
    num_envs = args.num_envs
    _asymmetric = args.asymmetric_critic
    object_state_dim = 10 if _asymmetric else 0

    if _asymmetric:
        # Asymmetric: actor=state-only, critic=depth
        image_keys = ["observation.depth.front", "observation.depth.wrist"]
        img_c, img_h, img_w = 1, 84, 84  # depth is 1-channel
        _log(f"ASYMMETRIC mode: critic uses depth {image_keys}, actor is state-only")
    else:
        image_keys = [
            "observation.images.front",
            "observation.images.back",
            "observation.images.wrist",
        ]
        img_c, img_h, img_w = 3, 84, 84

    lowdim_keys = ["observation.state", "observation.base_action"]
    if object_state_dim > 0:
        lowdim_keys.append("observation.object_state")

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim  # 7 (residual)

    _log(f"lowdim_dim={lowdim_dim}, img=({img_c},{img_h},{img_w}), action_dim={action_dim}, object_state={object_state_dim}")

    # ── QAgent ──
    cfg = ResidualTD3MuJoCoConfig()
    # Override from CLI
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
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=_asymmetric,
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
            f"--offline_fraction={args.offline_fraction} > 0 but --offline_data_dir is not set. "
            "Either provide --offline_data_dir or set --offline_fraction 0."
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
        # ── Buffer cache: hash config to skip slow npz loading ──
        import hashlib, json as _json
        # Only hash keys that change the offline buffer CONTENT
        # (data source + transform config, NOT training hyperparams)
        _cache_key_dict = {
            "offline_data_dir": str(Path(args.offline_data_dir).resolve()),
            "n_step": args.n_step,
            "gamma": args.gamma,
            "buffer_size": args.buffer_size,
            "image_keys": sorted(image_keys),
            "lowdim_keys": sorted(lowdim_keys),
        }
        _cache_hash = hashlib.sha256(_json.dumps(_cache_key_dict, sort_keys=True).encode()).hexdigest()[:16]
        _cache_root = Path(__file__).resolve().parents[3] / "buffer_cache"
        _cache_dir = _cache_root / f"offline_{_cache_hash}"
        _loaded_from_cache = False

        if _cache_dir.exists() and (_cache_dir / "cache_meta.json").exists():
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
            # ── Populate offline buffer from .npz files ──
            _load_offline_data(
                offline_rb=offline_rb,
                data_dir=args.offline_data_dir,
                image_keys=image_keys,
                lowdim_keys=lowdim_keys,
                device="cpu",
            )
            _log(f"Offline buffer populated: {len(offline_rb)} transitions")
            # Save cache for next run
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
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mujoco_td3_seed{args.seed}"
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
            # Eval metrics use their own step axis (avoids step conflict with training)
            _wb.define_metric("eval/*", step_metric="eval/step")
        except Exception as e:
            _log(f"[WARN] wandb.init() failed: {e}")
            _wb = None

    # ══════════════════════════════════════════════════════════════════
    # Async eval setup
    # ══════════════════════════════════════════════════════════════════
    _async_eval = None
    if args.async_eval:
        _async_eval_dir = outputs_dir / "async_eval_results"
        _async_eval = AsyncEvaluator(args, checkpoint_dir, _async_eval_dir)
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

        # Store transitions — use noise (7D residual), not combined (8D absolute)
        _add_transitions(
            obs=obs, next_obs=next_obs, actions=noise,
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

        # Curriculum: update random_cube_range based on global_step
        if _curriculum_stages is not None:
            _new_range = None
            for _cs in _curriculum_stages:
                if global_step >= _cs["step"]:
                    _new_range = _cs["_parsed"]
            if _new_range is not None and _new_range != mujoco_env._random_cube_range:
                mujoco_env._random_cube_range = _new_range
                _log(f"[Curriculum] step={global_step}: range updated to {_new_range}")

        # Linear noise schedule: stddev_max -> stddev_min over total_timesteps
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

        # Store transition — use residual_action (7D), not combined (8D absolute)
        _t0 = time.perf_counter()
        _add_transitions(
            obs=obs, next_obs=next_obs, actions=residual_action,
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
                _use_offline = (offline_rb is not None and len(offline_rb) > 0
                                and offline_batch_size > 0
                                and not (args.offline_pretrain_only and global_step >= args.critic_warmup_steps))
                if _use_offline:
                    obatch = offline_rb.sample(offline_batch_size).to(device, non_blocking=True)
                    # Drop _priority key to avoid mismatch during concat
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

            # Timing stats (avg ms over last window)
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

            # Log timing to wandb
            if _wb is not None and _wb.run is not None and _timing_wb:
                _wb.log(_timing_wb, step=global_step)

        # ── (4) Evaluation ──
        if global_step % args.eval_interval == 0:
            if _async_eval is not None:
                _async_eval.trigger_eval(agent, global_step)
            else:
                # Fallback: sync eval (only if async not enabled)
                from eval_async_mujoco import evaluate as _sync_evaluate
                _log(f"Evaluating ({args.eval_num_episodes} episodes)...")
                eval_metrics = _sync_evaluate(
                    env=env, agent=agent,
                    num_episodes=args.eval_num_episodes,
                    device=device, image_keys=image_keys,
                )
                sr = eval_metrics["eval/success_rate"]
                _log(f"  success_rate={sr:.2%}, mean_return={eval_metrics['eval/mean_return']:.2f}")
                if _wb is not None and _wb.run is not None:
                    _wb.log(eval_metrics, step=global_step)
                if sr >= best_success:
                    best_success = sr
                    ckpt_path = checkpoint_dir / "best.pt"
                    torch.save({"model": agent.state_dict(), "args": vars(args)}, ckpt_path)
                    _log(f"  Saved best checkpoint: {ckpt_path}")

        # ── Collect async eval results (non-blocking) ──
        if _async_eval is not None:
            async_result = _async_eval.collect_results()
            if async_result is not None:
                sr = async_result.get("eval/success_rate", 0.0)
                step_eval = async_result.get("eval/step", global_step)
                if _wb is not None and _wb.run is not None:
                    _wb.log(async_result)  # eval/* uses eval/step axis via define_metric
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

    # ── Final eval ──
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
    else:
        from eval_async_mujoco import evaluate as _sync_evaluate
        _log("Final evaluation...")
        eval_metrics = _sync_evaluate(
            env=env, agent=agent,
            num_episodes=args.eval_num_episodes,
            device=device, image_keys=image_keys,
        )
        _log(f"Final: success_rate={eval_metrics['eval/success_rate']:.2%}")
        if _wb is not None and _wb.run is not None:
            _wb.log(eval_metrics, step=global_step)

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

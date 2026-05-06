#!/usr/bin/env python3
"""
Rollout video collector for base vs residual policy comparison.

Renders specific positions and saves per-episode mp4 videos + trajectory data.
Uses the same env/agent setup as rollout_trajectory_analysis.py.

Usage:
  python scripts/rollout_video.py \
      --task stack --mode residual --positions 0,5,6,8,14,15 \
      --output_dir outputs/trajectory_videos --num_rounds 3
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types as _types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# -- Deepspeed mock --
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

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import cv2
import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

# Force CUDA initialization early (prevents FlashAttention2 CPU fallback)
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


# ── Task-specific configurations (same as rollout_trajectory_analysis.py) ──

TASK_CONFIGS = {
    "lift": dict(
        groot_checkpoint=os.path.expanduser(
            "~/DATA/INTERN/training/gr00t_sim_100ep/checkpoint-300000"),
        residual_checkpoint=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/outputs/mujoco_td3/"
            "gr00t_sim_100ep/easy_s1L2_s2/checkpoints/best.pt"),
        task_description="lift the cube",
        max_episode_steps=300,
        action_scale=0.2,
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        asymmetric_critic=True,
        object_state_dim=7,
        residual_pos_scale=0.02,
        residual_rot_scale=0.05,
        residual_grip_scale=0.1,
        ema_alpha=0.0,
        open_loop_horizon=16,
        embodiment_tag="NEW_EMBODIMENT",
        eval_positions_file=None,
    ),
    "pnp": dict(
        groot_checkpoint=os.path.expanduser(
            "~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"),
        residual_checkpoint=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/outputs/pnp_train_v3/"
            "v3_normal_as0.1_l21.0_g0.95_s42/checkpoints/best.pt"),
        task_description="Pick up the red cube and place it onto the plate.",
        max_episode_steps=500,
        action_scale=0.1,
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        asymmetric_critic=True,
        object_state_dim=10,
        residual_pos_scale=0.02,
        residual_rot_scale=0.05,
        residual_grip_scale=0.1,
        ema_alpha=0.0,
        open_loop_horizon=16,
        embodiment_tag="NEW_EMBODIMENT",
        eval_positions_file=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/configs/pnp_eval_normal.json"),
    ),
    "stack": dict(
        groot_checkpoint=os.path.expanduser(
            "~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-75000"),
        residual_checkpoint=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/outputs/stack_td3_75k/"
            "hard_l25.0_g095/checkpoints/best.pt"),
        task_description="Pick up the white cube and stack it on the green cube.",
        max_episode_steps=1000,
        action_scale=0.1,
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        asymmetric_critic=True,
        object_state_dim=10,
        residual_pos_scale=0.02,
        residual_rot_scale=0.05,
        residual_grip_scale=0.004,
        ema_alpha=0.0,
        open_loop_horizon=16,
        embodiment_tag="NEW_EMBODIMENT",
        eval_positions_file=os.path.expanduser(
            "~/Repos/Intern/residual-offpolicy-rl/configs/stack_cube_positions.json"),
    ),
}


def _log(msg: str):
    print(f"[video] {msg}", flush=True)


def _load_positions(cfg: dict, position_indices: list[int]):
    """Load specific positions from eval positions file."""
    pos_file = cfg.get("eval_positions_file")
    if not pos_file or not os.path.exists(pos_file):
        raise ValueError(f"No eval positions file: {pos_file}")

    with open(pos_file) as f:
        pos_list = json.load(f)

    # Deduplicate by episode id
    seen = set()
    unique_pos = []
    for pp in pos_list:
        eid = pp.get("episode", len(unique_pos))
        if eid not in seen:
            seen.add(eid)
            unique_pos.append(pp)

    selected = [unique_pos[i] for i in position_indices if i < len(unique_pos)]
    _log(f"Selected {len(selected)} positions: indices {position_indices}")
    return selected


def _create_env(task: str, cfg: dict, positions: list[dict], device: str):
    """Create env with specific fixed positions."""
    num_envs = len(positions)

    if task == "lift":
        from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
        from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
        mujoco_env = MuJoCoVecEnv(
            num_envs=num_envs,
            max_episode_steps=cfg["max_episode_steps"],
            device=device,
        )
    elif task == "pnp":
        from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP as MuJoCoVecEnv
        from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP as MuJoCoResidualWrapper
        mujoco_env = MuJoCoVecEnv(
            num_envs=num_envs,
            max_episode_steps=cfg["max_episode_steps"],
            reward_type="dense",
            device=device,
            cube_positions=[p["cube_pos"] for p in positions],
            bowl_positions=[p["bowl_pos"] for p in positions],
            cube_yaw_degs=[p.get("cube_yaw_deg", 0.0) for p in positions],
        )
    elif task == "stack":
        from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import MuJoCoVecEnvStack as MuJoCoVecEnv
        from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack as MuJoCoResidualWrapper
        mujoco_env = MuJoCoVecEnv(
            num_envs=num_envs,
            max_episode_steps=cfg["max_episode_steps"],
            reward_type="dense",
            device=device,
            white_cube_positions=[p["white_cube_pos"] for p in positions],
            green_cube_positions=[p["green_cube_pos"] for p in positions],
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=cfg["groot_checkpoint"],
        embodiment_tag=cfg["embodiment_tag"],
        policy_device=device,
        task_description=cfg["task_description"],
        open_loop_horizon=cfg["open_loop_horizon"],
        residual_pos_scale=cfg["residual_pos_scale"],
        residual_rot_scale=cfg["residual_rot_scale"],
        residual_grip_scale=cfg["residual_grip_scale"],
        ema_alpha=cfg["ema_alpha"],
    )
    return env, mujoco_env


def _create_agent(env, cfg: dict, device: torch.device):
    """Create QAgent matching the task config."""
    image_keys = ["observation.depth.front", "observation.depth.wrist"] \
        if cfg["asymmetric_critic"] else \
        ["observation.images.front", "observation.images.back", "observation.images.wrist"]
    img_c = 1 if cfg["asymmetric_critic"] else 3

    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim

    agent_cfg = ResidualTD3MuJoCoConfig()
    agent_cfg.agent.actor.action_scale = cfg["action_scale"]
    agent_cfg.agent.actor.hidden_dim = cfg["actor_hidden_dim"]
    agent_cfg.agent.critic.hidden_dim = cfg["critic_hidden_dim"]

    agent = QAgent(
        obs_shape=(img_c, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=agent_cfg.agent,
        residual_actor=True,
        object_state_dim=cfg.get("object_state_dim", 0),
        asymmetric_critic=cfg["asymmetric_critic"],
    )
    return agent


def _render_frame(mujoco_env, env_idx: int) -> np.ndarray:
    """Render a single RGB frame (640x360) from the front camera for a given env."""
    import mujoco as mj
    env = mujoco_env._envs[env_idx]
    data = env["data"]
    renderer = env["renderer_base"]  # 640x360 cam_base (overview)
    cam_id = env["cam_base_id"]
    renderer.update_scene(data, camera=cam_id)
    return renderer.render().copy()  # (360, 640, 3) uint8


def _save_video(frames: list[np.ndarray], path: str, fps: int = 30):
    """Save frames as mp4 using cv2."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    _log(f"  Saved video: {path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")


def run_video_rollouts(env, mujoco_env, agent, num_rounds: int, max_steps: int,
                       num_envs: int, device: torch.device, mode: str,
                       output_dir: Path, task: str, position_indices: list[int]):
    """Run rollouts with video capture."""
    agent.eval()

    for rd in range(num_rounds):
        _log(f"Round {rd}/{num_rounds}")
        obs, _ = env.reset()

        # Per-env frame buffers
        frames = [[] for _ in range(num_envs)]
        env_done = [False] * num_envs
        env_success = [False] * num_envs
        ep_lengths = [0] * num_envs

        # Capture initial frame
        for i in range(num_envs):
            frames[i].append(_render_frame(mujoco_env, i))

        for step in range(max_steps):
            with torch.no_grad():
                if mode == "base":
                    action = torch.zeros(num_envs, env.action_dim, device=device)
                else:
                    action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for i in range(num_envs):
                if not env_done[i]:
                    frames[i].append(_render_frame(mujoco_env, i))
                    ep_lengths[i] = step + 1
                    if done[i]:
                        env_done[i] = True
                        if terminated[i]:
                            env_success[i] = True

            if all(env_done):
                break

        # Save videos for this round
        for i in range(num_envs):
            pos_idx = position_indices[i]
            success_str = "succ" if env_success[i] else "fail"
            fname = f"{task}_{mode}_pos{pos_idx}_r{rd}_{success_str}.mp4"
            _save_video(frames[i], str(output_dir / fname), fps=30)

        sr = sum(env_success) / num_envs
        _log(f"  Round {rd} SR: {sr:.0%} ({sum(env_success)}/{num_envs})")


def parse_args():
    p = argparse.ArgumentParser(description="Rollout video collector")
    p.add_argument("--task", type=str, required=True, choices=["lift", "pnp", "stack"])
    p.add_argument("--mode", type=str, required=True, choices=["base", "residual"])
    p.add_argument("--positions", type=str, required=True,
                   help="Comma-separated position indices, e.g. '0,5,6,8,14,15'")
    p.add_argument("--output_dir", type=str, default="outputs/trajectory_videos")
    p.add_argument("--num_rounds", type=int, default=3,
                   help="Number of eval rounds per position set")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    position_indices = [int(x) for x in args.positions.split(",")]
    cfg = TASK_CONFIGS[args.task].copy()

    _log(f"Task={args.task}  Mode={args.mode}  Positions={position_indices}  Rounds={args.num_rounds}")

    # Load fixed positions
    positions = _load_positions(cfg, position_indices)
    num_envs = len(positions)

    # Create env
    _log("Creating environment...")
    env, mujoco_env = _create_env(args.task, cfg, positions, args.device)
    _log(f"Environment ready ({num_envs} envs).")

    # Create agent
    agent = _create_agent(env, cfg, device)
    agent.to(device)

    # Load checkpoint
    if args.mode == "residual":
        ckpt_path = cfg["residual_checkpoint"]
        _log(f"Loading residual checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        agent.load_state_dict(ckpt, strict=False)
        _log("Residual checkpoint loaded.")
    else:
        _log("Base-only mode — residual will be zeros.")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run
    _log("Starting video rollouts...")
    t0 = time.time()
    run_video_rollouts(
        env=env,
        mujoco_env=mujoco_env,
        agent=agent,
        num_rounds=args.num_rounds,
        max_steps=cfg["max_episode_steps"],
        num_envs=num_envs,
        device=device,
        mode=args.mode,
        output_dir=output_dir,
        task=args.task,
        position_indices=position_indices,
    )
    elapsed = time.time() - t0
    _log(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

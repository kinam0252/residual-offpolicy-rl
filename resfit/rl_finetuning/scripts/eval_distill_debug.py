"""
Eval a distilled (or image-RL) checkpoint and save input images for visual inspection.

Usage:
  python eval_distill_debug.py \
    --checkpoint outputs/lift_rl_1803/checkpoints/best.pt \
    --task lift \
    --groot_checkpoint checkpoints/<TASK>/checkpoint \
    --num_envs 5 --num_episodes 1 \
    --save_dir outputs/lift_rl_1803/debug_images
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types as _types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified
from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.configs.task_configs import get_task_config, TaskConfig


def _log(msg):
    print(f"[eval-debug] {msg}", flush=True)


def save_obs_images(obs: dict, step: int, env_idx: int, save_dir: Path, cam_keys: list[str]):
    """Save observation images as PNGs."""
    for cam_key in cam_keys:
        if cam_key not in obs:
            continue
        img_tensor = obs[cam_key]  # (num_envs, C, H, W) or (C, H, W)
        if img_tensor.dim() == 4:
            img = img_tensor[env_idx].cpu()
        else:
            img = img_tensor.cpu()

        # (C, H, W) -> (H, W, C)
        # Handle both uint8 (0-255) and float (0-1)
        img_np = img.permute(1, 2, 0).numpy()
        if img.dtype == torch.float32 or img.dtype == torch.float16:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        cam_name = cam_key.split(".")[-1]  # e.g. "back", "wrist"
        fname = save_dir / f"env{env_idx}_step{step:04d}_{cam_name}.png"
        Image.fromarray(img_np).save(fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--groot_checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=300)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save images every N steps (to avoid too many files)")
    parser.add_argument("--positions_file", type=str, default=None)
    parser.add_argument("--rl_img_size", type=int, default=84)
    parser.add_argument("--reward_type", type=str, default="dense")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    task_cfg = get_task_config(args.task)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint).expanduser()
    _log(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    _log(f"  SR={ckpt.get('success_rate', '?')}, step={ckpt.get('eval_step', '?')}")
    _log(f"  distill={ckpt_args.get('distill')}, realistic={ckpt_args.get('realistic')}")

    # Save dir
    save_dir = Path(args.save_dir) if args.save_dir else ckpt_path.parent.parent / "debug_images"
    save_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Saving images to: {save_dir}")

    # Create VecEnv
    _log("Creating VecEnv...")
    mod = importlib.import_module(task_cfg.vec_env_module)
    VecEnvClass = getattr(mod, task_cfg.vec_env_class)

    vec_env_kwargs = dict(
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        device=args.device,
    )

    if args.task == "lift":
        cube_positions = [[0.45, -0.05, 0.02]] * args.num_envs
        if args.positions_file:
            with open(args.positions_file) as f:
                all_pos = json.load(f)
            eval_pos = all_pos[-args.num_envs:]
            cube_positions = [p["cube_pos"] for p in eval_pos]
        vec_env_kwargs["cube_positions"] = cube_positions
    elif args.task == "pnp":
        cube_positions = [[0.45, -0.05, 0.02]] * args.num_envs
        if args.positions_file:
            with open(args.positions_file) as f:
                all_pos = json.load(f)
            eval_pos = all_pos[-args.num_envs:]
            cube_positions = [p["cube_pos"] for p in eval_pos]
        vec_env_kwargs["cube_positions"] = cube_positions
        vec_env_kwargs["rl_img_size"] = args.rl_img_size
    elif args.task == "stack":
        if args.positions_file:
            with open(args.positions_file) as f:
                all_pos = json.load(f)
            eval_pos = all_pos[-args.num_envs:]
            vec_env_kwargs["white_cube_positions"] = [p["white_cube_pos"] for p in eval_pos]
            vec_env_kwargs["green_cube_positions"] = [p["green_cube_pos"] for p in eval_pos]

    mujoco_env = VecEnvClass(**vec_env_kwargs)
    _log(f"VecEnv created: {args.num_envs} envs")

    # Wrapper config
    groot_ckpt = str(Path(args.groot_checkpoint).expanduser())
    action_scale = ckpt_args.get("action_scale", 0.1)
    env = MuJoCoResidualWrapperUnified(
        vec_env=mujoco_env,
        groot_checkpoint=groot_ckpt,
        embodiment_tag=ckpt_args.get("groot_embodiment_tag", "new_embodiment"),
        policy_device=str(device),
        task_description=task_cfg.task_description,
        open_loop_horizon=ckpt_args.get("open_loop_horizon", 1),
        residual_pos_scale=ckpt_args.get("residual_pos_scale", 1.0),
        residual_rot_scale=ckpt_args.get("residual_rot_scale", 1.0),
        residual_grip_scale=ckpt_args.get("residual_grip_scale", 1.0),
        ema_alpha=ckpt_args.get("ema_alpha", 1.0),
        chunk_sync=ckpt_args.get("chunk_sync", True),
        grip_min=task_cfg.grip_min,
        grip_max=task_cfg.grip_max,
        use_gripper_latch=task_cfg.use_gripper_latch,
        grip_close_latch_thresh=task_cfg.grip_close_latch_thresh,
        grip_open_latch_thresh=task_cfg.grip_open_latch_thresh,
        grip_latch_open_steps=task_cfg.grip_latch_open_steps,
        camera_keys=task_cfg.camera_keys,
        async_prefetch=False,
        use_images=True,
        realistic=True,
        realistic_task=args.task,
        rl_img_size=args.rl_img_size,
    )
    _log("Wrapper ready (realistic rendering ON)")

    # Image keys
    cam_keys = task_cfg.rl_image_keys
    _log(f"Image keys: {cam_keys}")

    # Build agent
    lowdim_dim = 10
    object_state_dim = task_cfg.object_state_dim
    action_dim = 7

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = action_scale
    actor_hd = ckpt_args.get("actor_hidden_dim", None)
    if actor_hd:
        cfg.agent.actor.hidden_dim = actor_hd
    critic_hd = ckpt_args.get("critic_hidden_dim", None)
    if critic_hd:
        cfg.agent.critic.hidden_dim = critic_hd

    agent = QAgent(
        obs_shape=(3, args.rl_img_size, args.rl_img_size),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=cam_keys,
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
    )

    # Load weights
    _missing, _unexpected = agent.load_state_dict(ckpt["model"], strict=False)
    _log(f"Loaded weights: missing={len(_missing)}, unexpected={len(_unexpected)}")
    agent.to(device)
    agent.eval()

    # Run eval
    _log(f"Running {args.num_episodes} episode(s) × {args.num_envs} envs...")
    total_success = 0
    total_episodes = 0
    img_count = 0

    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        env_done = [False] * args.num_envs
        env_success = [False] * args.num_envs

        # Save initial images
        for ei in range(args.num_envs):
            save_obs_images(obs, 0, ei, save_dir, cam_keys)
            img_count += len(cam_keys)

        for step in range(1, args.max_episode_steps + 1):
            with torch.no_grad():
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            # Save images periodically
            if step % args.save_every == 0:
                for ei in range(min(args.num_envs, 2)):  # save only first 2 envs to limit files
                    if not env_done[ei]:
                        save_obs_images(obs, step, ei, save_dir, cam_keys)
                        img_count += len(cam_keys)

            for i in range(args.num_envs):
                if not env_done[i] and done[i]:
                    env_done[i] = True
                    if terminated[i]:
                        env_success[i] = True
                        _log(f"  ep={ep} env={i} SUCCESS at step {step}")
                    else:
                        _log(f"  ep={ep} env={i} TIMEOUT at step {step}")

            if all(env_done):
                break

        for i in range(args.num_envs):
            if env_success[i]:
                total_success += 1
            total_episodes += 1

    sr = total_success / max(1, total_episodes)
    _log(f"\n{'='*50}")
    _log(f"Results: {total_success}/{total_episodes} success = {sr:.1%}")
    _log(f"Images saved: {img_count} files in {save_dir}")
    _log(f"{'='*50}")

    env.close()


if __name__ == "__main__":
    main()

"""Replay failed episodes from offline collection with video output.

Reruns GR00T base policy (residual=0) on the same env config and renders video
to diagnose why episodes fail.

Usage (SLURM):
    sbatch scripts/slurm_replay_fail_videos.sh
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ['MUJOCO_GL'] = 'egl'

import torch
torch.backends.cuda.enable_cudnn_sdp(False)

import importlib, types as _types
if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed"); _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__ = []; _ds.__file__ = __file__
    _dz = _types.ModuleType("deepspeed.zero")
    _dz.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _dz.Init = lambda *a, **kw: (lambda f: f)
    _ds.zero = _dz
    sys.modules["deepspeed"] = _ds
    sys.modules["deepspeed.zero"] = _dz
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig = _hf_val.validate_repo_id
    def _pv(repo_id):
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")): return
        return _orig(repo_id)
    _hf_val.validate_repo_id = _pv
except:
    pass

import argparse
import numpy as np
import cv2
import imageio
import time
from pathlib import Path

from resfit.rl_finetuning.wrappers.mujoco_vec_env_stack import (
    MuJoCoVecEnvStack, CUBE_HALF, get_tcp_pose,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_stack import MuJoCoResidualWrapperStack


FPS = 15


def render_frame(vec_env, env_idx=0):
    """Render base camera frame (H, W, 3) RGB."""
    e = vec_env._envs[env_idx]
    e["renderer_base"].update_scene(e["data"], camera=e["cam_base_id"],
                                    scene_option=e["opt_base"])
    return e["renderer_base"].render().copy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--offline_dir", type=str, required=True,
                   help="Directory with .npz files from collection")
    p.add_argument("--output_dir", type=str, default="outputs/stack_fail_videos")
    p.add_argument("--max_videos", type=int, default=5,
                   help="Max number of failure videos to render")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--white_pos", type=float, nargs=3, default=[0.45, 0.22, 0.02])
    p.add_argument("--green_pos", type=float, nargs=3, default=[0.45, -0.06, 0.02])
    p.add_argument("--max_episode_steps", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    offline_dir = Path(args.offline_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find failed episodes that had grasp
    fail_eps = []
    for f in sorted(offline_dir.glob("*.npz")):
        d = np.load(str(f))
        if not d["success"][0] and d["grasped"].any():
            grasp_step = int(np.argmax(d["grasped"]))
            fail_eps.append((f.name, grasp_step))
    
    print(f"Found {len(fail_eps)} 'grasped but failed' episodes")
    fail_eps = fail_eps[:args.max_videos]
    print(f"Rendering {len(fail_eps)} videos to {args.output_dir}")

    # Create env + wrapper (reuse across episodes)
    vec_env = MuJoCoVecEnvStack(
        num_envs=1,
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
        white_cube_positions=[args.white_pos],
        green_cube_positions=[args.green_pos],
    )
    wrapper = MuJoCoResidualWrapperStack(
        vec_env=vec_env,
        groot_checkpoint=args.groot_checkpoint,
        ema_alpha=0.9,
        policy_device=args.device,
    )

    for idx, (ep_name, grasp_step) in enumerate(fail_eps):
        print(f"\n[{idx+1}/{len(fail_eps)}] Replaying {ep_name} (grasp@step{grasp_step})...")
        
        obs, info = wrapper.reset()
        vid_path = str(out_dir / f"fail_{ep_name.replace('.npz', '.mp4')}")
        writer = imageio.get_writer(
            vid_path, fps=FPS, codec="libx264",
            macro_block_size=1,
            output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
        )

        residual = torch.zeros((1, 7), dtype=torch.float32)
        grasped_in_replay = False
        max_reward = 0.0

        for step in range(args.max_episode_steps):
            next_obs, reward, terminated, truncated, info = wrapper.step(residual)
            r = float(reward[0]) if hasattr(reward, "__getitem__") else float(reward)
            max_reward = max(max_reward, r)

            # Check grasp state
            env = vec_env._envs[0]
            currently_grasped = env["grasp_state"]["grasped"]
            if currently_grasped and not grasped_in_replay:
                grasped_in_replay = True
                print(f"  Grasped at step {step}")

            # Render frame with overlay
            frame = render_frame(vec_env)
            # Add text overlay
            status = "GRASPED" if currently_grasped else "no grasp"
            text = f"{ep_name} f={step}/{args.max_episode_steps} r={r:.3f} {status}"
            # Convert to BGR for cv2 text, then back
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

            obs = next_obs
            if terminated[0] or truncated[0]:
                break

        writer.close()
        tag = "SUCC" if terminated[0] else "FAIL"
        print(f"  {tag} steps={step+1}, max_r={max_reward:.3f}, grasped={grasped_in_replay}")
        print(f"  Video: {vid_path}")

    print(f"\nDone. {len(fail_eps)} videos saved to {args.output_dir}")
    try:
        vec_env.close()
    except:
        pass


if __name__ == "__main__":
    main()

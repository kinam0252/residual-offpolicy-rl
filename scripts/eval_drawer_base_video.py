"""
Base-only drawer eval with video recording.

Runs GR00T base policy (residual=0) on D3/D4, saves cam_base frames as MP4.

Usage:
    cd ~/Repos/Intern/residual-offpolicy-rl
    NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
    LD_LIBRARY_PATH="$HOME/lib-compat:$HOME/.local/lib/gl:/usr/lib/x86_64-linux-gnu:$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cublas/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:$HOME/.local/lib:${LD_LIBRARY_PATH:-}" \
    MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    ~/.venvs/groot/bin/python3 scripts/eval_drawer_base_video.py --drawer 3
"""
from __future__ import annotations

import importlib
import os
import sys
import types as _types

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DS_BUILD_OPS", "0")

if "deepspeed" not in sys.modules:
    _ds = _types.ModuleType("deepspeed")
    _ds.__version__ = "0.0.0"
    _ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds.__path__ = []
    sys.modules["deepspeed"] = _ds

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resfit.rl_finetuning.wrappers.mujoco_vec_env_drawer import (
    MuJoCoVecEnvDrawer,
    DRAWER_SLIDE,
)
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import (
    MuJoCoResidualWrapperUnified,
)


def run_episode_with_video(
    drawer_id: int,
    groot_ckpt: str,
    max_steps: int,
    device: str,
    out_dir: Path,
    episode: int,
):
    """Run one episode, save cam_base video."""
    vec_env = MuJoCoVecEnvDrawer(
        num_envs=1,
        active_drawers=[drawer_id],
        max_episode_steps=max_steps,
        reward_type="delta",
        device=device,
        rl_img_size=84,
        parallel_envs=False,
    )
    wrapper = MuJoCoResidualWrapperUnified(
        vec_env=vec_env,
        groot_checkpoint=groot_ckpt,
        task_description="Close the drawer.",
        chunk_sync=True,
        async_prefetch=False,
        policy_device=device,
    )

    obs, info = wrapper.reset()
    zero_act = torch.zeros(1, 7, device=device)
    frames = []
    success = False

    for step in range(max_steps):
        # Render cam_base at full res (640×360) before stepping
        env = vec_env._envs[0]
        import mujoco

        env["renderer_base"].update_scene(
            env["data"], camera=env["cam_base_id"], scene_option=env["opt_base"]
        )
        frame = env["renderer_base"].render().copy()  # (360, 640, 3) RGB
        frames.append(frame)

        obs, reward, terminated, truncated, info = wrapper.step(zero_act)

        if step % 50 == 0:
            dq = vec_env._drawer_qpos[0]
            tcp = env["data"].xpos[env["ids"]["hand_id"]]
            ba = wrapper._held_base_action[0]
            print(
                f"  s{step}: dq={dq:.4f} tcp=[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}] "
                f"base=[{ba[0]:.3f},{ba[1]:.3f},{ba[2]:.3f}] r={float(reward[0]):.3f}"
            )

        if terminated[0]:
            success = True
            # Capture final frame
            env["renderer_base"].update_scene(
                env["data"], camera=env["cam_base_id"], scene_option=env["opt_base"]
            )
            frames.append(env["renderer_base"].render().copy())
            print(f"  => SUCCESS at step {step}")
            break
        if truncated[0]:
            print(f"  => TRUNCATED at step {step}")
            break

    # Save video (H264/avc1 via imageio-ffmpeg)
    tag = "success" if success else "fail"
    video_path = out_dir / f"D{drawer_id}_ep{episode}_{tag}.mp4"
    h, w = frames[0].shape[:2]
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    writer = imageio_ffmpeg.write_frames(
        str(video_path),
        (w, h),
        fps=20.0,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "23"],
        ffmpeg_path=ffmpeg_exe,
    )
    writer.send(None)  # init
    for f in frames:
        writer.send(f.tobytes())
    writer.close()
    print(f"  Saved {video_path} ({len(frames)} frames)")

    del wrapper, vec_env
    torch.cuda.empty_cache()
    return success


def main():
    parser = argparse.ArgumentParser(description="Base-only drawer eval with video")
    parser.add_argument(
        "--drawer", type=int, nargs="+", default=[2, 3, 4],
        help="Drawer IDs (e.g. 2 3 4)"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--groot_checkpoint",
        type=str,
        default="~/DATA/INTERN/training/groot_drawer_sim_33ep/checkpoint-100000",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--physics_drawer", action="store_true",
                        help="Use MuJoCo physics for drawer instead of kinematic")
    parser.add_argument(
        "--out_dir", type=str, default="outputs/drawer_base_video"
    )
    args = parser.parse_args()

    groot_ckpt = str(Path(args.groot_checkpoint).expanduser())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Drawer Base Policy Eval (video) — D{args.drawer} ===")
    print(f"GR00T: {groot_ckpt}")
    print(f"Output: {out_dir}")

    all_results = {}
    for drawer_id in args.drawer:
        print(f"\n{'='*50}")
        print(f"  Drawer {drawer_id}")
        print(f"{'='*50}")

        # Create env + wrapper per drawer (active_drawers differs)
        vec_env = MuJoCoVecEnvDrawer(
            num_envs=1,
            active_drawers=[drawer_id],
            max_episode_steps=args.max_steps,
            reward_type="delta",
            device=args.device,
            rl_img_size=84,
            parallel_envs=False,
            physics_drawer=args.physics_drawer,
        )
        wrapper = MuJoCoResidualWrapperUnified(
            vec_env=vec_env,
            groot_checkpoint=groot_ckpt,
            task_description="Close the drawer.",
            chunk_sync=True,
            async_prefetch=False,
            policy_device=args.device,
        )
        zero_act = torch.zeros(1, 7, device=args.device)
        import mujoco  # noqa: E402

        results = []
        for ep in range(args.episodes):
            print(f"\n--- D{drawer_id} Episode {ep} ---")
            obs, info = wrapper.reset()
            frames = []
            success = False
            env = vec_env._envs[0]

            for step in range(args.max_steps):
                # Render both cameras and concat side-by-side
                env["renderer_base"].update_scene(
                    env["data"], camera=env["cam_base_id"], scene_option=env["opt_base"]
                )
                frame_base = env["renderer_base"].render().copy()
                env["renderer_wrist"].update_scene(
                    env["data"], camera=env["cam_wrist_id"], scene_option=env["opt_base"]
                )
                frame_wrist = env["renderer_wrist"].render().copy()
                if frame_wrist.shape[0] != frame_base.shape[0]:
                    h_base = frame_base.shape[0]
                    scale = h_base / frame_wrist.shape[0]
                    w_new = int(frame_wrist.shape[1] * scale)
                    frame_wrist = cv2.resize(frame_wrist, (w_new, h_base))
                frames.append(np.concatenate([frame_base, frame_wrist], axis=1))

                obs, reward, terminated, truncated, info = wrapper.step(zero_act)

                if step % 50 == 0:
                    dq = vec_env._drawer_qpos[0]
                    tcp = env["data"].xpos[env["ids"]["hand_id"]]
                    ba = wrapper._held_base_action[0]
                    print(
                        f"  s{step}: dq={dq:.4f} tcp=[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}] "
                        f"base=[{ba[0]:.3f},{ba[1]:.3f},{ba[2]:.3f}] r={float(reward[0]):.3f}"
                    )

                if terminated[0]:
                    success = True
                    env["renderer_base"].update_scene(
                        env["data"], camera=env["cam_base_id"], scene_option=env["opt_base"]
                    )
                    fb = env["renderer_base"].render().copy()
                    env["renderer_wrist"].update_scene(
                        env["data"], camera=env["cam_wrist_id"], scene_option=env["opt_base"]
                    )
                    fw = env["renderer_wrist"].render().copy()
                    if fw.shape[0] != fb.shape[0]:
                        h_b = fb.shape[0]
                        s = h_b / fw.shape[0]
                        fw = cv2.resize(fw, (int(fw.shape[1] * s), h_b))
                    frames.append(np.concatenate([fb, fw], axis=1))
                    print(f"  => SUCCESS at step {step}")
                    break
                if truncated[0]:
                    print(f"  => TRUNCATED at step {step}")
                    break

            # Save video
            tag = "success" if success else "fail"
            video_path = out_dir / f"D{drawer_id}_ep{ep}_{tag}.mp4"
            h, w = frames[0].shape[:2]
            vwriter = imageio_ffmpeg.write_frames(
                str(video_path), (w, h), fps=20.0, codec="libx264",
                pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                output_params=["-crf", "23"],
            )
            vwriter.send(None)
            for f in frames:
                vwriter.send(f.tobytes())
            vwriter.close()
            print(f"  Saved {video_path} ({len(frames)} frames)")
            results.append(success)

        sr = sum(results) / len(results)
        print(f"\n  D{drawer_id}: {sum(results)}/{len(results)} = {sr*100:.0f}%")
        all_results[drawer_id] = (sum(results), len(results))

        del wrapper, vec_env
        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print("  Summary")
    print(f"{'='*50}")
    for d, (s, t) in all_results.items():
        print(f"  D{d}: {s}/{t} = {s/t*100:.0f}%")


if __name__ == "__main__":
    main()

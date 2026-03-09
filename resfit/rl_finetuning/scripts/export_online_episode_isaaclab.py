from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from isaaclab.app import AppLauncher

from resfit.rl_finetuning.policies.groot_policy import GR00TBasePolicy
from resfit.rl_finetuning.wrappers.isaaclab_env_wrapper import create_isaaclab_env
from resfit.rl_finetuning.wrappers.isaaclab_groot_rollout_env import IsaacLabGrootRolloutEnv


parser = argparse.ArgumentParser(description="Export one online buffer episode from IsaacLab + local GR00T")
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
parser.add_argument(
    "--output_npz",
    type=str,
    default="/home/t-kinamkim/Repos/VLA_RL/residual-offpolicy-rl/workspace/online_buffer_exports/online_episode_000.npz",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _to_chw_uint8_84(frame_hwc: np.ndarray) -> np.ndarray:
    frame = torch.from_numpy(frame_hwc)
    if frame.dtype != torch.uint8:
        if frame.max() <= 1.0:
            frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
        else:
            frame = frame.clamp(0, 255).to(torch.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    frame = frame.permute(2, 0, 1).contiguous().float().unsqueeze(0)
    frame = F.interpolate(frame, size=(84, 84), mode="bilinear", align_corners=False)
    return frame.squeeze(0).to(torch.uint8).cpu().numpy()


def main() -> None:
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    extra_cfg_overrides = {
        "use_api_for_pose": False,
        "gr00t_host": args_cli.gr00t_host,
        "gr00t_port": args_cli.gr00t_port,
    }
    if args_cli.csv_base_dir:
        extra_cfg_overrides["csv_base_dir"] = args_cli.csv_base_dir
        extra_cfg_overrides["csv_init_row_index"] = args_cli.csv_init_row_index

    vec_env = create_isaaclab_env(
        task="Isaac-Franka-Pickup-Direct-v0",
        num_envs=args_cli.num_envs,
        enable_cameras=True,
        device=args_cli.device,
        image_size=(84, 84),
        extra_cfg_overrides=extra_cfg_overrides,
    )

    policy_device = args_cli.groot_policy_device if args_cli.groot_policy_device else args_cli.device
    base_policy = GR00TBasePolicy(
        host=args_cli.gr00t_host,
        port=args_cli.gr00t_port,
        num_envs=args_cli.num_envs,
        device=policy_device,
        task_description=args_cli.task_description,
        language_override=args_cli.language_override,
        model_path=args_cli.groot_model_path,
        embodiment_tag=args_cli.groot_embodiment_tag,
        strict=args_cli.groot_policy_strict,
    )

    env = IsaacLabGrootRolloutEnv(vec_env=vec_env, base_policy=base_policy)
    obs, _ = env.reset()

    timestep = []
    state = []
    base_action = []
    action = []
    reward = []
    done = []
    front_image = []
    render_used = 0
    obs_fallback_used = 0
    zero_frame_count = 0

    for step in range(args_cli.max_steps):
        timestep.append(step)
        state.append(obs["observation.state"][0].detach().cpu().numpy())
        base_action.append(obs["observation.base_action"][0].detach().cpu().numpy())
        front_obs = None

        rendered = vec_env.render()
        if isinstance(rendered, np.ndarray) and rendered.ndim == 4 and rendered.shape[0] > 0:
            render_chw = _to_chw_uint8_84(rendered[0])
            if int(render_chw.max()) > 0:
                front_obs = render_chw
                render_used += 1

        if front_obs is None:
            front_obs = obs["observation.images.front"][0].detach().cpu().numpy()
            obs_fallback_used += 1

        if int(front_obs.max()) == 0:
            zero_frame_count += 1
        front_image.append(front_obs)

        residual = torch.zeros((args_cli.num_envs, 7), device=env.device, dtype=torch.float32)
        next_obs, rew, terminated, truncated, info = env.step(residual)

        combined = info.get("scaled_action", residual)
        d = (terminated | truncated)

        action.append(combined[0].detach().cpu().numpy())
        reward.append(float(rew[0].item()))
        done.append(bool(d[0].item()))

        obs = next_obs
        if done[-1]:
            break

    env.close()

    out_path = Path(args_cli.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        timestep=np.asarray(timestep, dtype=np.int32),
        state=np.asarray(state, dtype=np.float32),
        base_action=np.asarray(base_action, dtype=np.float32),
        action=np.asarray(action, dtype=np.float32),
        reward=np.asarray(reward, dtype=np.float32),
        done=np.asarray(done, dtype=np.bool_),
        front_image=np.asarray(front_image, dtype=np.uint8),
    )

    print(f"saved_npz={out_path}")
    print(f"steps={len(timestep)} done={bool(done[-1]) if done else False}")
    print(f"render_used={render_used}")
    print(f"obs_fallback_used={obs_fallback_used}")
    print(f"zero_frame_count={zero_frame_count}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

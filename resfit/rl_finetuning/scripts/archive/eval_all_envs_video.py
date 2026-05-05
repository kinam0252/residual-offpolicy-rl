"""Evaluate checkpoint on all envs with cam_base video recording."""
import argparse, json, sys, os, time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


def evaluate_all_envs(env, agent, num_episodes, device, image_keys, video_dir, camera="back", base_only=False):
    agent.eval()
    num_envs = env.vec_env.num_envs
    per_env_sr = [0] * num_envs

    for ep in range(num_episodes):
        obs, _ = env.reset()
        env_done = [False] * num_envs
        env_success = [False] * num_envs
        ep_frames = [[] for _ in range(num_envs)]

        for step_i in range(env.vec_env.max_episode_steps):
            # Capture frames
            for ei in range(num_envs):
                if not env_done[ei]:
                    frame = env.vec_env.get_frame(ei, camera=camera, size=(360, 640))
                    ep_frames[ei].append(frame)

            with torch.no_grad():
                if base_only:
                    action = torch.zeros(num_envs, env.action_dim, device=device)
                else:
                    action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            for ei in range(num_envs):
                if not env_done[ei] and done[ei]:
                    env_done[ei] = True
                    if terminated[ei]:
                        env_success[ei] = True

            if all(env_done):
                break

        # Save videos
        if video_dir and imageio is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
            for ei in range(num_envs):
                if ep_frames[ei]:
                    status = 'ok' if env_success[ei] else 'fail'
                    vid_path = video_dir / f'env{ei:02d}_ep{ep}_{status}.mp4'
                    imageio.mimwrite(str(vid_path), ep_frames[ei], fps=30)

        for ei in range(num_envs):
            if env_success[ei]:
                per_env_sr[ei] += 1

        n_succ = sum(1 for s in env_success if s)
        print(f'  Episode {ep}: {n_succ}/{num_envs} success', flush=True)

    sr_per_env = [s / num_episodes for s in per_env_sr]
    total_sr = sum(per_env_sr) / (num_envs * num_episodes)

    print(f'\nTotal SR: {total_sr:.1%}', flush=True)
    for ei in range(num_envs):
        print(f'  env{ei:02d}: {sr_per_env[ei]:.0%}', flush=True)

    return total_sr, sr_per_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--groot_checkpoint', type=str, required=True)
    p.add_argument('--perturb_table', type=str, required=True)
    p.add_argument('--eval_num_episodes', type=int, default=1)
    p.add_argument('--max_episode_steps', type=int, default=500)
    p.add_argument('--reward_type', type=str, default='dense_clipped')
    p.add_argument('--action_scale', type=float, default=1.0)
    p.add_argument('--actor_hidden_dim', type=int, default=256)
    p.add_argument('--critic_hidden_dim', type=int, default=256)
    p.add_argument('--asymmetric_critic', action='store_true')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--video_dir', type=str, required=True)
    p.add_argument('--camera', type=str, default='back', choices=['front', 'back', 'wrist'])
    p.add_argument('--calib_path', type=str, default=None)
    p.add_argument('--use_calibrated_wrist', action='store_true', default=False)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--base_only', action='store_true', default=False)
    p.add_argument('--success_threshold', type=float, default=0.03)
    args = p.parse_args()

    from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig

    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.perturb_table) as f:
        ptable = json.load(f)
    base_pos = ptable['base_pos']
    envs_cfg = ptable['envs']
    cube_positions = [
        [base_pos[0] + e['dx'], base_pos[1] + e['dy'], base_pos[2]]
        for e in envs_cfg
    ]
    num_envs = len(cube_positions)
    print(f'Loaded {num_envs} env positions', flush=True)

    mujoco_env = MuJoCoVecEnv(
        num_envs=num_envs,
        cube_positions=cube_positions,
        calib_path=args.calib_path,
        use_calibrated_wrist=args.use_calibrated_wrist,
        max_episode_steps=args.max_episode_steps,
        success_threshold=args.success_threshold,
        reward_type=args.reward_type,
        device=args.device,
    )

    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        policy_device=args.device,
    )

    obs, _ = env.reset()

    image_keys = ['observation.depth.front', 'observation.depth.wrist']
    lowdim_dim = env.observation_space['observation.state'].shape[1]
    action_dim = env.action_dim

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim

    agent = QAgent(
        obs_shape=(1, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=7,
        asymmetric_critic=True,
    )
    agent.to(device)

    if not args.base_only:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'actor' in ckpt and isinstance(ckpt['actor'], dict):
            agent.actor.load_state_dict(ckpt['actor'])
            agent.critic.load_state_dict(ckpt['critic'])
            if 'encoders' in ckpt:
                agent.encoders.load_state_dict(ckpt['encoders'])
            print(f'Loaded structured checkpoint', flush=True)
        else:
            agent.load_state_dict(ckpt, strict=False)
            print(f'Loaded flat state_dict', flush=True)
    else:
        print('Base-only mode: using zero residual actions', flush=True)

    video_dir = Path(args.video_dir)
    print(f'Evaluating {num_envs} envs x {args.eval_num_episodes} episodes, camera={args.camera}', flush=True)

    t0 = time.time()
    total_sr, sr_per_env = evaluate_all_envs(
        env, agent, args.eval_num_episodes, device, image_keys,
        video_dir=video_dir, camera=args.camera, base_only=args.base_only,
    )
    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s. Videos saved to {video_dir}', flush=True)

if __name__ == '__main__':
    main()

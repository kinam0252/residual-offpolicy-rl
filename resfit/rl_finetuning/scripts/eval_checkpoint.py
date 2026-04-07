"""Standalone checkpoint evaluation with multi-seed averaging.

Usage:
  python -m resfit.rl_finetuning.scripts.eval_checkpoint \
    --checkpoint best.pt \
    --perturb_table configs/mujoco_cube_perturb_table.json \
    --groot_checkpoint .../checkpoint-30000 \
    --eval_num_episodes 2 \
    --seed 42 \
    --action_scale 0.2 \
    --actor_hidden_dim 512 --critic_hidden_dim 1024 \
    --asymmetric_critic \
    --reward_type dense \
    --max_episode_steps 500
"""
import argparse, json, sys, os, time, traceback
import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint .pt file, or 'none' for base policy")
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--perturb_table", type=str, required=True)
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--eval_num_episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--reward_type", type=str, default="dense")
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--residual_pos_scale", type=float, default=0.02)
    p.add_argument("--residual_rot_scale", type=float, default=0.05)
    p.add_argument("--residual_grip_scale", type=float, default=0.1)
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--critic_hidden_dim", type=int, default=1024)
    p.add_argument("--asymmetric_critic", action="store_true")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--label", type=str, default="eval")
    p.add_argument("--groot_embodiment_tag", type=str, default="NEW_EMBODIMENT")
    p.add_argument("--task_description", type=str, default="lift the cube")
    p.add_argument("--open_loop_horizon", type=int, default=16)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    p.add_argument("--success_threshold", type=float, default=0.03)
    p.add_argument("--calib_path", type=str, default=None)
    p.add_argument("--cube_yaw", type=float, default=0.0, help="Cube yaw rotation in degrees")
    p.add_argument("--hover_offset", type=str, default=None,
                   help="JSON: {xy:[dx,dy]} or {random_xy:0.05} or {z:0.30} or {random_z:[0.20,0.30]} in meters")
    p.add_argument("--obj_pos_noise", type=float, default=0.0, help="Gaussian noise std for object position (meters)")
    p.add_argument("--obj_rot_noise", type=float, default=0.0, help="Gaussian noise std for object rotation (degrees)")
    return p.parse_args()


def main():
    args = parse_args()
    sys.stdout.flush()

    from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
    from resfit.rl_finetuning.scripts.eval_async_mujoco import evaluate

    device = torch.device(args.device)

    # Parse perturb table (same logic as training script)
    with open(args.perturb_table) as f:
        ptable = json.load(f)
    base_pos = ptable["base_pos"]
    envs_cfg = ptable["envs"]
    cube_positions = [
        [base_pos[0] + e["dx"], base_pos[1] + e["dy"], base_pos[2]]
        for e in envs_cfg
    ]
    args.num_envs = len(cube_positions)
    print(f"Loaded {len(cube_positions)} env positions from {args.perturb_table}", flush=True)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    seed = args.seed
    print(f"\nSeed: {seed}", flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Creating MuJoCo env...", flush=True)
    # Parse hover offset
    hover_offset = None
    if args.hover_offset:
        import json as _json
        hover_offset = _json.loads(args.hover_offset)
        print(f"Hover offset: {hover_offset}", flush=True)

    mujoco_env = MuJoCoVecEnv(
        num_envs=args.num_envs,
        cube_positions=cube_positions,
        cube_yaw_deg=args.cube_yaw,
        hover_offset=hover_offset,
        scene_xml=args.scene_xml,
        calib_path=args.calib_path,
        max_episode_steps=args.max_episode_steps,
        success_threshold=args.success_threshold,
        reward_type=args.reward_type,
        device=args.device,
    )

    print("Creating residual wrapper with GR00T...", flush=True)
    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=args.device,
        task_description=args.task_description,
        open_loop_horizon=args.open_loop_horizon,
        residual_pos_scale=args.residual_pos_scale,
        residual_rot_scale=args.residual_rot_scale,
        residual_grip_scale=args.residual_grip_scale,
        ema_alpha=args.ema_alpha,
    )

    print("Running initial reset...", flush=True)
    obs, _ = env.reset()

    _asymmetric = args.asymmetric_critic
    object_state_dim = 7 if _asymmetric else 0

    if _asymmetric:
        image_keys = ["observation.depth.front", "observation.depth.wrist"]
        img_c, img_h, img_w = 1, 84, 84
    else:
        image_keys = ["observation.images.front", "observation.images.back", "observation.images.wrist"]
        img_c, img_h, img_w = 3, 84, 84

    lowdim_keys = ["observation.state", "observation.base_action"]
    if object_state_dim > 0:
        lowdim_keys.append("observation.object_state")

    lowdim_dim = env.observation_space["observation.state"].shape[1]

    action_dim = env.action_dim

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim

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
    agent.to(device)

    if args.checkpoint.lower() != "none":
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Check if structured (resume) or flat (best.pt) checkpoint
        if isinstance(ckpt, dict) and "actor" in ckpt and isinstance(ckpt["actor"], dict):
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            if "encoders" in ckpt:
                agent.encoders.load_state_dict(ckpt["encoders"])
            print(f"Loaded structured checkpoint: {args.checkpoint}", flush=True)
        else:
            # Flat state_dict (best.pt = agent.state_dict())
            agent.load_state_dict(ckpt, strict=False)
            print(f"Loaded flat state_dict: {args.checkpoint}", flush=True)
    else:
        print(f"Base policy eval (no residual)", flush=True)

    # Build noise injection function if requested
    obs_noise_fn = None
    if args.obj_pos_noise > 0 or args.obj_rot_noise > 0:
        pos_std = args.obj_pos_noise
        rot_std_rad = np.radians(args.obj_rot_noise) if args.obj_rot_noise > 0 else 0.0
        def _add_obj_noise(obs):
            import torch as th
            if "observation.object_state" in obs:
                obj = obs["observation.object_state"].clone()  # (N, 7): pos3 + quat_wxyz4
                N = obj.shape[0]
                # Position noise
                if pos_std > 0:
                    obj[:, :3] += th.randn(N, 3, device=obj.device) * pos_std
                # Rotation noise: random axis-angle -> delta quaternion
                if rot_std_rad > 0:
                    axes = th.randn(N, 3, device=obj.device)
                    axes = axes / (axes.norm(dim=1, keepdim=True) + 1e-8)
                    angles = th.randn(N, 1, device=obj.device) * rot_std_rad
                    half = angles / 2
                    dq_w = th.cos(half)
                    dq_xyz = axes * th.sin(half)
                    # Quaternion multiply: gt_quat (wxyz) * delta_quat (wxyz)
                    w1, x1, y1, z1 = obj[:, 3:4], obj[:, 4:5], obj[:, 5:6], obj[:, 6:7]
                    w2, x2, y2, z2 = dq_w, dq_xyz[:, 0:1], dq_xyz[:, 1:2], dq_xyz[:, 2:3]
                    obj[:, 3:4] = w1*w2 - x1*x2 - y1*y2 - z1*z2
                    obj[:, 4:5] = w1*x2 + x1*w2 + y1*z2 - z1*y2
                    obj[:, 5:6] = w1*y2 - x1*z2 + y1*w2 + z1*x2
                    obj[:, 6:7] = w1*z2 + x1*y2 - y1*x2 + z1*w2
                    # Normalize quaternion
                    obj[:, 3:7] = obj[:, 3:7] / (obj[:, 3:7].norm(dim=1, keepdim=True) + 1e-8)
                obs["observation.object_state"] = obj
            return obs
        obs_noise_fn = _add_obj_noise
        print(f"Object state noise: pos_std={pos_std:.3f}m rot_std={args.obj_rot_noise:.1f}deg", flush=True)

    print(f"Starting evaluation ({args.eval_num_episodes} episodes x {args.num_envs} envs)...", flush=True)
    t0 = time.time()
    try:
        metrics = evaluate(
            env=env,
            agent=agent,
            num_episodes=args.eval_num_episodes,
            device=device,
            image_keys=image_keys,
            obs_noise_fn=obs_noise_fn,
        )
    except Exception as e:
        print(f"ERROR during evaluate(): {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.time() - t0

    sr = metrics["eval/success_rate"]
    ret = metrics["eval/mean_return"]
    env_srs = [metrics.get(f"eval/sr_env{i}", 0) for i in range(args.num_envs)]

    print(f"\nRESULT: SR={sr:.1%}  return={ret:.1f}  time={elapsed:.1f}s", flush=True)
    env_strs = " ".join(f"e{i}={v:.0%}" for i, v in enumerate(env_srs))
    print(f"Per-env: {env_strs}", flush=True)

    result = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "seed": seed,
        "sr": float(sr),
        "return": float(ret),
        "env_srs": [float(v) for v in env_srs],
        "time": float(elapsed),
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved: {args.output_json}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()

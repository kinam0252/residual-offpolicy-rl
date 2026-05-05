"""
Eval with DA-V2 predicted depth instead of GT depth.

Loads a trained depth-mode checkpoint, runs Isaac Sim eval,
but replaces GT depth observations with DA-V2 predictions
(scale+shift aligned to GT range).

Usage:
  isaac_python eval_da2_depth_isaaclab.py --headless --enable_cameras \
    --checkpoint /path/to/agent_stepXXXXX.pt \
    --num_envs 20 --max_episode_steps 1000 ...
"""
from __future__ import annotations
import argparse, os, sys, time, json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval with DA-V2 predicted depth")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint .pt")
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.03)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--cube_perturb_table_path", type=str, default=None)
parser.add_argument("--action_scale", type=float, default=0.1)
parser.add_argument("--reward_type", type=str, default="dense_clipped")
parser.add_argument("--depth_norm_path", type=str, default=None)
parser.add_argument("--da2_encoder", type=str, default="vits", choices=["vits", "vitl"])
parser.add_argument("--da2_checkpoint", type=str, default=None, help="DA-V2 checkpoint path")
parser.add_argument("--align_mode", type=str, default="per_frame", choices=["per_frame", "fixed", "gt"],
                    help="per_frame: use GT for alignment each frame; fixed: use preset s,t; gt: skip DA-V2, use GT depth")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to average")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))

from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.config.rlpd import QAgentConfig, ActorConfig
from resfit.rl_finetuning.off_policy.common_utils import utils
import isaaclab.sim as sim_utils

_log = lambda msg: print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [da2-eval] {msg}", flush=True)


def main():
    device = torch.device(args.groot_policy_device)
    
    # ── 1. Create env ──
    _log("Creating environment...")
    _sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    env = IfaceEnvWrapper(
        sim=_sim,
        csv_dir=args.csv_base_dir,
        groot_model_path=args.groot_model_path,
        embodiment_tag=args.groot_embodiment_tag,
        policy_device=args.groot_policy_device,
        policy_strict=True,
        task_description="Pick up the white object.",
        language_override=args.language_override,
        max_episode_steps=args.max_episode_steps,
        num_envs=args.num_envs,
        success_threshold=args.success_threshold,
        cube_perturb_table_path=args.cube_perturb_table_path,
        reward_type=args.reward_type,
        use_depth=True,
        depth_norm_path=args.depth_norm_path,
    )
    N = args.num_envs
    _log(f"Env ready: {N} envs")

    # ── 2. Create agent ──
    _log("Creating agent...")
    image_keys = ["observation.depth.front", "observation.depth.wrist"]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    
    agent_cfg = QAgentConfig(
        actor_lr=1e-6, critic_lr=1e-4, critic_target_tau=0.005,
        actor=ActorConfig(action_scale=args.action_scale, actor_last_layer_init_scale=0.0, action_l2_reg_weight=1.0),
    )
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=agent_cfg,
        residual_actor=True,
    )
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    agent.load_checkpoint_compat(ckpt)
    agent.eval()
    agent.to(device)
    _log(f"Agent loaded from {args.checkpoint}")

    # ── 3. Load DA-V2 (skip if gt mode) ──
    da2_model = None
    if args.align_mode != "gt":
        _log(f"Loading DA-V2 ({args.da2_encoder})...")
        _da2_candidates = [
            str(Path(__file__).resolve().parents[3] / "Depth-Anything-V2"),
            "/home/nas_main/kinamkim/Repos/Intern/Depth-Anything-V2",
        ]
        da2_path = None
        for _p in _da2_candidates:
            if Path(_p).exists():
                da2_path = _p
                break
        if da2_path is None:
            raise FileNotFoundError(f"DA-V2 not found in: {_da2_candidates}")
        sys.path.insert(0, da2_path)
        _log(f"DA-V2 path: {da2_path}")
        from depth_anything_v2.dpt import DepthAnythingV2
    
        da2_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        da2_model = DepthAnythingV2(**da2_configs[args.da2_encoder])
        
        if args.da2_checkpoint:
            ckpt_path = args.da2_checkpoint
        else:
            ckpt_path = str(Path(da2_path) / f"checkpoints/depth_anything_v2_{args.da2_encoder}.pth")
        da2_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        da2_model = da2_model.to(device).eval()
        _log(f"DA-V2 loaded: {ckpt_path}")
    else:
        _log("GT depth mode: skipping DA-V2 loading")

    # ── 4. Depth normalization params ──
    norm_path = args.depth_norm_path or str(Path(__file__).resolve().parents[3] / 
                "residual-offpolicy-rl/depth_min_max/depth_normalization.json")
    depth_norm = json.loads(Path(norm_path).read_text())
    _log(f"Depth norm: front=[{depth_norm['front']['min']:.3f},{depth_norm['front']['max']:.3f}] "
         f"wrist=[{depth_norm['wrist']['min']:.3f},{depth_norm['wrist']['max']:.3f}]")

    # Fixed alignment params (from offline analysis)
    fixed_params = {
        'front': {'s': -0.153, 't': 1.35},
        'wrist': {'s': -0.025, 't': 0.21},
    }

    # ── 5. Run eval (multi-episode) ──
    all_rates = []
    all_returns = []
    da2_times = []
    
    for ep_i in range(args.num_episodes):
        _log(f"Running eval episode {ep_i+1}/{args.num_episodes}...")
        env.reset()
        obs = env._build_obs()
        
        ep_rewards = torch.zeros(N, device=device)
        ep_ever_success = torch.zeros(N, device=device)
    
        for step in range(args.max_episode_steps):
            # Replace GT depth with DA-V2 predictions (skip if gt mode)
            if args.align_mode != "gt":
              for cam_name, cam_key in [("front", "observation.depth.front"), ("wrist", "observation.depth.wrist")]:
                cam_obj = env.camera_front if cam_name == "front" else env.camera_wrist
                rgb_raw = cam_obj.data.output["rgb"]  # (N, H, W, 4) RGBA
                d_min = depth_norm[cam_name]['min']
                d_max = depth_norm[cam_name]['max']
                
                da2_depths = []
                t0 = time.time()
                for eid in range(N):
                    rgb_np = rgb_raw[eid, :, :, :3].detach().cpu().numpy().astype(np.uint8)
                    rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                    with torch.no_grad():
                        pred_rel = da2_model.infer_image(rgb_bgr)
                    pred_84 = cv2.resize(pred_rel, (84, 84), interpolation=cv2.INTER_LINEAR)
                    
                    if args.align_mode == "per_frame":
                        gt_84 = obs[cam_key][eid, 0].detach().cpu().numpy()
                        gt_metric = gt_84 * (d_max - d_min) + d_min
                        valid = gt_metric > 0.01
                        if valid.sum() > 10:
                            p = pred_84[valid].flatten().astype(np.float64)
                            g = gt_metric[valid].flatten().astype(np.float64)
                            A = np.stack([p, np.ones_like(p)], axis=1)
                            s, t = np.linalg.lstsq(A, g, rcond=None)[0]
                        else:
                            s, t = fixed_params[cam_name]['s'], fixed_params[cam_name]['t']
                    else:
                        s = fixed_params[cam_name]['s']
                        t = fixed_params[cam_name]['t']
                    
                    pred_metric = np.clip(s * pred_84 + t, 0, 20).astype(np.float32)
                    pred_normed = np.clip(pred_metric, d_min, d_max)
                    pred_normed = (pred_normed - d_min) / (d_max - d_min)
                    pred_normed = np.nan_to_num(pred_normed, nan=0.0)
                    da2_depths.append(torch.tensor(pred_normed, dtype=torch.float32, device=device).unsqueeze(0))
                
                da2_times.append(time.time() - t0)
                obs[cam_key] = torch.stack(da2_depths, dim=0)  # (N, 1, 84, 84)
            
            # Debug log + sanity check images
            if step == 0 and ep_i == 0:
                for ck in ["observation.depth.front", "observation.depth.wrist"]:
                    v = obs[ck]
                    _log(f"  {ck}: shape={v.shape} range=[{v.min().item():.4f},{v.max().item():.4f}]")
            
            # Save depth comparison images (GT vs DA-V2) at a few steps
            _save_steps = {0, 100, 300, 500, 800}
            if args.output_dir and step in _save_steps and args.align_mode != "gt":
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    _sanity_dir = Path(args.output_dir) / "sanity_images"
                    _sanity_dir.mkdir(parents=True, exist_ok=True)
                    # Get GT depth from env (before DA-V2 replacement in next_obs)
                    for ck in ["observation.depth.front", "observation.depth.wrist"]:
                        gt_v = next_obs[ck][0, 0].detach().cpu().numpy()  # GT from env step
                        da_v = obs[ck][0, 0].detach().cpu().numpy()  # DA-V2 (current obs)
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                        ax1.imshow(gt_v, cmap='turbo', vmin=0, vmax=1); ax1.set_title('GT Depth'); ax1.axis('off')
                        ax2.imshow(da_v, cmap='turbo', vmin=0, vmax=1); ax2.set_title('DA-V2 Depth'); ax2.axis('off')
                        diff = np.abs(gt_v - da_v)
                        ax3.imshow(diff, cmap='turbo', vmin=0, vmax=0.3); ax3.set_title(f'|Error| RMSE={np.sqrt(np.mean(diff**2)):.3f}'); ax3.axis('off')
                        cam = ck.split('.')[-1]
                        fig.suptitle(f'Ep{ep_i} Step{step} {cam}', fontsize=12)
                        plt.tight_layout()
                        plt.savefig(str(_sanity_dir / f'ep{ep_i}_step{step}_{cam}.png'), dpi=100)
                        plt.close()
                except Exception as e:
                    _log(f"  Sanity image save error: {e}")
            
            # Agent act
            with torch.no_grad(), utils.eval_mode(agent):
                residual_action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
            
            # Step env
            next_obs, reward, terminated, truncated, info = env.step(residual_action)
            ep_rewards += reward[:N].to(device)
            
            # Success check
            cube_z = env.cube.data.root_state_w[:N, 2].to(device)
            cube_init_z = env._initial_cube_z[:N].to(device)
            cube_lifted = ((cube_z - cube_init_z) >= args.success_threshold).float()
            _contact = getattr(env, '_contact_history', None)
            if _contact is not None:
                _gate = _contact.any(dim=1).float().to(device)
                cube_lifted = cube_lifted * _gate
            ep_ever_success = torch.max(ep_ever_success, cube_lifted)
            
            obs = next_obs
            done = terminated | truncated
            if done[:N].all():
                break
        
        # Episode summary (outside step loop, inside episode loop)
        success_rate = ep_ever_success.mean().item()
        mean_return = ep_rewards.mean().item()
        all_rates.append(success_rate)
        all_returns.append(mean_return)
        _log(f"  Episode {ep_i+1}: success={success_rate*100:.1f}% return={mean_return:.1f}")
    
    avg_rate = np.mean(all_rates)
    avg_return = np.mean(all_returns)
    avg_da2_ms = np.mean(da2_times) * 1000 / N if da2_times else 0
    
    _log(f"\n{'='*60}")
    _log(f"RESULT: success_rate={avg_rate:.3f} ({avg_rate*100:.1f}%)")
    _log(f"  mean_return={avg_return:.1f}")
    _log(f"  per_episode_rates={[f'{r*100:.0f}%' for r in all_rates]}")
    _log(f"  align_mode={args.align_mode}")
    _log(f"  da2_encoder={args.da2_encoder}")
    _log(f"  checkpoint={args.checkpoint}")
    _log(f"  num_episodes={args.num_episodes}")
    _log(f"  avg DA-V2 time per env: {avg_da2_ms:.1f}ms")
    _log(f"{'='*60}")
    
    # Save result
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = {
            "success_rate": avg_rate,
            "mean_return": avg_return,
            "all_rates": all_rates,
            "all_returns": all_returns,
            "align_mode": args.align_mode,
            "da2_encoder": args.da2_encoder,
            "checkpoint": str(args.checkpoint),
            "num_envs": N,
            "num_episodes": args.num_episodes,
            "max_episode_steps": args.max_episode_steps,
        }
        (out / "da2_eval_result.json").write_text(json.dumps(result, indent=2))
        _log(f"Result saved to {out / 'da2_eval_result.json'}")
    
    simulation_app.close()


if __name__ == "__main__":
    main()

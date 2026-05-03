"""Render base policy videos - one position at a time, save immediately."""
import sys, os, argparse, numpy as np, torch, time
torch.backends.cuda.enable_cudnn_sdp(False)

def main():
    os.chdir('/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl')
    sys.path.insert(0, '.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", type=int, required=True)
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--out_dir", type=str, default="outputs/cup_base_videos")
    args = parser.parse_args()

    from resfit.rl_finetuning.wrappers.mujoco_vec_env_cup import MuJoCoVecEnvCup
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_cup import MuJoCoResidualWrapperCup
    import imageio
    import json

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load positions and split
    with open('configs/cup_positions.json') as f:
        positions = json.load(f)
    n_pos = len(positions)
    
    # Split positions across chunks
    chunk_size = (n_pos + args.num_chunks - 1) // args.num_chunks
    start = args.chunk_id * chunk_size
    end = min(start + chunk_size, n_pos)
    my_positions = list(range(start, end))
    print(f"Chunk {args.chunk_id}: positions {my_positions} ({len(my_positions)} total)")

    # Create 1 env at a time for cleaner video
    for pos_idx in my_positions:
        print(f"\n--- Position {pos_idx} ---")
        t0 = time.time()
        
        env = MuJoCoVecEnvCup(
            num_envs=1,
            episode_ids=[pos_idx],
            max_episode_steps=args.max_steps,
            reward_type='dense',
            device='cuda:0',
            cup_positions_path='configs/cup_positions.json',
            parallel_envs=False,
        )
        wrapper = MuJoCoResidualWrapperCup(
            vec_env=env,
            groot_checkpoint=os.path.expanduser('~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-200000'),
            embodiment_tag='NEW_EMBODIMENT',
            policy_device='cuda:0',
            task_description="Pick up the cup lying on its side and stand it upright",
            ema_alpha=0.3,
        )
        
        obs, info = wrapper.reset()
        frames = []
        total_reward = 0.0
        
        for step in range(args.max_steps):
            residual = torch.zeros((1, wrapper.action_dim), device='cuda:0')
            obs, reward, terminated, truncated, info = wrapper.step(residual)
            total_reward += reward.item()
            
            # Render every 2 steps
            if step % 2 == 0:
                e = env._envs[0]
                e['renderer_base'].update_scene(e['data'], camera=e['cam_base_id'])
                img = e['renderer_base'].render().copy()
                frames.append(img)
            
            if step % 100 == 0:
                gs = env._envs[0]['grasp_state']
                fid = env._envs[0]['ids']['finger_ids'][0]
                grip = float(env._envs[0]['data'].qpos[env._envs[0]['model'].jnt_qposadr[fid]])
                print(f"  step={step:3d} R={total_reward:.4f} grasped={gs['grasped']} grip={grip:.4f}")
        
        # Final state
        gs = env._envs[0]['grasp_state']
        # Direct uprightness calc: 1 - 2(x² + y²) from cup quaternion
        cup_qpos = env._envs[0]['data'].qpos[env._envs[0]['ids']['cup_jnt_qposadr']:]
        if len(cup_qpos) >= 7:
            qw, qx, qy, qz = cup_qpos[3], cup_qpos[4], cup_qpos[5], cup_qpos[6]
            uprightness = 1.0 - 2.0*(qx*qx + qy*qy)
        else:
            uprightness = 0.0
        elapsed = time.time() - t0
        print(f"  DONE: R={total_reward:.4f} grasped={gs['grasped']} upright={uprightness:.3f} ({elapsed:.1f}s)")
        
        # Save video immediately
        video_path = os.path.join(args.out_dir, f"pos{pos_idx:02d}_base.mp4")
        imageio.mimwrite(video_path, frames, fps=15)
        print(f"  SAVED: {video_path} ({len(frames)} frames)")
        
        wrapper.close()
    
    print(f"\n=== Chunk {args.chunk_id} complete ===")

if __name__ == "__main__":
    main()

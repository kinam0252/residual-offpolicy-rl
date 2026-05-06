import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    import torch
    torch.backends.cuda.enable_cudnn_sdp(False)  # Like training script
    import numpy as np
    
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_unified import MuJoCoResidualWrapperUnified

    device = torch.device("cuda:0")
    N = 30

    # Create env + wrapper (like training)
    env = MuJoCoVecEnvPnP(num_envs=N, max_episode_steps=50, parallel_envs=True, num_workers=8)
    groot_ckpt = os.path.expanduser("~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000")
    wrapper = MuJoCoResidualWrapperUnified(
        vec_env=env, groot_checkpoint=groot_ckpt, embodiment_tag="franka",
        policy_device="cuda:0", task_description="Pick up the red cube and place it onto the plate.",
        chunk_sync=True, render_parallel=False,
    )
    
    # Check GPU memory
    alloc = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"GPU mem after env+GR00T: alloc={alloc:.2f}GB, reserved={reserved:.2f}GB", flush=True)
    
    # Create realistic critic/actor (like training)
    from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
    cfg = ResidualTD3MuJoCoConfig()
    cfg.actor_hidden_dim = 256
    cfg.critic_hidden_dim = 256
    agent = QAgent(obs_dim=27, action_dim=7, image_shape=None, device=device, cfg=cfg)
    alloc2 = torch.cuda.memory_allocated(0) / 1e9
    print(f"GPU mem after QAgent: alloc={alloc2:.2f}GB (+{alloc2-alloc:.2f}GB)", flush=True)
    
    # Allocate ~4GB tensor to simulate replay buffer on GPU
    dummy_tensors = [torch.randn(500, 1024, device=device) for _ in range(8)]
    alloc3 = torch.cuda.memory_allocated(0) / 1e9
    print(f"GPU mem with buffers: alloc={alloc3:.2f}GB", flush=True)

    # Phase A: Reset + warmup
    obs, _ = wrapper.reset()
    print("A. first reset OK", flush=True)
    for step in range(34):
        action = torch.zeros(N, 7)
        obs, reward, terminated, truncated, info = wrapper.step(action)
    print("B. warmup 34 steps OK", flush=True)

    # Phase B: Real critic warmup (2000 iters with QAgent)
    from tensordict import TensorDict
    t0 = time.time()
    for i in range(500):  # reduced from 2000 for speed
        batch = TensorDict({
            "observation": torch.randn(256, 27, device=device),
            "action": torch.randn(256, 7, device=device),
            "reward": torch.randn(256, 1, device=device),
            "next_observation": torch.randn(256, 27, device=device),
            "done": torch.zeros(256, 1, device=device),
        })
        metrics = agent.update(batch, stddev=0.0, update_actor=False, bc_batch=None, ref_agent=agent)
        if i % 100 == 0:
            print(f"  critic warmup {i}/500 loss={metrics.get('train/critic_loss', 0):.4f}", flush=True)
    print(f"C. critic warmup OK ({time.time()-t0:.1f}s)", flush=True)

    alloc4 = torch.cuda.memory_allocated(0) / 1e9
    print(f"GPU mem after critic warmup: alloc={alloc4:.2f}GB", flush=True)

    # Phase C: Second reset (WHERE TRAINING CRASHES)
    print("D. attempting second reset...", flush=True)
    try:
        obs, _ = wrapper.reset()
        print("E. SECOND RESET OK!", flush=True)
    except Exception as e:
        print(f"E. SECOND RESET FAILED: {e}", flush=True)
        # Try to diagnose
        print(f"   GPU mem: alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)
        wrapper.close()
        return

    for step in range(10):
        action = torch.zeros(N, 7)
        obs, reward, terminated, truncated, info = wrapper.step(action)
    print("F. post-reset steps OK", flush=True)

    # Cleanup
    del dummy_tensors
    wrapper.close()
    print("ALL PASS", flush=True)

if __name__ == "__main__":
    main()

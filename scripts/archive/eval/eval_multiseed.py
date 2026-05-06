"""
Multi-seed evaluation of a residual RL checkpoint vs base policy.
Runs 15-env batch eval N times with different seeds and reports average SR.

Usage:
    # Evaluate best checkpoint (3 seeds)
    python scripts/eval_multiseed.py --checkpoint outputs/.../checkpoints/best.pt --seeds 0,1,2

    # Evaluate base policy only (zero residual, 3 seeds)
    python scripts/eval_multiseed.py --base-only --seeds 0,1,2
"""
import sys, os, json, time, importlib, types as _types, argparse
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DS_BUILD_OPS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Deepspeed mock
if "deepspeed" not in sys.modules:
    m = _types.ModuleType("deepspeed"); m.__version__ = "0.0.0"
    m.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None); m.__path__ = []; m.__file__ = "fake"
    z = _types.ModuleType("deepspeed.zero"); z.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    z.Init = lambda *a, **kw: (lambda f: f); m.zero = z
    sys.modules["deepspeed"] = m; sys.modules["deepspeed.zero"] = z

import numpy as np
import torch

SCENE_XML = os.path.expanduser(
    "~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
)
MAX_STEPS = 500


def load_residual_agent(checkpoint_path, obs_sample, action_dim, device):
    """Load trained residual actor from best.pt checkpoint."""
    from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_args = ckpt.get("args", {})

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.hidden_dim = saved_args.get("actor_hidden_dim", 256)
    cfg.agent.critic.hidden_dim = saved_args.get("critic_hidden_dim", 256)
    cfg.agent.asymmetric_critic = saved_args.get("asymmetric_critic", True)

    # Determine dimensions from obs
    # Training uses only observation.state for prop_shape (line 649 of train script)
    state_dim = obs_sample["observation.state"].shape[-1]

    object_state_dim = 10 if saved_args.get("asymmetric_critic", True) else 0
    prop_dim = state_dim
    _asymmetric = saved_args.get("asymmetric_critic", True)

    if _asymmetric:
        rl_cameras = ["observation.depth.front", "observation.depth.wrist"]
    else:
        rl_cameras = []

    agent = QAgent(
        obs_shape=(1, 84, 84),
        prop_shape=(prop_dim,),
        action_dim=action_dim,
        rl_cameras=rl_cameras,
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=_asymmetric,
    )

    agent.load_state_dict(ckpt["model"])
    agent.to(device)
    agent.eval()
    return agent


def eval_single_seed(seed, positions, groot_ckpt, residual_ckpt=None, device="cuda"):
    """Run one eval pass with given seed. Returns per-env success list."""
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP
    from scipy.spatial.transform import Rotation
    import mujoco

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_envs = len(positions)
    cube_positions = [p["cube_pos"] for p in positions]
    bowl_positions = [p["bowl_pos"] for p in positions]
    cube_yaws = [p.get("cube_yaw_deg", 0.0) for p in positions]

    vec_env = MuJoCoVecEnvPnP(
        num_envs=num_envs,
        cube_positions=cube_positions,
        bowl_positions=bowl_positions,
        max_episode_steps=MAX_STEPS,
        reward_type="dense",
        scene_xml=SCENE_XML,
    )

    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=groot_ckpt,
        task_description="Pick up the red cube and place it onto the plate.",
        ema_alpha=0.0,
    )

    # Load residual actor if provided
    agent = None
    if residual_ckpt:
        obs_tmp, _ = wrapper.reset()
        agent = load_residual_agent(residual_ckpt, obs_tmp, wrapper.action_dim, device)

    obs, _ = wrapper.reset()

    # Override positions
    for ei in range(num_envs):
        env_data = vec_env._envs[ei]
        data, model = env_data["data"], env_data["model"]
        cqa = env_data["cube_qposadr"]
        if cqa is not None:
            data.qpos[cqa:cqa+3] = cube_positions[ei]
            q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
            data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        bowl_body_id = env_data["bowl_body_id"]
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_positions[ei]
        env_data["cube_pos_init"] = np.array(cube_positions[ei])
        env_data["bowl_pos_init"] = np.array(bowl_positions[ei])
        mujoco.mj_forward(model, data)

    vec_env._step_counts = np.zeros(num_envs, dtype=int)
    vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

    # Re-reset to get clean obs after position override
    obs, _ = wrapper.reset()
    for ei in range(num_envs):
        env_data = vec_env._envs[ei]
        data, model = env_data["data"], env_data["model"]
        cqa = env_data["cube_qposadr"]
        if cqa is not None:
            data.qpos[cqa:cqa+3] = cube_positions[ei]
            q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
            data.qpos[cqa+3:cqa+7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        bowl_body_id = env_data["bowl_body_id"]
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_positions[ei]
        env_data["cube_pos_init"] = np.array(cube_positions[ei])
        env_data["bowl_pos_init"] = np.array(bowl_positions[ei])
        mujoco.mj_forward(model, data)

    vec_env._step_counts = np.zeros(num_envs, dtype=int)
    vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

    # Step loop
    env_done = [False] * num_envs
    env_success = [False] * num_envs

    for step in range(MAX_STEPS):
        if agent is not None:
            obs_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            with torch.no_grad():
                action = agent.act(obs_gpu, eval_mode=True).cpu().numpy()
        else:
            action = np.zeros((num_envs, wrapper.action_dim))

        obs, rew, term, trunc, info = wrapper.step(action)
        done = term | trunc if hasattr(term, '__or__') else np.array(term) | np.array(trunc)

        for ei in range(num_envs):
            if env_done[ei]:
                continue
            cqa = vec_env._envs[ei]["cube_qposadr"]
            if cqa is not None:
                cp = vec_env._envs[ei]["data"].qpos[cqa:cqa+3].copy()
                d = np.linalg.norm(cp[:2] - np.array(bowl_positions[ei][:2]))
                if d < 0.095 and cp[2] < 0.08:
                    env_success[ei] = True

            d_val = done[ei] if hasattr(done, '__getitem__') else done
            if d_val:
                env_done[ei] = True
                t_val = term[ei] if hasattr(term, '__getitem__') else term
                if t_val:
                    env_success[ei] = True

        if all(env_done):
            break

    # Cleanup
    del wrapper, vec_env
    if agent is not None:
        del agent
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return env_success


def main():
    parser = argparse.ArgumentParser(description="Multi-seed eval")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best.pt residual checkpoint")
    parser.add_argument("--base-only", action="store_true",
                        help="Evaluate base policy only (zero residual)")
    parser.add_argument("--positions-file", type=str, default="configs/pnp_eval_hard.json",
                        help="Eval positions JSON")
    parser.add_argument("--groot-checkpoint", type=str,
                        default=os.path.expanduser("~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"))
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated seeds")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if not args.base_only and args.checkpoint is None:
        parser.error("Provide --checkpoint or --base-only")

    seeds = [int(s) for s in args.seeds.split(",")]
    positions = json.loads(Path(args.positions_file).read_text())

    mode = "base" if args.base_only else "residual"
    if args.output_dir is None:
        if args.base_only:
            args.output_dir = "outputs/eval_multiseed/base_policy"
        else:
            ckpt_name = Path(args.checkpoint).parent.parent.name
            args.output_dir = f"outputs/eval_multiseed/{ckpt_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Mode: {mode}")
    print(f"Seeds: {seeds}")
    print(f"Positions: {args.positions_file} ({len(positions)} envs)")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(flush=True)

    all_results = []
    for seed in seeds:
        t0 = time.time()
        print(f"\n{'='*50}")
        print(f"Seed {seed}...")
        env_success = eval_single_seed(
            seed=seed,
            positions=positions,
            groot_ckpt=args.groot_checkpoint,
            residual_ckpt=args.checkpoint if not args.base_only else None,
        )
        sr = sum(env_success) / len(env_success)
        elapsed = time.time() - t0
        print(f"  Seed {seed}: SR={sr*100:.1f}% ({sum(env_success)}/{len(env_success)}) [{elapsed:.0f}s]")

        per_env = {f"env{i}": int(env_success[i]) for i in range(len(env_success))}
        all_results.append({
            "seed": seed,
            "success_rate": sr,
            "n_success": sum(env_success),
            "n_total": len(env_success),
            "per_env": per_env,
            "time_sec": elapsed,
        })

    # Summary
    srs = [r["success_rate"] for r in all_results]
    mean_sr = np.mean(srs)
    std_sr = np.std(srs)

    print(f"\n{'='*50}")
    print(f"SUMMARY ({mode})")
    print(f"  Seeds: {seeds}")
    for r in all_results:
        print(f"  Seed {r['seed']}: SR={r['success_rate']*100:.1f}%")
    print(f"  Mean SR: {mean_sr*100:.1f}% ± {std_sr*100:.1f}%")
    print(f"{'='*50}")

    # Per-env average
    num_envs = len(positions)
    print("\nPer-env SR (averaged over seeds):")
    for ei in range(num_envs):
        env_sr = np.mean([r["per_env"][f"env{ei}"] for r in all_results])
        print(f"  env{ei:2d}: {env_sr*100:.0f}%")

    # Save results
    result = {
        "mode": mode,
        "checkpoint": args.checkpoint,
        "positions_file": args.positions_file,
        "seeds": seeds,
        "per_seed": all_results,
        "mean_sr": mean_sr,
        "std_sr": std_sr,
    }
    out_path = Path(args.output_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

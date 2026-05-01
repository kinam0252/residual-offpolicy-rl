"""
Evaluate base policy SR under random cube+bowl perturbation.
Generates N random environments per difficulty level and measures base SR.

Usage:
    python scripts/eval_base_perturb.py --n-envs 50 --seeds 0,1,2
    python scripts/eval_base_perturb.py --n-envs 100 --seeds 0 --levels easy,medium,hard
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

CUBE_BASE = np.array([0.42, -0.03, 0.02])
BOWL_BASE = np.array([0.42, 0.03, 0.0])

# Difficulty levels: perturbation ranges in meters and degrees
LEVELS = {
    "zero":     {"cube_dx": 0.00, "cube_dy": 0.00, "bowl_dx": 0.00, "bowl_dy": 0.00, "yaw": 0},
    "easy":     {"cube_dx": 0.02, "cube_dy": 0.02, "bowl_dx": 0.02, "bowl_dy": 0.02, "yaw": 5},
    "medium":   {"cube_dx": 0.05, "cube_dy": 0.05, "bowl_dx": 0.05, "bowl_dy": 0.05, "yaw": 15},
    "hard":     {"cube_dx": 0.08, "cube_dy": 0.08, "bowl_dx": 0.08, "bowl_dy": 0.08, "yaw": 30},
    "extreme":  {"cube_dx": 0.12, "cube_dy": 0.10, "bowl_dx": 0.10, "bowl_dy": 0.10, "yaw": 50},
}


def generate_positions(n_envs, level_name, rng):
    """Generate n random positions for a given difficulty level."""
    lvl = LEVELS[level_name]
    positions = []
    for i in range(n_envs):
        cube_pos = CUBE_BASE.copy()
        cube_pos[0] += rng.uniform(-lvl["cube_dx"], lvl["cube_dx"])
        cube_pos[1] += rng.uniform(-lvl["cube_dy"], lvl["cube_dy"])

        bowl_pos = BOWL_BASE.copy()
        bowl_pos[0] += rng.uniform(-lvl["bowl_dx"], lvl["bowl_dx"])
        bowl_pos[1] += rng.uniform(-lvl["bowl_dy"], lvl["bowl_dy"])

        yaw = rng.uniform(-lvl["yaw"], lvl["yaw"])

        dist = np.linalg.norm(cube_pos[:2] - bowl_pos[:2])
        positions.append({
            "episode": i,
            "cube_pos": cube_pos.tolist(),
            "bowl_pos": bowl_pos.tolist(),
            "cube_yaw_deg": float(yaw),
            "distance": float(dist),
        })
    return positions


def eval_batch(positions, groot_ckpt, device="cuda"):
    """Evaluate base policy on a batch of positions. Returns per-env success list."""
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP
    from scipy.spatial.transform import Rotation
    import mujoco

    num_envs = len(positions)
    cube_positions = [p["cube_pos"] for p in positions]
    bowl_positions = [p["bowl_pos"] for p in positions]
    cube_yaws = [p["cube_yaw_deg"] for p in positions]

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

    # Re-reset + re-override
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

    del wrapper, vec_env
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return env_success


def main():
    parser = argparse.ArgumentParser(description="Base policy eval with random perturbation")
    parser.add_argument("--n-envs", type=int, default=50,
                        help="Number of random envs per level per seed")
    parser.add_argument("--batch-size", type=int, default=15,
                        help="Max envs per batch (GPU memory)")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated seeds")
    parser.add_argument("--levels", type=str, default="zero,easy,medium,hard,extreme",
                        help="Comma-separated difficulty levels")
    parser.add_argument("--groot-checkpoint", type=str,
                        default=os.path.expanduser("~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"))
    parser.add_argument("--output-dir", type=str, default="outputs/eval_base_perturb")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    levels = [l.strip() for l in args.levels.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Seeds: {seeds}")
    print(f"Levels: {levels}")
    print(f"Envs per level per seed: {args.n_envs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print(flush=True)

    all_results = {}

    for level in levels:
        level_results = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            positions = generate_positions(args.n_envs, level, rng)

            t0 = time.time()
            print(f"\n{'='*50}")
            print(f"Level={level}, Seed={seed}, N={args.n_envs}")

            # Run in batches
            all_success = []
            for batch_start in range(0, len(positions), args.batch_size):
                batch_pos = positions[batch_start:batch_start + args.batch_size]
                batch_success = eval_batch(batch_pos, args.groot_checkpoint)
                all_success.extend(batch_success)
                sr_so_far = sum(all_success) / len(all_success)
                print(f"  batch {batch_start//args.batch_size + 1}: "
                      f"{sum(batch_success)}/{len(batch_success)} "
                      f"(running SR={sr_so_far*100:.1f}%)")

            sr = sum(all_success) / len(all_success)
            elapsed = time.time() - t0
            print(f"  → Level={level} Seed={seed}: SR={sr*100:.1f}% "
                  f"({sum(all_success)}/{len(all_success)}) [{elapsed:.0f}s]")

            # Distance stats for this batch
            dists = [p["distance"] for p in positions]
            yaws = [abs(p["cube_yaw_deg"]) for p in positions]

            level_results.append({
                "seed": seed,
                "level": level,
                "success_rate": sr,
                "n_success": sum(all_success),
                "n_total": len(all_success),
                "per_env": [int(s) for s in all_success],
                "positions": positions,
                "dist_mean": float(np.mean(dists)),
                "dist_std": float(np.std(dists)),
                "yaw_mean": float(np.mean(yaws)),
                "time_sec": elapsed,
            })

        # Aggregate across seeds
        srs = [r["success_rate"] for r in level_results]
        mean_sr = np.mean(srs)
        std_sr = np.std(srs) if len(srs) > 1 else 0.0
        dist_mean = np.mean([r["dist_mean"] for r in level_results])

        all_results[level] = {
            "mean_sr": float(mean_sr),
            "std_sr": float(std_sr),
            "dist_mean": float(dist_mean),
            "per_seed": level_results,
        }

        print(f"\n>>> Level={level}: SR={mean_sr*100:.1f}% ±{std_sr*100:.1f}% "
              f"(avg_dist={dist_mean:.3f}m)")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Level':<10} {'SR':>8} {'±std':>8} {'avg_dist':>10} {'avg_yaw':>10}")
    print(f"{'-'*60}")
    for level in levels:
        r = all_results[level]
        yaw_mean = np.mean([s["yaw_mean"] for s in r["per_seed"]])
        print(f"{level:<10} {r['mean_sr']*100:>7.1f}% {r['std_sr']*100:>7.1f}% "
              f"{r['dist_mean']:>10.3f} {yaw_mean:>9.1f}°")

    # Save
    out_file = os.path.join(args.output_dir, "results.json")
    with open(out_file, "w") as f:
        # Strip positions for compact JSON (save separately)
        compact = {}
        for level, data in all_results.items():
            compact[level] = {
                "mean_sr": data["mean_sr"],
                "std_sr": data["std_sr"],
                "dist_mean": data["dist_mean"],
                "per_seed": [{k: v for k, v in s.items() if k != "positions"}
                             for s in data["per_seed"]],
            }
        json.dump(compact, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Also save full positions for reproducibility
    pos_file = os.path.join(args.output_dir, "positions.json")
    pos_data = {}
    for level, data in all_results.items():
        pos_data[level] = [s["positions"] for s in data["per_seed"]]
    with open(pos_file, "w") as f:
        json.dump(pos_data, f)
    print(f"Positions saved to {pos_file}")


if __name__ == "__main__":
    main()

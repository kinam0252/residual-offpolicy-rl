"""Evaluate GR00T base policy on PnP: N episodes per difficulty, save videos + JSON results."""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import imageio

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

# Difficulty definitions matching training configs
DIFFICULTY_CONFIGS = {
    "easy": None,  # Use episode positions from JSON as-is
    "normal": {"dx": (-0.06, 0.06), "dy": (-0.06, 0.06), "yaw": (-30, 30)},
    "hard": {"dx": (-0.10, 0.10), "dy": (-0.10, 0.10), "yaw": (-60, 60)},
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PnP base policy SR")
    p.add_argument("--groot_checkpoint", required=True, help="Path to GR00T checkpoint")
    p.add_argument("--difficulty", required=True, choices=["easy", "normal", "hard"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True, help="Output directory for results + videos")
    p.add_argument("--max_episode_steps", type=int, default=300)
    p.add_argument("--scene_xml", default=None)
    p.add_argument("--task_description", default="Pick up the red cube and place it onto the plate.")
    p.add_argument("--positions_file", default="configs/pnp_sim33ep_positions.json")
    p.add_argument("--no_video", action="store_true", default=False)
    p.add_argument("--ema_alpha", type=float, default=0.0, help="EMA smoothing alpha (0=disabled)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    save_video = not args.no_video

    np.random.seed(args.seed)

    # Load episode positions
    with open(args.positions_file) as f:
        all_positions = json.load(f)

    difficulty_config = DIFFICULTY_CONFIGS[args.difficulty]

    num_eps = min(args.episodes, len(all_positions))

    # Pre-generate seeds for reproducible noise per episode
    episode_seeds = [args.seed + i for i in range(num_eps)]

    print(f"[Eval] Checkpoint: {args.groot_checkpoint}")
    print(f"[Eval] Difficulty: {args.difficulty}")
    print(f"[Eval] Episodes: {num_eps}")
    print(f"[Eval] Output: {args.out_dir}")
    print()

    # Resolve scene XML
    scene_xml = args.scene_xml
    if scene_xml is None:
        default_xml = os.path.join(
            os.path.dirname(__file__), "..",
            "..", "mujoco_menagerie", "franka_fr3", "fr3_with_hand.xml"
        )
        if os.path.exists(default_xml):
            scene_xml = default_xml
        else:
            scene_xml = "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"

    # Create env + wrapper once, reuse for all episodes
    first_ep = all_positions[0]
    vec_env = MuJoCoVecEnvPnP(
        num_envs=1,
        cube_positions=[first_ep["cube_pos"]],
        bowl_positions=[first_ep["bowl_pos"]],
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
        scene_xml=scene_xml,
    )
    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=args.groot_checkpoint,
        task_description=args.task_description,
        ema_alpha=args.ema_alpha,
    )

    # Real teleop initial joint position (idx=2, sorted by joint name)
    REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])
    from scipy.spatial.transform import Rotation
    import mujoco

    results = []
    successes = []

    for i in range(num_eps):
        ep = all_positions[i]
        base_cube_pos = np.array(ep["cube_pos"], dtype=np.float64)
        bowl_pos = np.array(ep["bowl_pos"], dtype=np.float64)

        # Apply difficulty noise
        actual_cube_pos = base_cube_pos.copy()
        yaw_deg = ep.get("cube_yaw_deg", 0.0)
        if difficulty_config is not None:
            rng = np.random.RandomState(episode_seeds[i])
            dx = rng.uniform(difficulty_config["dx"][0], difficulty_config["dx"][1])
            dy = rng.uniform(difficulty_config["dy"][0], difficulty_config["dy"][1])
            actual_cube_pos[0] += dx
            actual_cube_pos[1] += dy
            yaw_lo, yaw_hi = difficulty_config.get("yaw", (0, 0))
            yaw_deg = rng.uniform(yaw_lo, yaw_hi)

        # Reset wrapper first (this also resets vec_env internally)
        obs = wrapper.reset()

        # Now override positions AFTER reset (reset would overwrite them)
        env_data = vec_env._envs[0]
        data = env_data["data"]
        model = env_data["model"]
        cqa = env_data["cube_qposadr"]
        bowl_body_id = env_data["bowl_body_id"]

        # Set robot joints to real teleop initial pose
        for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
            data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
        for fid in env_data["ids"]["finger_ids"]:
            if fid >= 0:
                data.qpos[model.jnt_qposadr[fid]] = 0.04

        # Set cube position + yaw
        if cqa is not None:
            data.qpos[cqa:cqa + 3] = actual_cube_pos
            q_xyzw = Rotation.from_euler("z", np.radians(yaw_deg)).as_quat()
            data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # Set bowl position
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_pos

        env_data["cube_pos_init"] = actual_cube_pos.copy()
        env_data["bowl_pos_init"] = bowl_pos.copy()
        env_data["grasp_active"] = False
        env_data["grasp_offset"] = None
        env_data["grasp_quat_offset"] = None

        mujoco.mj_forward(model, data)
        vec_env._step_counts = np.zeros(vec_env.num_envs, dtype=int)
        vec_env._episode_rewards = np.zeros(vec_env.num_envs, dtype=np.float64)

        frames_base = []
        frames_wrist = []
        max_reward = 0.0
        ever_success = False
        min_dist = 999.0
        rewards_log = []

        for step in range(args.max_episode_steps):
            action = np.zeros((1, wrapper.action_dim))
            obs, rew, term, trunc, info = wrapper.step(action)
            r = float(rew[0]) if hasattr(rew, '__getitem__') else float(rew)
            max_reward = max(max_reward, r)
            rewards_log.append(r)

            # Save video frames (every 3rd step)
            if save_video and step % 3 == 0:
                fb = vec_env.get_frame(0, camera='back', size=(360, 640))
                fw = vec_env.get_frame(0, camera='wrist', size=(360, 640))
                frames_base.append(fb)
                frames_wrist.append(fw)

            # Check success: cube within bowl radius and near table
            if cqa is not None:
                cp = data.qpos[cqa:cqa + 3].copy()
                d = np.linalg.norm(cp[:2] - bowl_pos[:2])
                min_dist = min(min_dist, d)
                if d < 0.095 and cp[2] < 0.08:
                    ever_success = True

            done = bool(term) if np.isscalar(term) else bool(term[0])
            truncated = bool(trunc) if np.isscalar(trunc) else bool(trunc[0])
            if done or truncated:
                break

        tag = "OK" if ever_success else "FAIL"
        elapsed_r = rewards_log[-1] if rewards_log else 0.0
        print(f"  ep{i:02d}: min_dist={min_dist:.3f}m, max_r={max_reward:.2f}, "
              f"final_r={elapsed_r:.2f}, steps={step+1}, {tag}")

        successes.append(ever_success)

        # Save videos
        if save_video and frames_base:
            imageio.mimsave(
                os.path.join(args.out_dir, f"ep{i:02d}_base.mp4"),
                frames_base, fps=15, macro_block_size=1
            )
            imageio.mimsave(
                os.path.join(args.out_dir, f"ep{i:02d}_wrist.mp4"),
                frames_wrist, fps=15, macro_block_size=1
            )

        results.append({
            "episode_idx": i,
            "success": bool(ever_success),
            "min_dist": float(min_dist),
            "max_reward": float(max_reward),
            "final_reward": float(elapsed_r),
            "steps": step + 1,
            "cube_pos": actual_cube_pos.tolist(),
            "bowl_pos": bowl_pos.tolist(),
            "yaw_deg": float(yaw_deg),
        })

    sr = sum(successes) / len(successes) if successes else 0
    print(f"\n[{args.difficulty}] Success Rate: {sum(successes)}/{len(successes)} = {sr:.0%}")

    # Save JSON results
    summary = {
        "groot_checkpoint": args.groot_checkpoint,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "num_episodes": len(successes),
        "success_rate": sr,
        "num_successes": sum(successes),
        "episodes": results,
    }
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

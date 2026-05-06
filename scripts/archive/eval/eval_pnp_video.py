"""
PnP video evaluation: record base policy vs residual side-by-side.
Records per-env videos showing base cam + wrist cam with success/fail overlay.

Usage:
    # Residual checkpoint
    python scripts/eval_pnp_video.py --checkpoint outputs/.../checkpoints/best.pt --output-dir outputs/eval_video/E3

    # Base policy only
    python scripts/eval_pnp_video.py --base-only --output-dir outputs/eval_video/base
"""
import sys, os, json, time, argparse, importlib, types as _types
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
import mujoco
import cv2
import imageio

SCENE_XML = os.path.expanduser(
    "~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
)
MAX_STEPS = 500
RENDER_FPS = 20


def load_residual_agent(checkpoint_path, obs_sample, action_dim, device):
    from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_args = ckpt.get("args", {})

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.hidden_dim = saved_args.get("actor_hidden_dim", 256)
    cfg.agent.critic.hidden_dim = saved_args.get("critic_hidden_dim", 256)
    cfg.agent.asymmetric_critic = saved_args.get("asymmetric_critic", True)

    state_dim = obs_sample["observation.state"].shape[-1]
    object_state_dim = 10 if saved_args.get("asymmetric_critic", True) else 0
    _asymmetric = saved_args.get("asymmetric_critic", True)
    rl_cameras = ["observation.depth.front", "observation.depth.wrist"] if _asymmetric else []

    agent = QAgent(
        obs_shape=(1, 84, 84),
        prop_shape=(state_dim,),
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


def render_frame(vec_env, env_idx, step, success, label=""):
    """Render base + wrist camera side-by-side with overlay text."""
    e = vec_env._envs[env_idx]
    data, model = e["data"], e["model"]

    e["renderer_base"].update_scene(data, camera=e["cam_base_id"], scene_option=e["opt_base"])
    frame_base = e["renderer_base"].render().copy()
    e["renderer_wrist"].update_scene(data, camera=e["cam_wrist_id"], scene_option=e["opt_wrist"])
    frame_wrist = e["renderer_wrist"].render().copy()

    h1, w1 = frame_base.shape[:2]
    h2, w2 = frame_wrist.shape[:2]
    h = max(h1, h2)
    combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = frame_base
    combined[:h2, w1:w1 + w2] = frame_wrist

    # Overlay text
    cv2.putText(combined, f"{label} env{env_idx} step={step}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if success:
        cv2.putText(combined, "SUCCESS", (w1 + w2 - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return combined


def run_eval_with_video(positions, groot_ckpt, residual_ckpt, output_dir, label, seed=0, device="cuda"):
    from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
    from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP
    from scipy.spatial.transform import Rotation

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
        reward_type="dense_clipped",
        scene_xml=SCENE_XML,
    )

    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=groot_ckpt,
        task_description="Pick up the red cube and place it onto the plate.",
        ema_alpha=0.0,
    )

    agent = None
    if residual_ckpt:
        obs_tmp, _ = wrapper.reset()
        agent = load_residual_agent(residual_ckpt, obs_tmp, wrapper.action_dim, device)

    obs, _ = wrapper.reset()

    # Override positions (same as eval_multiseed)
    for ei in range(num_envs):
        env_data = vec_env._envs[ei]
        data, model = env_data["data"], env_data["model"]
        cqa = env_data["cube_qposadr"]
        if cqa is not None:
            data.qpos[cqa:cqa + 3] = cube_positions[ei]
            q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
            data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        bowl_body_id = env_data["bowl_body_id"]
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_positions[ei]
        mujoco.mj_forward(model, data)

    vec_env._step_counts = np.zeros(num_envs, dtype=int)
    vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

    obs, _ = wrapper.reset()
    for ei in range(num_envs):
        env_data = vec_env._envs[ei]
        data, model = env_data["data"], env_data["model"]
        cqa = env_data["cube_qposadr"]
        if cqa is not None:
            data.qpos[cqa:cqa + 3] = cube_positions[ei]
            q_xyzw = Rotation.from_euler("z", np.radians(cube_yaws[ei])).as_quat()
            data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        bowl_body_id = env_data["bowl_body_id"]
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_positions[ei]
        mujoco.mj_forward(model, data)

    vec_env._step_counts = np.zeros(num_envs, dtype=int)
    vec_env._episode_rewards = np.zeros(num_envs, dtype=np.float64)

    # Per-env frame buffers and reward tracking
    env_frames = [[] for _ in range(num_envs)]
    env_done = [False] * num_envs
    env_success = [False] * num_envs
    env_cum_reward = np.zeros(num_envs, dtype=np.float64)
    env_max_reward = np.zeros(num_envs, dtype=np.float64)
    env_success_step = [None] * num_envs  # step when terminated (success)

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

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

            # Track reward
            r_val = float(rew[ei]) if hasattr(rew, '__getitem__') else float(rew)
            env_cum_reward[ei] += r_val
            env_max_reward[ei] = max(env_max_reward[ei], r_val)

            t_val = term[ei] if hasattr(term, '__getitem__') else term
            if t_val:
                env_success[ei] = True
                env_success_step[ei] = step

            # Record frame every 2 steps to keep video size manageable
            if step % 2 == 0:
                frame = render_frame(vec_env, ei, step, env_success[ei], label=label)
                env_frames[ei].append(frame)

            d_val = done[ei] if hasattr(done, '__getitem__') else done
            if d_val:
                env_done[ei] = True

        if all(env_done):
            break

    elapsed = time.time() - t0

    # Save per-env videos
    n_success = 0
    for ei in range(num_envs):
        tag = "succ" if env_success[ei] else "fail"
        if env_success[ei]:
            n_success += 1
        vpath = os.path.join(output_dir, f"env{ei:02d}_{tag}.mp4")
        if env_frames[ei]:
            writer = imageio.get_writer(vpath, fps=RENDER_FPS // 2)
            for fr in env_frames[ei]:
                writer.append_data(fr)
            writer.close()
            steps_taken = env_success_step[ei] if env_success_step[ei] is not None else MAX_STEPS
            print(f"  env{ei:2d}: {tag} | steps={steps_taken:3d} | cum_R={env_cum_reward[ei]:.2f} | max_r={env_max_reward[ei]:.3f} → {vpath}", flush=True)

    sr = n_success / num_envs
    avg_cum_r = np.mean(env_cum_reward)
    print(f"\n  {label}: SR={sr*100:.1f}% ({n_success}/{num_envs}) | avg_cum_R={avg_cum_r:.2f} in {elapsed:.0f}s", flush=True)

    # Save results
    result = {
        "label": label,
        "seed": seed,
        "success_rate": sr,
        "n_success": n_success,
        "n_total": num_envs,
        "avg_cumulative_reward": float(avg_cum_r),
        "per_env": {},
    }
    for i in range(num_envs):
        result["per_env"][f"env{i}"] = {
            "success": int(env_success[i]),
            "cumulative_reward": float(env_cum_reward[i]),
            "max_step_reward": float(env_max_reward[i]),
            "steps": env_success_step[i] if env_success_step[i] is not None else MAX_STEPS,
        }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    del wrapper, vec_env
    if agent:
        del agent
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="PnP video eval")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--positions-file", type=str, default="configs/pnp_eval_hard.json")
    parser.add_argument("--groot-checkpoint", type=str,
                        default=os.path.expanduser("~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000"))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.base_only and args.checkpoint is None:
        parser.error("Provide --checkpoint or --base-only")

    positions = json.loads(Path(args.positions_file).read_text())
    label = args.label or ("base" if args.base_only else Path(args.checkpoint).parent.parent.name)

    print(f"=== PnP Video Eval: {label} ===")
    print(f"Positions: {len(positions)} envs")
    print(f"Seed: {args.seed}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(flush=True)

    run_eval_with_video(
        positions=positions,
        groot_ckpt=args.groot_checkpoint,
        residual_ckpt=args.checkpoint if not args.base_only else None,
        output_dir=args.output_dir,
        label=label,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

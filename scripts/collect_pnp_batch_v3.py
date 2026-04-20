"""
Offline data collection v3: 15-env batch inference (same as eval sweep).

Key difference from v2 (1-env sequential):
  - Uses MuJoCoResidualWrapperPnP with 15 envs simultaneously
  - GR00T batch inference → higher SR (~93% vs ~57% on normal)
  - Envs auto-reset independently; per-env episode counter tracks progress

Usage:
    python scripts/collect_pnp_batch_v3.py \
        --positions_file configs/pnp_eval_normal.json \
        --groot_checkpoint ~/DATA/INTERN/training/groot_pnp_sim_66ep/checkpoint-100000 \
        --output_dir outputs/offline_data/pnp_normal_v3_66ep100k \
        --episodes_per_pos 34 --max_episode_steps 500
"""
import sys, os, json, time, argparse, signal, importlib, types as _types
from pathlib import Path

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

try:
    import huggingface_hub.utils._validators as v; o = v.validate_repo_id
    def _p(r):
        if r and (r.startswith("/") or r.startswith(".")): return
        return o(r)
    v.validate_repo_id = _p
except: pass

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation

torch.backends.cuda.enable_cudnn_sdp(False)

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

SCENE_XML = os.path.expanduser(
    "~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
)

_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    print(f"\n[SIGNAL] Received {signal.Signals(sig).name}, finishing current episodes then exit...")
    _shutdown = True


def override_positions(vec_env, positions):
    """Override cube/bowl positions for all envs after reset."""
    for ei in range(vec_env.num_envs):
        env_data = vec_env._envs[ei]
        data, model = env_data["data"], env_data["model"]
        cqa = env_data["cube_qposadr"]
        pos = positions[ei]
        cube_pos = pos["cube_pos"]
        bowl_pos = pos["bowl_pos"]
        yaw_deg = pos.get("cube_yaw_deg", 0.0)

        if cqa is not None:
            data.qpos[cqa:cqa + 3] = cube_pos
            q_xyzw = Rotation.from_euler("z", np.radians(yaw_deg)).as_quat()
            data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        bowl_body_id = env_data["bowl_body_id"]
        if bowl_body_id >= 0:
            model.body_pos[bowl_body_id] = bowl_pos

        env_data["cube_pos_init"] = np.array(cube_pos)
        env_data["bowl_pos_init"] = np.array(bowl_pos)
        mujoco.mj_forward(model, data)

    vec_env._step_counts = np.zeros(vec_env.num_envs, dtype=int)
    vec_env._episode_rewards = np.zeros(vec_env.num_envs, dtype=np.float64)


def override_single_env(vec_env, ei, pos):
    """Override position for a single env after auto-reset."""
    env_data = vec_env._envs[ei]
    data, model = env_data["data"], env_data["model"]
    cqa = env_data["cube_qposadr"]
    cube_pos = pos["cube_pos"]
    bowl_pos = pos["bowl_pos"]
    yaw_deg = pos.get("cube_yaw_deg", 0.0)

    if cqa is not None:
        data.qpos[cqa:cqa + 3] = cube_pos
        q_xyzw = Rotation.from_euler("z", np.radians(yaw_deg)).as_quat()
        data.qpos[cqa + 3:cqa + 7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    bowl_body_id = env_data["bowl_body_id"]
    if bowl_body_id >= 0:
        model.body_pos[bowl_body_id] = bowl_pos

    env_data["cube_pos_init"] = np.array(cube_pos)
    env_data["bowl_pos_init"] = np.array(bowl_pos)
    mujoco.mj_forward(model, data)
    vec_env._step_counts[ei] = 0
    vec_env._episode_rewards[ei] = 0.0
    vec_env._initial_cube_z[ei] = data.qpos[cqa + 2] if cqa is not None else 0.0


def _to_np(t):
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    p = argparse.ArgumentParser(description="Batch offline data collection v3")
    p.add_argument("--positions_file", type=str, required=True)
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--episodes_per_pos", type=int, default=34)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema_alpha", type=float, default=0.0)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load positions
    with open(args.positions_file) as f:
        positions = json.load(f)
    num_envs = len(positions)
    total_episodes = num_envs * args.episodes_per_pos

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume: scan existing episodes (+ lock files from other jobs)
    def scan_existing():
        """Scan for completed (.npz) and claimed (.lock) episodes."""
        taken = set()
        for path in out_dir.glob("ep_*.npz"):
            try:
                taken.add(int(path.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass
        for path in out_dir.glob("ep_*.lock"):
            try:
                taken.add(int(path.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass
        return taken

    existing = scan_existing()
    resumed_from = len([e for e in existing if (out_dir / f"ep_{e:04d}.npz").exists()])
    if resumed_from > 0:
        print(f"Resume: found {resumed_from} existing + {len(existing)-resumed_from} claimed episodes.", flush=True)

    def claim_next_episode(ei):
        """Find and atomically claim next available episode for position ei.
        Returns global episode index, or -1 if all done."""
        base = ei * args.episodes_per_pos
        for off in range(args.episodes_per_pos):
            gidx = base + off
            lock_path = out_dir / f"ep_{gidx:04d}.lock"
            npz_path = out_dir / f"ep_{gidx:04d}.npz"
            if npz_path.exists():
                continue
            # Try atomic lock
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.close(fd)
                return gidx
            except FileExistsError:
                continue
        return -1

    # Per-env episode tracking
    env_ep_count = [0] * num_envs  # how many completed for this env
    env_ep_global = [0] * num_envs  # current claimed episode
    env_active = [True] * num_envs
    env_transitions = [[] for _ in range(num_envs)]

    # Count already-done per position
    for ei in range(num_envs):
        base = ei * args.episodes_per_pos
        for off in range(args.episodes_per_pos):
            if (out_dir / f"ep_{base+off:04d}.npz").exists():
                env_ep_count[ei] += 1

    # Claim first episode for each env
    for ei in range(num_envs):
        gidx = claim_next_episode(ei)
        if gidx < 0:
            env_active[ei] = False
        else:
            env_ep_global[ei] = gidx

    completed = sum(env_ep_count)
    if not any(env_active):
        print(f"All episodes already collected/claimed. Done.", flush=True)
        return

    print(f"=== Batch Offline Data Collection v3 ===", flush=True)
    print(f"  Positions: {num_envs}", flush=True)
    print(f"  Episodes/position: {args.episodes_per_pos}", flush=True)
    print(f"  Total episodes: {total_episodes}", flush=True)
    print(f"  Already done: {completed}", flush=True)
    print(f"  Remaining: {total_episodes - completed}", flush=True)
    print(f"  Max steps/episode: {args.max_episode_steps}", flush=True)
    print(flush=True)

    # Create envs
    print("Creating environments...", flush=True)
    cube_positions = [p["cube_pos"] for p in positions]
    bowl_positions = [p["bowl_pos"] for p in positions]

    vec_env = MuJoCoVecEnvPnP(
        num_envs=num_envs,
        cube_positions=cube_positions,
        bowl_positions=bowl_positions,
        max_episode_steps=args.max_episode_steps,
        reward_type="dense",
        scene_xml=SCENE_XML,
    )

    print("Loading GR00T policy...", flush=True)
    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=args.groot_checkpoint,
        task_description="Pick up the red cube and place it onto the plate.",
        ema_alpha=args.ema_alpha,
    )

    # Initial reset + position override
    obs, _ = wrapper.reset()
    override_positions(vec_env, positions)
    # Re-reset to get correct obs after override
    obs, _ = wrapper.reset()
    override_positions(vec_env, positions)

    print("Starting collection...\n", flush=True)
    t_start = time.time()
    total_success = 0
    total_done = completed
    total_transitions = 0

    # Main loop: step all envs simultaneously
    step_in_ep = np.zeros(num_envs, dtype=int)

    while any(env_active) and not _shutdown:
        # Zero residual for all envs
        residual = torch.zeros((num_envs, wrapper.action_dim), dtype=torch.float32)

        # Save pre-step obs per active env
        prev_obs_per_env = {}
        for ei in range(num_envs):
            if env_active[ei]:
                prev_obs_per_env[ei] = {
                    k: v[ei].clone() if isinstance(v, torch.Tensor) else v
                    for k, v in obs.items()
                }

        # Batch step
        next_obs, reward, terminated, truncated, info = wrapper.step(residual)
        done = terminated | truncated

        # Process each env
        for ei in range(num_envs):
            if not env_active[ei]:
                continue

            # Record transition
            po = prev_obs_per_env[ei]
            r_env = float(reward[ei]) if hasattr(reward, "__getitem__") else float(reward)

            transition = {
                "obs_state": _to_np(po.get("observation.state", torch.zeros(10))),
                "obs_base_action": _to_np(po.get("observation.base_action", torch.zeros(7))),
                "obs_object_state": _to_np(po.get("observation.object_state", torch.zeros(10))),
                "obs_depth_front": _to_np(po.get("observation.depth.front", torch.zeros(1, 84, 84))),
                "obs_depth_wrist": _to_np(po.get("observation.depth.wrist", torch.zeros(1, 84, 84))),
                "action": np.zeros(7, dtype=np.float32),  # zero residual
                "reward": np.float32(r_env),
                "done": bool(done[ei]),
                "terminated": bool(terminated[ei]),
                "env_id": ei,
                "step": int(step_in_ep[ei]),
            }
            env_transitions[ei].append(transition)
            step_in_ep[ei] += 1

            # Episode done?
            d_val = done[ei].item() if hasattr(done[ei], 'item') else bool(done[ei])
            if d_val or step_in_ep[ei] >= args.max_episode_steps:
                success = bool(terminated[ei].item() if hasattr(terminated[ei], 'item') else terminated[ei])
                ep_global = env_ep_global[ei]
                n_steps = len(env_transitions[ei])

                # Save npz
                ep_data = env_transitions[ei]
                tmp_path = out_dir / f"ep_{ep_global:04d}.tmp.npz"
                final_path = out_dir / f"ep_{ep_global:04d}.npz"

                np.savez_compressed(
                    tmp_path,
                    obs_state=np.array([t["obs_state"] for t in ep_data]),
                    obs_base_action=np.array([t["obs_base_action"] for t in ep_data]),
                    obs_object_state=np.array([t["obs_object_state"] for t in ep_data]),
                    obs_depth_front=np.array([t["obs_depth_front"] for t in ep_data]),
                    obs_depth_wrist=np.array([t["obs_depth_wrist"] for t in ep_data]),
                    action=np.array([t["action"] for t in ep_data]),
                    reward=np.array([t["reward"] for t in ep_data]),
                    done=np.array([t["done"] for t in ep_data]),
                    terminated=np.array([t["terminated"] for t in ep_data]),
                    env_id=np.array([t["env_id"] for t in ep_data]),
                    step_idx=np.array([t["step"] for t in ep_data]),
                    success=np.array([success]),
                )
                tmp_path.rename(final_path)
                # Remove lock file
                lock_path = out_dir / f"ep_{ep_global:04d}.lock"
                lock_path.unlink(missing_ok=True)

                if success:
                    total_success += 1
                total_done += 1
                total_transitions += n_steps

                tag = "SUCC" if success else "FAIL"
                elapsed = time.time() - t_start
                sr = total_success / max(total_done - completed, 1) * 100

                env_ep_count[ei] += 1

                # Print progress every 5 episodes
                if total_done % 5 == 0 or not any(env_active):
                    print(f"  ep={total_done}/{total_episodes} {tag} steps={n_steps} "
                          f"SR={sr:.1f}% ({total_success}/{total_done-completed}) "
                          f"t={elapsed:.0f}s", flush=True)

                # Clear buffer
                env_transitions[ei] = []
                step_in_ep[ei] = 0

                # Claim next episode for this env
                gidx = claim_next_episode(ei)
                if gidx < 0:
                    env_active[ei] = False
                    print(f"  [pos{ei}] Complete: {env_ep_count[ei]} episodes", flush=True)
                else:
                    env_ep_global[ei] = gidx
                    # Position override after auto-reset
                    override_single_env(vec_env, ei, positions[ei])

        obs = next_obs

    # Final summary
    elapsed = time.time() - t_start
    new_done = total_done - completed
    sr = total_success / max(new_done, 1) * 100
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL: {total_done} episodes ({new_done} new), {total_transitions} transitions", flush=True)
    print(f"SR={sr:.1f}% ({total_success}/{new_done})", flush=True)
    print(f"Time: {elapsed:.0f}s ({elapsed/max(new_done,1):.1f}s/ep)", flush=True)

    # Save result.json
    result = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "total_episodes": total_done,
        "new_episodes": new_done,
        "total_transitions": total_transitions,
        "success_rate": sr,
        "n_success": total_success,
        "episodes_per_pos": args.episodes_per_pos,
        "max_episode_steps": args.max_episode_steps,
        "checkpoint": args.groot_checkpoint,
        "ema_alpha": args.ema_alpha,
        "positions_file": args.positions_file,
        "num_envs": num_envs,
        "batch_mode": True,
        "resumed_from": resumed_from,
        "time_seconds": elapsed,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Clean up any lock files owned by this process
    for path in out_dir.glob("ep_*.lock"):
        try:
            with open(path) as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                path.unlink(missing_ok=True)
        except (ValueError, OSError):
            pass

    print(f"Result: {out_dir / 'result.json'}", flush=True)
    print("Done.", flush=True)

    wrapper.close() if hasattr(wrapper, 'close') else None
    vec_env.close()


if __name__ == "__main__":
    main()

"""Parallel PnP eval: 5 envs simultaneously with RAM/GPU monitoring."""
import sys, os, json, time, importlib, types as _types
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Deepspeed mock
if "deepspeed" not in sys.modules:
    m = _types.ModuleType("deepspeed"); m.__version__="0.0.0"
    m.__spec__=importlib.machinery.ModuleSpec("deepspeed",None); m.__path__=[]; m.__file__="fake"
    z = _types.ModuleType("deepspeed.zero"); z.__spec__=importlib.machinery.ModuleSpec("deepspeed.zero",None)
    z.Init=lambda *a,**kw:(lambda f:f); m.zero=z
    sys.modules["deepspeed"]=m; sys.modules["deepspeed.zero"]=z

try:
    import huggingface_hub.utils._validators as v; o=v.validate_repo_id
    def p(r):
        if r and (r.startswith("/") or r.startswith(".")): return
        return o(r)
    v.validate_repo_id=p
except: pass

_repo = str(Path(__file__).resolve().parents[1] / "residual-offpolicy-rl")
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import numpy as np
import torch
import psutil
import subprocess
torch.backends.cuda.enable_cudnn_sdp(False)

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

FFMPEG = os.path.expanduser(
    "~/.venvs/groot/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


def get_mem_usage():
    proc = psutil.Process(os.getpid())
    ram_gb = proc.memory_info().rss / (1024**3)
    try:
        gpu_mb = torch.cuda.memory_allocated() / (1024**2)
        gpu_reserved_mb = torch.cuda.memory_reserved() / (1024**2)
    except:
        gpu_mb = gpu_reserved_mb = 0
    return ram_gb, gpu_mb, gpu_reserved_mb


def main():
    # Config
    GROOT_CKPT = os.path.expanduser("~/DATA/INTERN/training/groot_pnp_sim_100ep/checkpoint-100000")
    POSITIONS_FILE = os.path.join(_repo, "configs/pnp_sim33ep_positions.json")
    SCENE_XML = os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml")
    VIEW_DIR = os.path.expanduser("~/Repos/Intern/.VIEW")
    RESULT_VIDEO = os.path.join(VIEW_DIR, "result.mp4")
    MAX_STEPS = 500
    NUM_EPISODES = 2
    SEED = 42

    os.makedirs(VIEW_DIR, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load first 5 positions
    with open(POSITIONS_FILE) as f:
        all_pos = json.load(f)
    positions = all_pos[:5]
    num_envs = len(positions)

    cube_positions = [p["cube_pos"] for p in positions]
    bowl_positions = [p["bowl_pos"] for p in positions]
    cube_yaws = [p.get("cube_yaw_deg", 0.0) for p in positions]

    print(f"[Config] {num_envs} envs x {NUM_EPISODES} episodes, {MAX_STEPS} steps")
    print(f"[Config] GR00T: {GROOT_CKPT}")

    # Memory before loading
    ram, gpu, gpu_r = get_mem_usage()
    print(f"\n[MEM baseline] RAM={ram:.1f}GB  GPU_alloc={gpu:.0f}MB  GPU_reserved={gpu_r:.0f}MB")

    # Create env
    t0 = time.time()
    vec_env = MuJoCoVecEnvPnP(
        num_envs=num_envs,
        cube_positions=cube_positions,
        bowl_positions=bowl_positions,
        max_episode_steps=MAX_STEPS,
        reward_type="dense",
        scene_xml=SCENE_XML,
    )
    ram, gpu, gpu_r = get_mem_usage()
    print(f"[MEM after env] RAM={ram:.1f}GB  GPU_alloc={gpu:.0f}MB  GPU_reserved={gpu_r:.0f}MB  ({time.time()-t0:.1f}s)")

    # Create wrapper (loads GR00T)
    t1 = time.time()
    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=GROOT_CKPT,
        task_description="Pick up the red cube and place it onto the plate.",
        ema_alpha=0.0,
    )
    ram, gpu, gpu_r = get_mem_usage()
    print(f"[MEM after GR00T] RAM={ram:.1f}GB  GPU_alloc={gpu:.0f}MB  GPU_reserved={gpu_r:.0f}MB  ({time.time()-t1:.1f}s)")

    # Set cube yaws after init
    from scipy.spatial.transform import Rotation
    import mujoco

    all_results = []
    total_success = 0
    total_episodes = 0
    last_ep_frames = None

    for ep in range(NUM_EPISODES):
        print(f"\n{'='*60}")
        print(f"Episode batch {ep+1}/{NUM_EPISODES}")

        # Reset
        obs = wrapper.reset()

        # Override cube positions + yaws
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

        # Rebuild obs after position override
        obs = wrapper.reset()
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
        env_min_dist = [999.0] * num_envs
        # Capture frames for grid video (every 3rd step)
        ep_frames = [[] for _ in range(num_envs)]

        for step in range(MAX_STEPS):
            # Zero residual = GR00T base only
            action = np.zeros((num_envs, wrapper.action_dim))
            obs, rew, term, trunc, info = wrapper.step(action)

            # Capture frames every 3 steps
            if step % 3 == 0:
                for ei in range(num_envs):
                    if not env_done[ei]:
                        frame = vec_env.get_frame(ei, camera='back', size=(240, 320))
                        ep_frames[ei].append(frame)

            done = term | trunc if hasattr(term, '__or__') else np.array(term) | np.array(trunc)

            for ei in range(num_envs):
                if env_done[ei]:
                    continue
                # Check cube-bowl distance
                cqa = vec_env._envs[ei]["cube_qposadr"]
                if cqa is not None:
                    cp = vec_env._envs[ei]["data"].qpos[cqa:cqa+3].copy()
                    d = np.linalg.norm(cp[:2] - np.array(bowl_positions[ei][:2]))
                    env_min_dist[ei] = min(env_min_dist[ei], d)
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

            # Log memory every 100 steps
            if step > 0 and step % 100 == 0:
                ram, gpu, gpu_r = get_mem_usage()
                print(f"  [step {step}] RAM={ram:.1f}GB  GPU_alloc={gpu:.0f}MB")

        # Episode results
        n_succ = sum(env_success)
        total_success += n_succ
        total_episodes += num_envs
        print(f"  Results: {n_succ}/{num_envs} success")
        for ei in range(num_envs):
            tag = "OK" if env_success[ei] else "FAIL"
            print(f"    env{ei} (ep{positions[ei]['episode']}): min_dist={env_min_dist[ei]:.3f}m {tag}")
            all_results.append({
                "batch": ep, "env": ei, "episode": positions[ei]["episode"],
                "success": env_success[ei], "min_dist": env_min_dist[ei],
            })

        # Collect frames for final video (last batch)
        if ep == NUM_EPISODES - 1:
            last_ep_frames = ep_frames

    # Final memory
    ram, gpu, gpu_r = get_mem_usage()
    print(f"\n[MEM final] RAM={ram:.1f}GB  GPU_alloc={gpu:.0f}MB  GPU_reserved={gpu_r:.0f}MB")

    sr = total_success / total_episodes if total_episodes > 0 else 0
    print(f"\nOVERALL SR: {total_success}/{total_episodes} = {sr:.0%}")

    # Save results JSON
    summary = {
        "groot_checkpoint": GROOT_CKPT,
        "num_envs": num_envs, "num_episodes": NUM_EPISODES,
        "max_steps": MAX_STEPS, "success_rate": sr,
        "total_success": total_success, "total_episodes": total_episodes,
        "episodes": all_results,
    }
    with open(os.path.join(VIEW_DIR, "result.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {VIEW_DIR}/result.json")

    # Build grid video from last batch frames and save as result.mp4
    if imageio is not None and last_ep_frames is not None:
        max_len = max(len(f) for f in last_ep_frames)
        grid_frames = []
        for t in range(max_len):
            row = []
            for ei in range(num_envs):
                if t < len(last_ep_frames[ei]):
                    row.append(last_ep_frames[ei][t])
                elif last_ep_frames[ei]:
                    row.append(last_ep_frames[ei][-1])  # repeat last frame
            if row:
                grid_frames.append(np.concatenate(row, axis=1))

        tmp_path = RESULT_VIDEO + ".tmp.mp4"
        imageio.mimwrite(tmp_path, grid_frames, fps=15, macro_block_size=1)
        # Convert to avc1
        subprocess.run([
            FFMPEG, "-y", "-i", tmp_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-tag:v", "avc1", "-crf", "18",
            RESULT_VIDEO
        ], capture_output=True)
        os.remove(tmp_path)
        print(f"Video saved to {RESULT_VIDEO} (avc1, {len(grid_frames)} frames)")


if __name__ == "__main__":
    main()

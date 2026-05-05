import sys, os, importlib, json, types as _types
from pathlib import Path
os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
if "deepspeed" not in sys.modules:
    m = _types.ModuleType("deepspeed"); m.__version__="0.0.0"; m.__spec__=importlib.machinery.ModuleSpec("deepspeed",None); m.__path__=[]; m.__file__="fake"
    z = _types.ModuleType("deepspeed.zero"); z.__spec__=importlib.machinery.ModuleSpec("deepspeed.zero",None); z.Init=lambda *a,**kw:(lambda f:f); m.zero=z
    sys.modules["deepspeed"]=m; sys.modules["deepspeed.zero"]=z
try:
    import huggingface_hub.utils._validators as v; o=v.validate_repo_id
    def p(r):
        if r and (r.startswith("/") or r.startswith(".")): return
        return o(r)
    v.validate_repo_id=p
except: pass
_repo = str(Path(__file__).resolve().parents[3])
if _repo not in sys.path: sys.path.insert(0, _repo)
import numpy as np, torch, cv2
torch.backends.cuda.enable_cudnn_sdp(False)
from scipy.spatial.transform import Rotation
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv

EMA = 0.7
BATCH_SIZE = 8  # run 8 envs at a time to limit memory
GROOT_CKPT = os.path.expanduser("~/DATA/INTERN/training/gr00t_sim_v2/checkpoint-20000")
SCENE_XML = os.path.expanduser("~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml")
CUBE_INFO_DIR = os.path.expanduser("~/DATA/INTERN/datasets/cube_info/")
OUTDIR = "outputs/eval_videos/sim_groot_v2"
os.makedirs(OUTDIR, exist_ok=True)

# Load cube info for all 32 episodes (sorted by timestamp = episode order)
cube_dirs = sorted([d for d in os.listdir(CUBE_INFO_DIR) if d.startswith("kinam_v2_20260406")])
assert len(cube_dirs) == 32, f"Expected 32, got {len(cube_dirs)}"
cube_infos = []
for d in cube_dirs:
    with open(os.path.join(CUBE_INFO_DIR, d, "cube_info.json")) as f:
        cube_infos.append(json.load(f))

print(f"Sim GR00T eval | EMA={EMA} | {len(cube_infos)} episodes | ckpt={GROOT_CKPT}", flush=True)

total_success = 0
total_episodes = 0

# Process in batches
for batch_start in range(0, 32, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, 32)
    batch_infos = cube_infos[batch_start:batch_end]
    ne = len(batch_infos)

    # Extract cube positions and yaws
    cube_positions = [ci["cube_table_pos"] for ci in batch_infos]

    # Create env with fixed cube positions (no random)
    me = MuJoCoVecEnv(
        num_envs=ne,
        cube_positions=cube_positions,
        cube_yaw_deg=0.0,  # we set yaw per-env below
        random_cube_range=None,
        scene_xml=SCENE_XML,
        max_episode_steps=500,
        reward_type="dense",
        device="cuda:0",
    )

    # Set per-env cube quaternion from cube_info
    for i, ci in enumerate(batch_infos):
        quat_wxyz = ci["cube_table_quat_wxyz"]
        env_data = me._envs[i]
        cqa = env_data["cube_qposadr"]
        if cqa is not None:
            env_data["data"].qpos[cqa + 3:cqa + 7] = quat_wxyz
            import mujoco
            mujoco.mj_forward(env_data["model"], env_data["data"])

    env = MuJoCoResidualWrapper(
        vec_env=me,
        groot_checkpoint=GROOT_CKPT,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device="cuda:0",
        task_description="lift the cube",
        open_loop_horizon=16,
        ema_alpha=EMA,
    )

    obs, _ = env.reset()

    # Set cube positions again after reset (reset may overwrite)
    for i, ci in enumerate(batch_infos):
        env_data = me._envs[i]
        cqa = env_data["cube_qposadr"]
        if cqa is not None:
            env_data["data"].qpos[cqa:cqa + 3] = ci["cube_table_pos"]
            env_data["data"].qpos[cqa + 3:cqa + 7] = ci["cube_table_quat_wxyz"]
            env_data["cube_pos_init"] = np.array(ci["cube_table_pos"])
            import mujoco
            mujoco.mj_forward(env_data["model"], env_data["data"])

    # Re-get obs after cube fix
    obs, _ = env.reset()
    # Fix cube again after second reset
    for i, ci in enumerate(batch_infos):
        env_data = me._envs[i]
        cqa = env_data["cube_qposadr"]
        if cqa is not None:
            env_data["data"].qpos[cqa:cqa + 3] = ci["cube_table_pos"]
            env_data["data"].qpos[cqa + 3:cqa + 7] = ci["cube_table_quat_wxyz"]
            env_data["cube_pos_init"] = np.array(ci["cube_table_pos"])
            import mujoco
            mujoco.mj_forward(env_data["model"], env_data["data"])

    # Collect frames
    all_frames = {i: [] for i in range(ne)}
    ed = [False] * ne
    es = [False] * ne

    for st in range(500):
        zero_action = torch.zeros(ne, env.action_dim, device="cuda:0")
        obs, r, tm, tr, inf = env.step(zero_action)
        d = tm | tr
        for i in range(ne):
            if not ed[i]:
                fb = me.get_frame(env_id=i, camera="back", size=(360, 640))
                fw = me.get_frame(env_id=i, camera="wrist", size=(360, 640))
                fb_bgr = cv2.cvtColor(fb, cv2.COLOR_RGB2BGR)
                fw_bgr = cv2.cvtColor(fw, cv2.COLOR_RGB2BGR)
                combined = np.hstack([fb_bgr, fw_bgr])
                all_frames[i].append(combined)
                if d[i]:
                    ed[i] = True
                    if tm[i]:
                        es[i] = True
        if all(ed):
            break

    # Save videos and report
    for i in range(ne):
        ep_idx = batch_start + i
        status = "OK" if es[i] else "FAIL"
        total_success += int(es[i])
        total_episodes += 1
        ci = batch_infos[i]
        pos = ci["cube_table_pos"]
        yaw = ci["yaw_deg"]
        print(f"  ep{ep_idx:2d}: {status} ({len(all_frames[i])} steps) pos=[{pos[0]:.3f},{pos[1]:.3f}] yaw={yaw:.1f}", flush=True)

        # Save video
        frames = all_frames[i]
        if frames:
            H, W = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vpath = f"{OUTDIR}/ep{ep_idx:02d}_{status}.mp4"
            out = cv2.VideoWriter(vpath, fourcc, 30, (W, H))
            for f in frames:
                out.write(f)
            out.release()

    del env, me
    torch.cuda.empty_cache()
    print(f"  Batch {batch_start}-{batch_end-1} done. Running SR: {total_success}/{total_episodes}", flush=True)

print(f"\n=== FINAL: SR={total_success}/{total_episodes} = {total_success/total_episodes:.1%} ===", flush=True)

#!/usr/bin/env python3
"""Debug: run 5 episodes per difficulty (easy/normal/hard), save videos."""
import sys, os, json, time
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP as MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP as MuJoCoResidualWrapper

REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])

def run_episode(wrapper, mujoco_env, cube_pos, bowl_pos, yaw_deg, label):
    obs, _ = wrapper.reset()
    env_data = mujoco_env._envs[0]
    data, model = env_data["data"], env_data["model"]
    cqa = env_data["cube_qposadr"]
    bowl_body_id = env_data["bowl_body_id"]

    for j, jid in enumerate(env_data["ids"]["jnt_ids"]):
        data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]
    for fid in env_data["ids"]["finger_ids"]:
        if fid >= 0:
            data.qpos[model.jnt_qposadr[fid]] = 0.04
    if cqa is not None:
        data.qpos[cqa:cqa+3] = cube_pos
        q = Rotation.from_euler("z", np.radians(yaw_deg)).as_quat()
        data.qpos[cqa+3:cqa+7] = [q[3], q[0], q[1], q[2]]
    if bowl_body_id >= 0:
        model.body_pos[bowl_body_id] = bowl_pos
    env_data["cube_pos_init"] = cube_pos.copy()
    env_data["bowl_pos_init"] = bowl_pos.copy()
    env_data["grasp_state"] = {"grasped": False, "contact_count": 0}
    # Zero velocities
    data.qvel[:] = 0.0
    # Set ctrl targets to current joint positions (physics-based)
    for aid, jid in zip(env_data["arm_actuator_ids"], env_data["ids"]["jnt_ids"]):
        if aid >= 0:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
    for aid in env_data["finger_actuator_ids"]:
        data.ctrl[aid] = 0.04  # open
    mujoco.mj_forward(model, data)
    mujoco_env._step_counts = np.zeros(1, dtype=int)
    if hasattr(mujoco_env, '_episode_rewards'):
        mujoco_env._episode_rewards = np.zeros(1, dtype=np.float64)
    mujoco_env._initial_cube_z[0] = data.qpos[cqa + 2] if cqa is not None else 0.0

    frames = []
    success = False
    for step in range(500):
        residual = torch.zeros((1, 7), dtype=torch.float32)
        next_obs, reward, terminated, truncated, info = wrapper.step(residual)
        if step % 5 == 0:
            frame = mujoco_env.get_frame(0, camera="back", size=(360, 640))
            frames.append(frame)
        done = terminated | truncated
        if done[0]:
            success = bool(terminated[0])
            break
        obs = next_obs
    
    tag = "SUCC" if success else "FAIL"
    print(f"  {label}: {tag} steps={step+1}", flush=True)
    return frames, success

def main():
    positions = json.load(open("configs/pnp_common5_positions.json"))
    # Deduplicate
    seen = set()
    unique = []
    for p in positions:
        if p["episode"] not in seen:
            seen.add(p["episode"])
            unique.append(p)
    
    scene_xml = "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
    groot_ckpt = "/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-100000"
    
    first = unique[0]
    mujoco_env = MuJoCoVecEnv(
        num_envs=1,
        cube_positions=[first["cube_pos"]],
        bowl_positions=[first["bowl_pos"]],
        scene_xml=scene_xml,
        max_episode_steps=500,
        reward_type="dense_clipped",
        success_threshold=0.095,
    )
    wrapper = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=groot_ckpt,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device="cuda:0",
        task_description="Pick up the red cube and place it onto the plate.",
        ema_alpha=0.0,
    )
    
    import imageio
    rng = np.random.default_rng(42)
    
    difficulties = {
        "easy": None,
        "normal": {"dx": (-0.03, 0.03), "dy": (-0.03, 0.03), "yaw": (-15.0, 15.0)},
        "hard": {"dx": (-0.06, 0.06), "dy": (-0.06, 0.06), "yaw": (-45.0, 45.0)},
    }
    
    all_results = {}
    for diff_name, perturb in difficulties.items():
        print(f"\n{'='*40} {diff_name.upper()} {'='*40}")
        diff_frames = []
        diff_success = []
        for i, pos in enumerate(unique):
            cube_pos = np.array(pos["cube_pos"], dtype=np.float64)
            bowl_pos = np.array(pos["bowl_pos"], dtype=np.float64)
            yaw_deg = pos.get("cube_yaw_deg", 0.0)
            
            if perturb is not None:
                cube_pos[0] += rng.uniform(*perturb["dx"])
                cube_pos[1] += rng.uniform(*perturb["dy"])
                yaw_deg += rng.uniform(*perturb["yaw"])
            
            frames, succ = run_episode(wrapper, mujoco_env, cube_pos, bowl_pos, yaw_deg,
                                       f"{diff_name}_pos{i}")
            diff_frames.append(frames)
            diff_success.append(succ)
        
        sr = sum(diff_success) / len(diff_success) * 100
        print(f"  {diff_name} SR = {sr:.0f}% ({sum(diff_success)}/{len(diff_success)})")
        all_results[diff_name] = (diff_frames, diff_success)
        
        # Save individual videos
        for i in range(5):
            tag = "SUCC" if diff_success[i] else "FAIL"
            imageio.mimsave(f"outputs/debug_{diff_name}_pos{i}_{tag}.mp4", diff_frames[i], fps=12)
    
    # Build combined grid: 3 rows (easy/normal/hard) x 5 cols (positions)
    print("\nBuilding combined grid video...")
    diffs = ["easy", "normal", "hard"]
    maxlen = 0
    for d in diffs:
        for vf in all_results[d][0]:
            maxlen = max(maxlen, len(vf))
    
    # Pad all to maxlen
    for d in diffs:
        for vi in range(5):
            vf = all_results[d][0][vi]
            while len(vf) < maxlen:
                vf.append(vf[-1])
    
    from PIL import Image, ImageDraw
    merged = []
    h, w = 360, 640
    # Scale down to fit: 5 cols x 3 rows at 320x180
    sh, sw = 180, 320
    for t in range(maxlen):
        rows = []
        for di, d in enumerate(diffs):
            row_frames = []
            for pi in range(5):
                f = np.array(all_results[d][0][pi][t])
                img = Image.fromarray(f).resize((sw, sh))
                draw = ImageDraw.Draw(img)
                succ = all_results[d][1][pi]
                color = (0,255,0) if succ else (255,80,80)
                tag = "OK" if succ else "X"
                draw.rectangle([0,0,120,16], fill=(0,0,0))
                draw.text((2,1), f"{d} p{pi} {tag}", fill=color)
                row_frames.append(np.array(img))
            rows.append(np.concatenate(row_frames, axis=1))
        grid = np.concatenate(rows, axis=0)
        merged.append(grid)
    
    imageio.mimsave("outputs/debug_all_difficulties.mp4", merged, fps=12, macro_block_size=1)
    print(f"Saved combined: {len(merged)} frames, {merged[0].shape}")
    
    # Print summary
    print("\n" + "="*60)
    for d in diffs:
        sr = sum(all_results[d][1]) / 5 * 100
        tags = ["OK" if s else "X" for s in all_results[d][1]]
        print(f"  {d:6s}: SR={sr:3.0f}%  {tags}")

if __name__ == "__main__":
    main()

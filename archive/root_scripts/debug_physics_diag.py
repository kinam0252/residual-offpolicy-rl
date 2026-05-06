#!/usr/bin/env python3
"""Diagnostic: run all 5 positions, log contacts, save per-position videos."""
import sys, os, json, time
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP as MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP as MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import PHYSICS_SUBSTEPS, get_tcp_pose

REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])

def setup_episode(wrapper, mujoco_env, cube_pos, bowl_pos, yaw_deg):
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
    data.qvel[:] = 0.0
    for aid, jid in zip(env_data["arm_actuator_ids"], env_data["ids"]["jnt_ids"]):
        if aid >= 0:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
    for aid in env_data["finger_actuator_ids"]:
        data.ctrl[aid] = 0.04
    mujoco.mj_forward(model, data)
    mujoco_env._step_counts = np.zeros(1, dtype=int)
    mujoco_env._initial_cube_z[0] = data.qpos[cqa + 2] if cqa is not None else 0.0
    return obs

def run_episode(wrapper, mujoco_env, label, max_steps=500):
    env_data = mujoco_env._envs[0]
    data, model = env_data["data"], env_data["model"]
    cqa = env_data["cube_qposadr"]

    frames = []
    had_contact = False
    max_contact_count = 0
    min_dist = 999.0
    for step in range(max_steps):
        residual = torch.zeros((1, 7), dtype=torch.float32)
        next_obs, reward, terminated, truncated, info = wrapper.step(residual)

        tcp, _ = get_tcp_pose(model, data, env_data["ids"]["hand_id"])
        cube_p = data.qpos[cqa:cqa+3].copy()
        dist = np.linalg.norm(tcp - cube_p)
        min_dist = min(min_dist, dist)
        cc = env_data["grasp_state"]["contact_count"]
        max_contact_count = max(max_contact_count, cc)
        contacts = mujoco_env._get_contacts(0)
        if contacts[2] > 0.5:
            had_contact = True

        if step % 50 == 0:
            finger0 = data.qpos[model.jnt_qposadr[env_data["ids"]["finger_ids"][0]]]
            print(f"  step {step:3d}: dist={dist:.4f} tcp_z={tcp[2]:.4f} cube_z={cube_p[2]:.4f} finger={finger0:.4f} contact={contacts} cc={cc}")

        if step % 5 == 0:
            frame = mujoco_env.get_frame(0, camera="front", size=(480, 640))
            frames.append(frame)

        done = terminated | truncated
        if done[0]:
            success = bool(terminated[0])
            print(f"  {label}: {'SUCC' if success else 'FAIL'} step={step+1} min_dist={min_dist:.4f} max_cc={max_contact_count} had_contact={had_contact}")
            return frames, success, had_contact

    print(f"  {label}: FAIL step=500 min_dist={min_dist:.4f} max_cc={max_contact_count} had_contact={had_contact}")
    return frames, False, had_contact

def main():
    positions = json.load(open("configs/pnp_common5_positions.json"))
    seen = set()
    unique = []
    for p in positions:
        if p["episode"] not in seen:
            seen.add(p["episode"])
            unique.append(p)

    scene_xml = "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
    groot_ckpt = "/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_100ep/checkpoint-100000"

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

    print(f"PHYSICS_SUBSTEPS={PHYSICS_SUBSTEPS}, dt={mujoco_env._envs[0]['model'].opt.timestep}")
    print(f"action_dt={mujoco_env._envs[0]['model'].opt.timestep * PHYSICS_SUBSTEPS:.4f}s\n")

    results = []
    all_frames = []
    for pi, pos in enumerate(unique):
        cube_pos = np.array(pos["cube_pos"], dtype=np.float64)
        bowl_pos = np.array(pos["bowl_pos"], dtype=np.float64)
        yaw_deg = pos.get("cube_yaw_deg", 0.0)
        print(f"\n{'='*60}")
        print(f"Position {pi}: cube={cube_pos}, yaw={yaw_deg}")

        setup_episode(wrapper, mujoco_env, cube_pos, bowl_pos, yaw_deg)
        frames, success, had_contact = run_episode(wrapper, mujoco_env, f"pos{pi}")
        results.append((success, had_contact))
        all_frames.append(frames)

        tag = "SUCC" if success else "FAIL"
        imageio.mimsave(f"outputs/debug_pos{pi}.mp4", frames, fps=12)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    for pi, (succ, contact) in enumerate(results):
        print(f"  pos{pi}: {'SUCC' if succ else 'FAIL'} contact={contact}")
    n_succ = sum(s for s, _ in results)
    n_contact = sum(c for _, c in results)
    print(f"  SR={n_succ}/5  Contact={n_contact}/5")

    # Grid video
    from PIL import Image, ImageDraw
    maxlen = max(len(f) for f in all_frames)
    for vf in all_frames:
        while len(vf) < maxlen:
            vf.append(vf[-1])

    merged = []
    for t in range(maxlen):
        row = []
        for pi in range(len(unique)):
            f = np.array(all_frames[pi][t])
            img = Image.fromarray(f).resize((320, 240))
            draw = ImageDraw.Draw(img)
            tag = "OK" if results[pi][0] else "X"
            color = (0,255,0) if results[pi][0] else (255,80,80)
            draw.rectangle([0,0,100,16], fill=(0,0,0))
            draw.text((2,1), f"p{pi} {tag}", fill=color)
            row.append(np.array(img))
        merged.append(np.concatenate(row, axis=1))

    imageio.mimsave("outputs/debug_physics_diag.mp4", merged, fps=12, macro_block_size=1)
    print(f"\nSaved grid: {len(merged)} frames")

if __name__ == "__main__":
    main()

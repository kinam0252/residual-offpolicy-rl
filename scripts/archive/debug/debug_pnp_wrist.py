"""Render cam_wrist videos for the same episodes."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import imageio

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

with open("configs/pnp_sim33ep_positions.json") as f:
    all_positions = json.load(f)

ep_indices = [0, 10, 20, 32]
CHECKPOINT = "/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-100000"
OUT_DIR = "outputs/debug_pnp"
os.makedirs(OUT_DIR, exist_ok=True)

saved_policy = None

for ep_idx in ep_indices:
    ep = all_positions[ep_idx]
    print(f"\n=== Episode {ep_idx}: cube={ep['cube_pos']}, bowl={ep['bowl_pos']} ===")

    vec_env = MuJoCoVecEnvPnP(
        num_envs=1,
        cube_positions=[ep["cube_pos"]],
        bowl_positions=[ep["bowl_pos"]],
        max_episode_steps=300,
        reward_type="dense",
    )
    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=CHECKPOINT,
        task_description="pick up the red cube and place it in the white bowl",
    )
    if saved_policy is not None:
        wrapper.policy = saved_policy
    else:
        saved_policy = wrapper.policy

    obs = wrapper.reset()
    frames_base = []
    frames_wrist = []
    print("Running episode...")
    for step in range(300):
        action = np.zeros((1, wrapper.action_dim))
        obs, rew, term, trunc, info = wrapper.step(action)
        if step % 3 == 0:
            frames_base.append(wrapper.vec_env.get_frame(0, camera='back', size=(360, 640)))
            frames_wrist.append(wrapper.vec_env.get_frame(0, camera='wrist', size=(360, 640)))
        if step % 100 == 0:
            r = float(rew[0]) if hasattr(rew, '__getitem__') else float(rew)
            print(f"  step {step}: reward={r:.4f}")

    for name, frames in [("cambase", frames_base), ("wrist", frames_wrist)]:
        path = f"{OUT_DIR}/pnp_ep{ep_idx}_{name}_ckpt100k.mp4"
        imageio.mimsave(path, frames, fps=15, macro_block_size=1)
        print(f"Saved {path}")

print("\nAll done!")

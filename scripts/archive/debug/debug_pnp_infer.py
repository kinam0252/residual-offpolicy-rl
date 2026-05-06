"""PnP inference with 33ep positions — cam_base + cam_wrist videos."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import imageio

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

with open("configs/pnp_sim33ep_positions.json") as f:
    all_positions = json.load(f)

ep_indices = [0, 5, 10, 20, 32]
CHECKPOINT = "/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_100ep/checkpoint-75000"
OUT_DIR = "outputs/debug_pnp"
os.makedirs(OUT_DIR, exist_ok=True)

# Create first env to load GR00T once
first_ep = all_positions[ep_indices[0]]
vec_env = MuJoCoVecEnvPnP(
    num_envs=1,
    cube_positions=[first_ep["cube_pos"]],
    bowl_positions=[first_ep["bowl_pos"]],
    max_episode_steps=300,
    reward_type="dense",
)
wrapper = MuJoCoResidualWrapperPnP(
    vec_env=vec_env,
    groot_checkpoint=CHECKPOINT,
    task_description="pick up the red cube and place it in the white bowl",
)
saved_policy = wrapper.policy

for ep_idx in ep_indices:
    ep = all_positions[ep_idx]
    cube_pos = ep["cube_pos"]
    bowl_pos = ep["bowl_pos"]
    print(f"\n=== Episode {ep_idx}: cube={cube_pos}, bowl={bowl_pos} ===")

    vec_env = MuJoCoVecEnvPnP(
        num_envs=1,
        cube_positions=[cube_pos],
        bowl_positions=[bowl_pos],
        max_episode_steps=300,
        reward_type="dense",
    )
    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=CHECKPOINT,
        task_description="pick up the red cube and place it in the white bowl",
    )
    wrapper.policy = saved_policy

    obs = wrapper.reset()
    frames_base = []
    frames_wrist = []
    print("Running episode...")
    for step in range(300):
        action = np.zeros((1, wrapper.action_dim))
        obs, rew, term, trunc, info = wrapper.step(action)
        if step % 3 == 0:
            fb = wrapper.vec_env.get_frame(0, camera='back', size=(360, 640))
            fw = wrapper.vec_env.get_frame(0, camera='wrist', size=(360, 640))
            frames_base.append(fb)
            frames_wrist.append(fw)
        if step % 50 == 0:
            r = float(rew[0]) if hasattr(rew, '__getitem__') else float(rew)
            print(f"  step {step}: reward={r:.4f}")

    # Save videos
    base_path = f"{OUT_DIR}/pnp_ep{ep_idx}_cambase_66ep.mp4"
    wrist_path = f"{OUT_DIR}/pnp_ep{ep_idx}_wrist_66ep.mp4"
    imageio.mimsave(base_path, frames_base, fps=15, macro_block_size=1)
    imageio.mimsave(wrist_path, frames_wrist, fps=15, macro_block_size=1)
    print(f"Saved {len(frames_base)} frames: {base_path}, {wrist_path}")

print("\nAll done!")

"""Inference video using actual episode positions from PnP_sim_33ep dataset."""
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

# Pre-load GR00T once by creating first wrapper
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
# Save the loaded policy
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
    # Reuse pre-loaded policy to avoid reloading
    wrapper.policy = saved_policy

    obs = wrapper.reset()
    frames = []
    print("Running episode...")
    for step in range(300):
        action = np.zeros((1, wrapper.action_dim))
        obs, rew, term, trunc, info = wrapper.step(action)
        if step % 3 == 0:
            frame = wrapper.vec_env.get_frame(0, camera='back', size=(360, 640))
            frames.append(frame)
        if step % 50 == 0:
            r = float(rew[0]) if hasattr(rew, '__getitem__') else float(rew)
            print(f"  step {step}: reward={r:.4f}")

    out_path = f"{OUT_DIR}/pnp_ep{ep_idx}_ckpt100k.mp4"
    imageio.mimsave(out_path, frames, fps=15, macro_block_size=1)
    print(f"Saved {len(frames)} frames to {out_path}")

print("\nAll done!")

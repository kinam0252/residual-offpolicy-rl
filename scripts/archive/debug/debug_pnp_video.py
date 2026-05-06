"""Render PnP env matching data episode 10 at 640x360 15fps."""
import os, numpy as np, torch

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

def main():
    # Episode 10 positions from dataset
    cube_pos = [0.3909, 0.1579, 0.02]
    bowl_pos = [0.3891, -0.1731, 0.0]

    print(f"Cube: {cube_pos}, Bowl: {bowl_pos}")
    env = MuJoCoVecEnvPnP(
        num_envs=1,
        scene_xml='/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml',
        cube_positions=[cube_pos],
        bowl_positions=[bowl_pos],
        max_episode_steps=500,
        reward_type='dense',
        device='cpu',
    )

    print("Loading GR00T...")
    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=env,
        groot_checkpoint='/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-75000',
        embodiment_tag='NEW_EMBODIMENT',
        policy_device='cuda:0',
        task_description='Pick up the red cube and place it onto the plate.',
    )

    obs, _ = wrapper.reset()
    frames = []

    print("Running episode...")
    for step in range(300):
        zero_residual = torch.zeros((1, 7))
        obs, rew, done, trunc, info = wrapper.step(zero_residual)

        # cam_base at 640x360 — note: get_frame(env_id, camera, size=(W, H))
        frame = env.get_frame(0, camera='back', size=(360, 640))
        # Only keep every 1/1 frame at ~50Hz sim, subsample to 15fps (every 3rd)
        if step % 3 == 0:
            frames.append(frame)

        if step % 100 == 0:
            eef = obs['observation.state'][0, :3].numpy()
            obj = obs['observation.object_state'][0, :3].numpy()
            print(f"  step {step}: eef={eef} cube={obj}")

        if done[0] or trunc[0]:
            break

    out_dir = '/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl/outputs/debug_pnp'
    os.makedirs(out_dir, exist_ok=True)

    import imageio
    print(f"Frame shape: {frames[0].shape}, num frames: {len(frames)}")
    imageio.mimsave(f'{out_dir}/pnp_env_ep10_cambase.mp4', frames, fps=15, macro_block_size=1)
    print(f"Saved to {out_dir}/pnp_env_ep10_cambase.mp4")

if __name__ == '__main__':
    main()

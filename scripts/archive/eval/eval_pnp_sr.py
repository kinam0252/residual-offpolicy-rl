"""Evaluate PnP checkpoint: 5 episodes, report SR + save videos."""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import imageio

from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import MuJoCoVecEnvPnP
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper_pnp import MuJoCoResidualWrapperPnP

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", default="")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--out_dir", default="outputs/debug_pnp")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

with open("configs/pnp_sim33ep_positions.json") as f:
    all_positions = json.load(f)

ep_indices = [0, 5, 10, 20, 32][:args.episodes]
SUCCESS_THRESHOLD = 0.05

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
    groot_checkpoint=args.checkpoint,
    task_description="pick up the red cube and place it in the white bowl",
)
saved_policy = wrapper.policy

successes = []
for ep_idx in ep_indices:
    ep = all_positions[ep_idx]
    cube_pos = np.array(ep["cube_pos"])
    bowl_pos = np.array(ep["bowl_pos"])

    vec_env = MuJoCoVecEnvPnP(
        num_envs=1,
        cube_positions=[cube_pos.tolist()],
        bowl_positions=[bowl_pos.tolist()],
        max_episode_steps=300,
        reward_type="dense",
    )
    wrapper = MuJoCoResidualWrapperPnP(
        vec_env=vec_env,
        groot_checkpoint=args.checkpoint,
        task_description="pick up the red cube and place it in the white bowl",
    )
    wrapper.policy = saved_policy

    obs = wrapper.reset()
    max_reward = 0.0
    final_reward = 0.0
    frames_base = []
    frames_wrist = []
    ever_success = False
    min_dist = 999.0
    for step in range(300):
        action = np.zeros((1, wrapper.action_dim))
        obs, rew, term, trunc, info = wrapper.step(action)
        r = float(rew[0]) if hasattr(rew, '__getitem__') else float(rew)
        max_reward = max(max_reward, r)
        final_reward = r
        if step % 3 == 0:
            fb = wrapper.vec_env.get_frame(0, camera='back', size=(360, 640))
            fw = wrapper.vec_env.get_frame(0, camera='wrist', size=(360, 640))
            frames_base.append(fb)
            frames_wrist.append(fw)

        # Check success at every step: cube inside bowl radius
        env_chk = wrapper.vec_env._envs[0]
        data_chk = env_chk["data"]
        cqa = env_chk["cube_qposadr"]
        if cqa is not None:
            cp = data_chk.qpos[cqa:cqa + 3].copy()
            d = np.linalg.norm(cp[:2] - bowl_pos[:2])
            min_dist = min(min_dist, d)
            if d < 0.095 and cp[2] < 0.08:  # inside bowl radius, near table
                ever_success = True

    successes.append(ever_success)
    tag = "OK" if ever_success else "FAIL"
    print(f"  ep{ep_idx}: max_r={max_reward:.2f}, final_r={final_reward:.2f}, "
          f"min_dist={min_dist:.3f}m, {tag}")

    # Save videos
    label = args.label.replace(" ", "_")
    imageio.mimsave(f"{args.out_dir}/{label}_ep{ep_idx}_base.mp4",
                    frames_base, fps=15, macro_block_size=1)
    imageio.mimsave(f"{args.out_dir}/{label}_ep{ep_idx}_wrist.mp4",
                    frames_wrist, fps=15, macro_block_size=1)

sr = sum(successes) / len(successes)
print(f"\n[{args.label}] Success Rate: {sum(successes)}/{len(successes)} = {sr:.0%}")

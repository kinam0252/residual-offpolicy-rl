"""Evaluate checkpoint on random cube placements with video recording."""
import argparse, json, sys, os, time, traceback, importlib, types as _types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ── Deepspeed mock ──
if "deepspeed" not in sys.modules:
    _ds_mock = _types.ModuleType("deepspeed")
    _ds_mock.__version__ = "0.0.0"
    _ds_mock.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    _ds_mock.__path__ = []
    _ds_mock.__file__ = __file__
    _ds_zero = _types.ModuleType("deepspeed.zero")
    _ds_zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", None)
    _ds_zero.Init = lambda *a, **kw: (lambda f: f)
    _ds_mock.zero = _ds_zero
    sys.modules["deepspeed"] = _ds_mock
    sys.modules["deepspeed.zero"] = _ds_zero

try:
    import huggingface_hub.utils._validators as _hf_val
    _orig_validate = _hf_val.validate_repo_id
    def _patched_validate_repo_id(repo_id: str) -> None:
        if repo_id and (repo_id.startswith("/") or repo_id.startswith(".")):
            return
        return _orig_validate(repo_id)
    _hf_val.validate_repo_id = _patched_validate_repo_id
except Exception:
    pass

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
torch.backends.cuda.enable_cudnn_sdp(False)

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from resfit.rl_finetuning.config.residual_td3_mujoco import ResidualTD3MuJoCoConfig
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv


def evaluate_with_video(env, agent, num_episodes, device, image_keys, video_dir, random_range_label=""):
    """Run eval episodes, save video for each env 0 episode, report per-episode results."""
    agent.eval()
    num_envs = env.vec_env.num_envs
    results = []
    all_successes = 0
    total_episodes = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_returns = [0.0] * num_envs
        env_done = [False] * num_envs
        env_success = [False] * num_envs
        frames = []

        for step in range(env.vec_env.max_episode_steps):
            # Capture frame from env 0
            frame = env.vec_env.get_frame(0, camera="front", size=(480, 640))
            frames.append(frame)

            with torch.no_grad():
                action = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            for i in range(num_envs):
                if not env_done[i]:
                    ep_returns[i] += reward[i].item()
                    if done[i]:
                        env_done[i] = True
                        if terminated[i]:
                            env_success[i] = True
            if all(env_done):
                break

        # Save video
        if frames and imageio is not None and video_dir:
            vid_path = video_dir / f"eval_{random_range_label}_ep{ep:02d}.mp4"
            imageio.mimwrite(str(vid_path), frames, fps=30)
            print(f"  Video saved: {vid_path} ({len(frames)} frames)", flush=True)

        ep_sr = sum(1 for s in env_success if s) / num_envs
        all_successes += sum(1 for s in env_success if s)
        total_episodes += num_envs
        results.append({"episode": ep, "sr": ep_sr, "env_success": env_success})
        print(f"  Episode {ep}: SR={ep_sr:.0%} success={env_success}", flush=True)

    overall_sr = all_successes / total_episodes if total_episodes > 0 else 0
    return overall_sr, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scene_xml", type=str, default=None)
    p.add_argument("--groot_checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=10)
    p.add_argument("--num_episodes", type=int, default=3)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--reward_type", type=str, default="dense")
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--actor_hidden_dim", type=int, default=512)
    p.add_argument("--critic_hidden_dim", type=int, default=1024)
    p.add_argument("--asymmetric_critic", action="store_true")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--video_dir", type=str, default="outputs/eval_videos")
    p.add_argument("--random_cube_range", type=str, required=True,
                   help='JSON: {"dx":[-8,14],"dy":[-15,15],"yaw":[-10,10]} in cm/deg')
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Parse random range
    rr = json.loads(args.random_cube_range)
    random_cube_range = {}
    for k, v in rr.items():
        if k == "yaw":
            random_cube_range[k] = [v[0], v[1]]
        else:
            random_cube_range[k] = [v[0] / 100.0, v[1] / 100.0]
    print(f"Random cube range (meters): {random_cube_range}", flush=True)

    print("Creating MuJoCo env...", flush=True)
    mujoco_env = MuJoCoVecEnv(
        num_envs=args.num_envs,
        random_cube_range=random_cube_range,
        scene_xml=args.scene_xml,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        device=args.device,
    )

    print("Creating residual wrapper with GR00T...", flush=True)
    env = MuJoCoResidualWrapper(
        vec_env=mujoco_env,
        groot_checkpoint=args.groot_checkpoint,
        embodiment_tag="NEW_EMBODIMENT",
        policy_device=args.device,
        task_description="lift the cube",
        open_loop_horizon=16,
        residual_pos_scale=0.02,
        residual_rot_scale=0.05,
        residual_grip_scale=0.1,
    )

    print("Initial reset...", flush=True)
    obs, _ = env.reset()

    image_keys = ["observation.depth.front", "observation.depth.wrist"]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    action_dim = env.action_dim
    object_state_dim = 7

    cfg = ResidualTD3MuJoCoConfig()
    cfg.agent.actor.action_scale = args.action_scale
    cfg.agent.actor.hidden_dim = args.actor_hidden_dim
    cfg.agent.critic.hidden_dim = args.critic_hidden_dim

    agent = QAgent(
        obs_shape=(1, 84, 84),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
        object_state_dim=object_state_dim,
        asymmetric_critic=True,
    )
    agent.to(device)

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "actor" in ckpt and isinstance(ckpt["actor"], dict):
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        if "encoders" in ckpt:
            agent.encoders.load_state_dict(ckpt["encoders"])
        print("Loaded structured checkpoint", flush=True)
    else:
        agent.load_state_dict(ckpt, strict=False)
        print("Loaded flat state_dict", flush=True)

    print(f"\nEvaluating: {args.num_episodes} episodes x {args.num_envs} envs (random normal range)", flush=True)
    t0 = time.time()
    overall_sr, results = evaluate_with_video(
        env=env, agent=agent, num_episodes=args.num_episodes,
        device=device, image_keys=image_keys,
        video_dir=video_dir, random_range_label="normal"
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL SR: {overall_sr:.1%} ({args.num_episodes} eps x {args.num_envs} envs = {args.num_episodes * args.num_envs} total)", flush=True)
    print(f"Time: {elapsed:.1f}s", flush=True)

    # Save results
    result_path = video_dir / "eval_results.json"
    with open(str(result_path), "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "overall_sr": overall_sr,
            "num_episodes": args.num_episodes,
            "num_envs": args.num_envs,
            "random_cube_range": args.random_cube_range,
            "time": elapsed,
            "episodes": [{"ep": r["episode"], "sr": r["sr"], "env_success": r["env_success"]} for r in results],
        }, f, indent=2)
    print(f"Results saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()

"""
Single-run eval: 10 envs with specified checkpoint and offset type.
Info bar ABOVE frame. Launched 4 times in parallel for 4-way eval.

Usage:
  isaaclab.sh -p eval_single_run.py --headless --enable_cameras \
    --checkpoint /path/to/agent_step0.pt \
    --env_type train \
    --run_label step0_train \
    --output_dir /path/to/output/step0_train \
    --csv_base_dir ... --groot_model_path ...
"""
from __future__ import annotations
import argparse, os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--env_type", type=str, required=True, choices=["train", "unseen"])
parser.add_argument("--run_label", type=str, required=True)
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.03)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--phase_probe_path", type=str, default=None, help="Path to frozen phase probe .pt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import faulthandler
try: faulthandler.cancel_dump_traceback_later()
except: pass

from pathlib import Path
from datetime import datetime
import numpy as np, torch, imageio, cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from resfit.rl_finetuning.wrappers.iface_env_wrapper import IfaceEnvWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.config.rlpd import ActorConfig, QAgentConfig
import isaaclab.sim as sim_utils

def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

TRAIN_OFFSETS = [
    (0.000, 0.000, "center"),       (0.050, 0.000, "right_5cm"),
    (-0.050, 0.000, "left_5cm"),    (0.000, 0.050, "fwd_5cm"),
    (0.000, -0.050, "back_5cm"),    (0.040, 0.040, "FR_5.6cm"),
    (-0.040, 0.040, "FL_5.6cm"),    (0.040, -0.040, "BR_5.6cm"),
    (-0.040, -0.040, "BL_5.6cm"),   (0.080, 0.000, "right_8cm"),
]
UNSEEN_OFFSETS = [
    (0.025, 0.020, "ctr+sh"),       (0.070, 0.025, "R5+sh"),
    (-0.070, -0.020, "L5+sh"),      (0.025, 0.070, "F5+sh"),
    (-0.020, -0.070, "B5+sh"),      (0.060, 0.060, "FR+sh"),
    (-0.060, 0.060, "FL+sh"),       (0.060, -0.060, "BR+sh"),
    (-0.060, -0.060, "BL+sh"),      (0.100, 0.025, "R10+sh"),
]

def make_info_bar(w, eid, ename, step, cf, hc, ch, rw, succ, ckpt, etype):
    bar_h = 70
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    f = cv2.FONT_HERSHEY_SIMPLEX
    s, t = 0.35, 1
    fc = (0,255,0) if hc > 0.5 else (100,100,100)
    hc2 = (0,255,0) if ch > 0.005 else (150,150,150)
    cv2.putText(bar, f"E{eid}:{ename}", (3,12), f, s, (255,255,255), t)
    cv2.putText(bar, f"{ckpt}[{etype}]", (w-130,12), f, 0.30, (180,180,180), t)
    cv2.putText(bar, f"S:{step}", (3,25), f, s, (200,200,200), t)
    cv2.putText(bar, f"F:{cf:.1f}N", (60,25), f, s, fc, t)
    cv2.putText(bar, f"{'ON' if hc>0.5 else '  '}", (145,25), f, s, fc, t)
    cv2.putText(bar, f"H:{ch*100:.1f}cm", (3,38), f, s, hc2, t)
    cv2.putText(bar, f"R:{rw:.3f}", (90,38), f, s, (200,200,200), t)
    if succ:
        cv2.putText(bar, "OK", (180,38), f, s, (0,255,0), 2)
    bw = w - 6
    ff = int(min(cf/5,1)*bw)
    cv2.rectangle(bar, (3,44), (3+bw,48), (50,50,50), -1)
    if ff > 0: cv2.rectangle(bar, (3,44), (3+ff,48), fc, -1)
    hf = int(min(max(ch,0)/0.05,1)*bw)
    cv2.rectangle(bar, (3,52), (3+bw,56), (50,50,50), -1)
    if hf > 0: cv2.rectangle(bar, (3,52), (3+hf,56), hc2, -1)
    return bar

def main():
    N = 10
    device = torch.device("cuda:0")
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = TRAIN_OFFSETS if args_cli.env_type == "train" else UNSEEN_OFFSETS
    run_label = args_cli.run_label

    _log(f"[{run_label}] Creating env with {N} envs...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    _sim = sim_utils.SimulationContext(sim_cfg)
    env = IfaceEnvWrapper(
        sim=_sim, csv_dir=args_cli.csv_base_dir,
        groot_model_path=args_cli.groot_model_path,
        embodiment_tag=args_cli.groot_embodiment_tag,
        policy_device=args_cli.groot_policy_device,
        policy_strict=True, task_description="Pick up the white object.",
        language_override=args_cli.language_override,
        max_episode_steps=args_cli.max_episode_steps,
        num_envs=N, success_threshold=args_cli.success_threshold,
    )
    _log(f"[{run_label}] Env ready")

    image_keys = ["observation.images.front", "observation.images.back", "observation.images.wrist"]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]

    vlm_latent_dim = env.observation_space["observation.vlm_latent"].shape[1] if "observation.vlm_latent" in env.observation_space.spaces else 0
    agent_cfg = QAgentConfig(
        actor_lr=3e-7, critic_lr=1e-4, critic_target_tau=0.005,
        clip_q_target_to_reward_range=True,
        actor=ActorConfig(action_scale=0.1, actor_last_layer_init_scale=0.0, action_l2_reg_weight=10.0),
    )
    use_phase_probe = bool(getattr(args_cli, 'phase_probe_path', None))
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w), prop_shape=(lowdim_dim,),
        action_dim=action_dim, rl_cameras=image_keys, cfg=agent_cfg, residual_actor=True,
        vlm_latent_dim=vlm_latent_dim,
        phase_probe_mode=use_phase_probe,
    )
    agent.to(device)
    # Load frozen phase probe if provided
    if getattr(args_cli, 'phase_probe_path', None) and agent.vlm_projector is not None:
        agent.load_phase_probe(args_cli.phase_probe_path)
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    agent.load_checkpoint_compat(ckpt)
    agent.eval()
    _log(f"[{run_label}] Loaded: {args_cli.checkpoint}")

    # Reset + perturb
    env.reset()
    sim_dt = env.sim_dt
    for eid, (dx, dy, name) in enumerate(offsets):
        cube_pos = env.cube.data.root_state_w[eid, :3].clone()
        cube_pos[0] += dx; cube_pos[1] += dy
        cube_quat = env.cube.data.root_state_w[eid, 3:7].clone()
        new_pose = torch.cat([cube_pos, cube_quat]).unsqueeze(0)
        eid_t = torch.tensor([eid], device=device, dtype=torch.long)
        env.cube.write_root_pose_to_sim(new_pose, env_ids=eid_t)
        env.cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=eid_t)

    for _ in range(50):
        env.scene.write_data_to_sim(); env.sim.step(); env.scene.update(sim_dt)
    env._initial_cube_z = env.cube.data.root_state_w[:N, 2].clone()
    obs = env._build_obs()

    # Run
    max_steps = args_cli.max_episode_steps
    ep_rewards = torch.zeros(N, device=device)
    ep_success = torch.zeros(N, device=device)
    FRAME_SIZE = (192, 256)
    frames_per_env = [[] for _ in range(N)]
    ckpt_label = Path(args_cli.checkpoint).stem.replace("agent_", "")

    _log(f"[{run_label}] Running {max_steps} steps...")
    t0 = time.time()
    for step in range(max_steps):
        with torch.no_grad(), utils.eval_mode(agent):
            res = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
        next_obs, reward, terminated, truncated, info = env.step(res)
        ep_rewards += reward[:N].to(device)
        cube_z = env.cube.data.root_state_w[:N, 2].to(device)
        cube_height = cube_z - env._initial_cube_z[:N].to(device)
        debug = getattr(env, "_last_step_debug", None)
        # Success requires BOTH height threshold AND contact force (matching training)
        has_contact = torch.zeros(N, device=device)
        if debug is not None and "has_contact" in debug:
            has_contact = debug["has_contact"].to(device)
        cube_lifted = ((cube_height >= args_cli.success_threshold) & (has_contact > 0.5)).float()
        ep_success = torch.max(ep_success, cube_lifted)

        if step % 5 == 0 and hasattr(env, "get_frame"):
            for eid in range(N):
                frame = env.get_frame(eid, camera="front", size=FRAME_SIZE)
                cf = debug["contact_force"][eid].item() if debug else 0.0
                hc = debug["has_contact"][eid].item() if debug else 0.0
                ch = debug["cube_height"][eid].item() if debug else 0.0
                rw = debug["reward"][eid].item() if debug else 0.0
                succ = bool(ep_success[eid].item() > 0)
                bar = make_info_bar(FRAME_SIZE[1], eid, offsets[eid][2], step, cf, hc, ch, rw, succ, ckpt_label, args_cli.env_type)
                frames_per_env[eid].append(np.concatenate([bar, frame], axis=0))

        obs = next_obs
        if (terminated | truncated)[:N].all(): break

    _log(f"[{run_label}] Done in {time.time()-t0:.0f}s")

    # Save
    results = []
    for eid in range(N):
        dx, dy, name = offsets[eid]
        succ = bool(ep_success[eid].item() > 0)
        ret = float(ep_rewards[eid].item())
        results.append({"env": eid, "name": name, "success": succ, "return": round(ret, 2)})
        _log(f"  E{eid} [{name:12s}] succ={succ} R={ret:.1f}")
        if frames_per_env[eid]:
            tag = "success" if succ else "fail"
            w = imageio.get_writer(str(output_dir / f"env{eid}_{name}_{tag}.mp4"), fps=20)
            for fr in frames_per_env[eid]: w.append_data(fr)
            w.close()

    # Grid
    valid = [f for f in frames_per_env if f]
    if valid:
        ml = min(len(f) for f in valid)
        w = imageio.get_writer(str(output_dir / f"grid_{run_label}.mp4"), fps=20)
        for fi in range(ml):
            rows = []
            for r in range(2):
                row = [valid[r*5+c][fi] if r*5+c < len(valid) else np.zeros_like(valid[0][fi]) for c in range(5)]
                rows.append(np.concatenate(row, axis=1))
            w.append_data(np.concatenate(rows, axis=0))
        w.close()

    succ_rate = sum(1 for r in results if r["success"]) / N
    summary = {"run_label": run_label, "ckpt": ckpt_label, "env_type": args_cli.env_type, "success_rate": succ_rate, "per_env": results}
    with open(output_dir / "results.json", "w") as f2:
        json.dump(summary, f2, indent=2)
    _log(f"[{run_label}] => {succ_rate*100:.0f}% success")
    env.close()
    os._exit(0)

if __name__ == "__main__":
    main()

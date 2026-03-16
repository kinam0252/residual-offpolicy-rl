"""
4-way eval: single Isaac Sim with 40 envs (4 runs × 10 envs each).
Run 0-9: step0 × train, 10-19: step2500 × train, 20-29: step0 × unseen, 30-39: step2500 × unseen.
Info bar ABOVE frame (concatenated).

Usage:
  isaaclab.sh -p eval_4way_parallel.py --headless --enable_cameras \
    --ckpt_dir /path/to --output_dir /path/to ...
"""
from __future__ import annotations
import argparse, os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--csv_base_dir", type=str, required=True)
parser.add_argument("--groot_model_path", type=str, required=True)
parser.add_argument("--groot_embodiment_tag", type=str, default="new_embodiment")
parser.add_argument("--groot_policy_device", type=str, default="cuda:0")
parser.add_argument("--language_override", type=str, default="pick up mushroom")
parser.add_argument("--max_episode_steps", type=int, default=1000)
parser.add_argument("--success_threshold", type=float, default=0.03)
parser.add_argument("--output_dir", type=str, default="/tmp/eval_4way")
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
import numpy as np, torch, imageio, cv2, copy

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

# 4 runs config
RUNS = [
    {"ckpt": "agent_step0.pt",    "ckpt_label": "step0",    "env_type": "train",  "offsets": TRAIN_OFFSETS},
    {"ckpt": "agent_step2500.pt", "ckpt_label": "step2500", "env_type": "train",  "offsets": TRAIN_OFFSETS},
    {"ckpt": "agent_step0.pt",    "ckpt_label": "step0",    "env_type": "unseen", "offsets": UNSEEN_OFFSETS},
    {"ckpt": "agent_step2500.pt", "ckpt_label": "step2500", "env_type": "unseen", "offsets": UNSEEN_OFFSETS},
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
    device = torch.device("cuda:0")
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args_cli.ckpt_dir)
    ENVS_PER_RUN = 10
    N_RUNS = len(RUNS)
    TOTAL_ENVS = ENVS_PER_RUN * N_RUNS  # 40

    _log(f"Creating env with {TOTAL_ENVS} envs (4 runs × 10)...")
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    _sim = sim_utils.SimulationContext(sim_cfg)

    env = IfaceEnvWrapper(
        sim=_sim,
        csv_dir=args_cli.csv_base_dir,
        groot_model_path=args_cli.groot_model_path,
        embodiment_tag=args_cli.groot_embodiment_tag,
        policy_device=args_cli.groot_policy_device,
        policy_strict=True,
        task_description="Pick up the white object.",
        language_override=args_cli.language_override,
        max_episode_steps=args_cli.max_episode_steps,
        num_envs=TOTAL_ENVS,
        success_threshold=args_cli.success_threshold,
    )
    _log("Env ready")

    image_keys = ["observation.images.front", "observation.images.back", "observation.images.wrist"]
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]

    # Build 4 agents (2 unique checkpoints, each loaded twice for different env groups)
    agents = {}
    for run in RUNS:
        ckpt_label = run["ckpt_label"]
        if ckpt_label not in agents:
            agent_cfg = QAgentConfig(
                actor_lr=3e-7, critic_lr=1e-4, critic_target_tau=0.005,
                clip_q_target_to_reward_range=True,
                actor=ActorConfig(action_scale=0.1, actor_last_layer_init_scale=0.0, action_l2_reg_weight=10.0),
            )
            agent = QAgent(
                obs_shape=(img_c, img_h, img_w), prop_shape=(lowdim_dim,),
                action_dim=action_dim, rl_cameras=image_keys, cfg=agent_cfg, residual_actor=True,
            )
            agent.to(device)
            ckpt = torch.load(str(ckpt_dir / run["ckpt"]), map_location=device)
            agent.load_checkpoint_compat(ckpt)
            agent.eval()
            agents[ckpt_label] = agent
            _log(f"Loaded agent: {ckpt_label}")

    # Reset all envs and apply perturbations
    env.reset()
    sim_dt = env.sim_dt
    for run_idx, run in enumerate(RUNS):
        offsets = run["offsets"]
        for local_eid, (dx, dy, name) in enumerate(offsets):
            global_eid = run_idx * ENVS_PER_RUN + local_eid
            cube_pos = env.cube.data.root_state_w[global_eid, :3].clone()
            cube_pos[0] += dx
            cube_pos[1] += dy
            cube_quat = env.cube.data.root_state_w[global_eid, 3:7].clone()
            new_pose = torch.cat([cube_pos, cube_quat]).unsqueeze(0)
            eid_t = torch.tensor([global_eid], device=device, dtype=torch.long)
            env.cube.write_root_pose_to_sim(new_pose, env_ids=eid_t)
            env.cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=eid_t)

    for _ in range(50):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(sim_dt)

    env._initial_cube_z = env.cube.data.root_state_w[:TOTAL_ENVS, 2].clone()
    obs = env._build_obs()

    # Eval loop — all 40 envs step together, but each group uses different agent
    max_steps = args_cli.max_episode_steps
    success_threshold = args_cli.success_threshold
    ep_rewards = torch.zeros(TOTAL_ENVS, device=device)
    ep_success = torch.zeros(TOTAL_ENVS, device=device)
    FRAME_SIZE = (192, 256)
    frames_per_env = [[] for _ in range(TOTAL_ENVS)]

    _log(f"Running 4-way eval ({TOTAL_ENVS} envs, {max_steps} steps)...")
    t0 = time.time()

    for step in range(max_steps):
        # Build per-run residual actions using appropriate agents
        residual_all = torch.zeros(TOTAL_ENVS, action_dim, device=device)
        with torch.no_grad():
            for run_idx, run in enumerate(RUNS):
                agent = agents[run["ckpt_label"]]
                start = run_idx * ENVS_PER_RUN
                end = start + ENVS_PER_RUN
                obs_slice = {k: v[start:end] for k, v in obs.items()}
                with utils.eval_mode(agent):
                    res = agent.act(obs_slice, eval_mode=True, stddev=0.0, cpu=False)
                residual_all[start:end] = res

        next_obs, reward, terminated, truncated, info = env.step(residual_all)
        ep_rewards += reward[:TOTAL_ENVS].to(device)

        cube_z = env.cube.data.root_state_w[:TOTAL_ENVS, 2].to(device)
        cube_init_z = env._initial_cube_z[:TOTAL_ENVS].to(device)
        cube_lifted = ((cube_z - cube_init_z) >= success_threshold).float()
        ep_success = torch.max(ep_success, cube_lifted)

        # Capture frames every 5 steps
        if step % 5 == 0 and hasattr(env, "get_frame"):
            debug = getattr(env, "_last_step_debug", None)
            for global_eid in range(TOTAL_ENVS):
                run_idx = global_eid // ENVS_PER_RUN
                local_eid = global_eid % ENVS_PER_RUN
                run = RUNS[run_idx]
                offsets = run["offsets"]

                frame = env.get_frame(global_eid, camera="front", size=FRAME_SIZE)
                cf = debug["contact_force"][global_eid].item() if debug else 0.0
                hc = debug["has_contact"][global_eid].item() if debug else 0.0
                ch = debug["cube_height"][global_eid].item() if debug else 0.0
                rw = debug["reward"][global_eid].item() if debug else 0.0
                succ = bool(ep_success[global_eid].item() > 0)

                info_bar = make_info_bar(
                    FRAME_SIZE[1], local_eid, offsets[local_eid][2], step,
                    cf, hc, ch, rw, succ, run["ckpt_label"], run["env_type"],
                )
                combined = np.concatenate([info_bar, frame], axis=0)
                frames_per_env[global_eid].append(combined)

        obs = next_obs
        done = terminated | truncated
        if done[:TOTAL_ENVS].all():
            break

    eval_time = time.time() - t0
    _log(f"All done in {eval_time:.0f}s, {step+1} steps")

    # Save per-run results and videos
    all_summaries = []
    for run_idx, run in enumerate(RUNS):
        run_name = f"{run['ckpt_label']}_{run['env_type']}"
        vid_dir = output_dir / run_name
        vid_dir.mkdir(parents=True, exist_ok=True)
        offsets = run["offsets"]
        results = []

        for local_eid in range(ENVS_PER_RUN):
            global_eid = run_idx * ENVS_PER_RUN + local_eid
            dx, dy, name = offsets[local_eid]
            succ = bool(ep_success[global_eid].item() > 0)
            ret = float(ep_rewards[global_eid].item())
            results.append({"env": local_eid, "name": name, "success": succ, "return": round(ret, 2)})
            _log(f"  {run_name} E{local_eid} [{name:12s}] succ={succ} R={ret:.1f}")

            if frames_per_env[global_eid]:
                vpath = vid_dir / f"env{local_eid}_{name}.mp4"
                w = imageio.get_writer(str(vpath), fps=20)
                for fr in frames_per_env[global_eid]:
                    w.append_data(fr)
                w.close()

        # Grid video (2×5)
        valid = [frames_per_env[run_idx * ENVS_PER_RUN + i] for i in range(ENVS_PER_RUN) if frames_per_env[run_idx * ENVS_PER_RUN + i]]
        if valid:
            min_len = min(len(f) for f in valid)
            grid_path = vid_dir / f"grid_{run_name}.mp4"
            w = imageio.get_writer(str(grid_path), fps=20)
            for fi in range(min_len):
                rows_list = []
                for r in range(2):
                    row = []
                    for c in range(5):
                        idx = r * 5 + c
                        row.append(valid[idx][fi] if idx < len(valid) else np.zeros_like(valid[0][fi]))
                    rows_list.append(np.concatenate(row, axis=1))
                grid = np.concatenate(rows_list, axis=0)
                w.append_data(grid)
            w.close()
            _log(f"  Grid: {grid_path}")

        succ_rate = sum(1 for r in results if r["success"]) / ENVS_PER_RUN
        summary = {"ckpt": run["ckpt_label"], "env_type": run["env_type"], "success_rate": succ_rate, "per_env": results}
        all_summaries.append(summary)
        with open(vid_dir / "results.json", "w") as f2:
            json.dump(summary, f2, indent=2)

    # Final summary
    _log("\n" + "=" * 60)
    _log("4-WAY EVAL SUMMARY:")
    for s in all_summaries:
        _log(f"  {s['ckpt']:8s} × {s['env_type']:6s} => {s['success_rate']*100:.0f}%")
    _log("=" * 60)
    with open(output_dir / "summary_4way.json", "w") as f2:
        json.dump(all_summaries, f2, indent=2)

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()

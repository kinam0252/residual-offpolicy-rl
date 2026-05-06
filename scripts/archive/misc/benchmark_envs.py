#!/usr/bin/env python3
"""Benchmark GPU/CPU/RAM usage per num_envs."""
import subprocess, time, os, sys, gc, json
import torch
import numpy as np

def get_gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True
        ).strip().split(",")
        return {"gpu_util": int(out[0]), "mem_used_MB": int(out[1]), "mem_total_MB": int(out[2])}
    except:
        return {}

def get_ram_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1e6

print("=" * 60)
print("ENV SCALING BENCHMARK")
print("=" * 60)

sys.path.insert(0, "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl")
from resfit.rl_finetuning.wrappers.mujoco_vec_env import MuJoCoVecEnv
from resfit.rl_finetuning.wrappers.mujoco_residual_wrapper import MuJoCoResidualWrapper

GROOT_CKPT = "/home/nas_main/kinamkim/DATA/INTERN/training/gr00t_sim_32ep/checkpoint-300000"
SCENE_XML = "/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"

results = []

for n_envs in [1, 5, 10, 20, 30, 40, 50]:
    print(f"\n{'='*40}")
    print(f"Testing num_envs = {n_envs}")
    print(f"{'='*40}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    gpu_before = get_gpu_util()
    ram_before = get_ram_mb()

    try:
        t0 = time.time()
        env = MuJoCoVecEnv(
            scene_xml=SCENE_XML,
            num_envs=n_envs,
            cube_positions=[[0.45, -0.05, 0.02]],
            max_episode_steps=500,
            random_cube_range={"dx": [-3, 11], "dy": [-10, 10], "yaw": [0, 0]},
        )

        wrapper = MuJoCoResidualWrapper(
            vec_env=env,
            groot_checkpoint=GROOT_CKPT,
        )
        t_create = time.time() - t0

        gpu_after_create = get_gpu_util()
        ram_after_create = get_ram_mb()

        obs = wrapper.reset()
        t0 = time.time()
        n_steps = 20
        for _ in range(n_steps):
            action = np.random.randn(n_envs, 7).astype(np.float32) * 0.01
            obs, reward, done, truncated, info = wrapper.step(action)
        t_steps = time.time() - t0

        gpu_after = get_gpu_util()
        ram_after = get_ram_mb()
        _, peak_torch = torch.cuda.memory_allocated() / 1e6, torch.cuda.max_memory_allocated() / 1e6

        result = {
            "num_envs": n_envs,
            "create_s": round(t_create, 1),
            "step_ms": round(t_steps / n_steps * 1000, 1),
            "step_per_s": round(n_steps / t_steps, 2),
            "gpu_mem_MB": gpu_after.get("mem_used_MB", 0),
            "gpu_util": gpu_after.get("gpu_util", 0),
            "torch_peak_MB": round(peak_torch, 0),
            "ram_MB": round(ram_after, 0),
            "ram_delta_MB": round(ram_after - ram_before, 0),
        }
        results.append(result)

        print(f"  Create: {result['create_s']}s")
        print(f"  Step: {result['step_ms']}ms/step, {result['step_per_s']} step/s")
        print(f"  GPU: {result['gpu_mem_MB']}MB, util: {result['gpu_util']}%")
        print(f"  RAM: {result['ram_MB']}MB (+{result['ram_delta_MB']}MB)")

        del wrapper, env
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {e}")
        results.append({"num_envs": n_envs, "error": str(e)})
        torch.cuda.empty_cache()
        gc.collect()
        break

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'envs':>5} | {'step_ms':>8} | {'step/s':>7} | {'GPU_MB':>7} | {'GPU%':>5} | {'RAM_MB':>8} | {'RAM+':>7}")
print("-" * 65)
for r in results:
    if "error" in r:
        print(f"{r['num_envs']:>5} | FAILED: {r['error'][:45]}")
    else:
        print(f"{r['num_envs']:>5} | {r['step_ms']:>7.1f} | {r['step_per_s']:>6.2f} | {r['gpu_mem_MB']:>7} | {r['gpu_util']:>4}% | {r['ram_MB']:>7.0f} | {r['ram_delta_MB']:>+6.0f}")

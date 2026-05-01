#!/usr/bin/env python3
"""
Analyze trajectory data collected by rollout_trajectory_analysis.py.

Computes:
  1. Success Rate (SR) comparison
  2. Action smoothness (jerk, acceleration) per axis
  3. Residual magnitude over time
  4. Episode length for successful episodes
  5. Action variance (consistency)

Usage:
  python analyze_trajectories.py --data_dir outputs/trajectory_analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def load_data(data_dir: str, task: str, mode: str) -> dict:
    """Load .npz file for a given task/mode."""
    path = Path(data_dir) / f"{task}_{mode}.npz"
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return None
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def compute_smoothness(actions_list: np.ndarray) -> dict:
    """Compute per-axis smoothness metrics across episodes.

    Args:
        actions_list: array of arrays, each (T, 7)

    Returns:
        dict with jerk_mean, accel_mean, jerk_per_axis, accel_per_axis
    """
    all_jerk = []
    all_accel = []
    per_axis_jerk = [[] for _ in range(7)]
    per_axis_accel = [[] for _ in range(7)]

    for ep_actions in actions_list:
        if len(ep_actions) < 3:
            continue
        a = np.array(ep_actions)  # (T, 7)
        # 1st difference (acceleration)
        diff1 = np.diff(a, axis=0)  # (T-1, 7)
        accel = np.abs(diff1)
        # 2nd difference (jerk)
        diff2 = np.diff(diff1, axis=0)  # (T-2, 7)
        jerk = np.abs(diff2)

        all_accel.append(accel.mean())
        all_jerk.append(jerk.mean())

        for d in range(7):
            per_axis_accel[d].append(accel[:, d].mean())
            per_axis_jerk[d].append(jerk[:, d].mean())

    axis_names = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip"]

    return {
        "accel_mean": np.mean(all_accel) if all_accel else 0.0,
        "accel_std": np.std(all_accel) if all_accel else 0.0,
        "jerk_mean": np.mean(all_jerk) if all_jerk else 0.0,
        "jerk_std": np.std(all_jerk) if all_jerk else 0.0,
        "accel_per_axis": {
            axis_names[d]: (np.mean(per_axis_accel[d]), np.std(per_axis_accel[d]))
            for d in range(7)
        },
        "jerk_per_axis": {
            axis_names[d]: (np.mean(per_axis_jerk[d]), np.std(per_axis_jerk[d]))
            for d in range(7)
        },
    }


def compute_residual_stats(residuals_list: np.ndarray) -> dict:
    """Compute residual magnitude statistics."""
    all_abs = []
    per_axis_abs = [[] for _ in range(7)]
    # Time-binned residual magnitude (normalize to 10 bins)
    time_bins = 10
    bin_vals = [[] for _ in range(time_bins)]

    for ep_res in residuals_list:
        a = np.array(ep_res)  # (T, 7)
        abs_res = np.abs(a)
        all_abs.append(abs_res.mean())
        for d in range(7):
            per_axis_abs[d].append(abs_res[:, d].mean())

        T = len(a)
        for t in range(T):
            b = min(int(t / T * time_bins), time_bins - 1)
            bin_vals[b].append(np.abs(a[t]).mean())

    axis_names = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip"]
    return {
        "mean_abs": np.mean(all_abs) if all_abs else 0.0,
        "per_axis": {
            axis_names[d]: np.mean(per_axis_abs[d]) if per_axis_abs[d] else 0.0
            for d in range(7)
        },
        "time_profile": [
            np.mean(bin_vals[b]) if bin_vals[b] else 0.0
            for b in range(time_bins)
        ],
    }


def compute_episode_length(successes: np.ndarray, lengths: np.ndarray) -> dict:
    """Episode length stats for successful vs failed episodes."""
    succ_mask = successes.astype(bool)
    succ_lens = lengths[succ_mask]
    fail_lens = lengths[~succ_mask]
    return {
        "success_mean": np.mean(succ_lens) if len(succ_lens) > 0 else 0.0,
        "success_std": np.std(succ_lens) if len(succ_lens) > 0 else 0.0,
        "fail_mean": np.mean(fail_lens) if len(fail_lens) > 0 else 0.0,
        "n_success": int(succ_mask.sum()),
        "n_fail": int((~succ_mask).sum()),
    }


def compute_action_variance(actions_list: np.ndarray) -> dict:
    """Compute per-step action variance across episodes (consistency)."""
    # Pad to max length and compute variance at each timestep
    max_T = max(len(a) for a in actions_list)
    n_ep = len(actions_list)

    padded = np.full((n_ep, max_T, 7), np.nan, dtype=np.float32)
    for i, a in enumerate(actions_list):
        arr = np.array(a)
        padded[i, :len(arr)] = arr

    # Per-timestep variance (ignoring NaN)
    with np.errstate(all='ignore'):
        var_per_step = np.nanvar(padded, axis=0)  # (T, 7)
        mean_var = np.nanmean(var_per_step, axis=0)  # (7,)

    axis_names = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip"]
    return {
        axis_names[d]: float(mean_var[d]) for d in range(7)
    }


# ── Printing ──

def print_comparison(task: str, base_data: dict, res_data: dict):
    """Print side-by-side comparison for a single task."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {task.upper()} — Base vs Residual")
    print(sep)

    # 1. SR
    base_sr = float(base_data["sr"])
    res_sr = float(res_data["sr"])
    base_n = int(base_data["total_episodes"])
    res_n = int(res_data["total_episodes"])
    delta = res_sr - base_sr
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"\n  SR:  Base {base_sr:.1%} ({base_n} ep)  |  Residual {res_sr:.1%} ({res_n} ep)  |  Δ {delta:+.1%} {arrow}")

    # 2. Episode length
    base_el = compute_episode_length(base_data["successes"], base_data["episode_lengths"])
    res_el = compute_episode_length(res_data["successes"], res_data["episode_lengths"])
    print(f"\n  Episode Length (success):")
    print(f"    Base:     {base_el['success_mean']:.0f} ± {base_el['success_std']:.0f}  (n={base_el['n_success']})")
    print(f"    Residual: {res_el['success_mean']:.0f} ± {res_el['success_std']:.0f}  (n={res_el['n_success']})")
    if base_el['success_mean'] > 0 and res_el['success_mean'] > 0:
        speed = (1 - res_el['success_mean'] / base_el['success_mean']) * 100
        print(f"    → Residual is {abs(speed):.1f}% {'faster' if speed > 0 else 'slower'}")

    # 3. Smoothness
    base_sm = compute_smoothness(base_data["actions"])
    res_sm = compute_smoothness(res_data["actions"])
    print(f"\n  Action Smoothness (lower = smoother):")
    print(f"    {'Metric':<20} {'Base':>10} {'Residual':>10} {'Δ':>10}")
    print(f"    {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
    print(f"    {'Acceleration (mean)':<20} {base_sm['accel_mean']:>10.5f} {res_sm['accel_mean']:>10.5f} {res_sm['accel_mean']-base_sm['accel_mean']:>+10.5f}")
    print(f"    {'Jerk (mean)':<20} {base_sm['jerk_mean']:>10.5f} {res_sm['jerk_mean']:>10.5f} {res_sm['jerk_mean']-base_sm['jerk_mean']:>+10.5f}")

    print(f"\n  Per-Axis Acceleration:")
    print(f"    {'Axis':<10} {'Base':>10} {'Residual':>10} {'Δ':>10}")
    print(f"    {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    axis_names = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip"]
    for ax in axis_names:
        bv = base_sm["accel_per_axis"][ax][0]
        rv = res_sm["accel_per_axis"][ax][0]
        print(f"    {ax:<10} {bv:>10.5f} {rv:>10.5f} {rv-bv:>+10.5f}")

    print(f"\n  Per-Axis Jerk:")
    print(f"    {'Axis':<10} {'Base':>10} {'Residual':>10} {'Δ':>10}")
    print(f"    {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for ax in axis_names:
        bv = base_sm["jerk_per_axis"][ax][0]
        rv = res_sm["jerk_per_axis"][ax][0]
        print(f"    {ax:<10} {bv:>10.5f} {rv:>10.5f} {rv-bv:>+10.5f}")

    # 4. Residual magnitude
    res_stats = compute_residual_stats(res_data["residuals"])
    print(f"\n  Residual Magnitude (residual mode only):")
    print(f"    Mean |residual|: {res_stats['mean_abs']:.5f}")
    print(f"    Per axis: ", end="")
    for ax in axis_names:
        print(f"{ax}={res_stats['per_axis'][ax]:.4f}  ", end="")
    print()
    print(f"    Time profile (10 bins): {['%.4f' % v for v in res_stats['time_profile']]}")

    # 5. Action variance
    base_av = compute_action_variance(base_data["actions"])
    res_av = compute_action_variance(res_data["actions"])
    print(f"\n  Action Variance (consistency, lower = more consistent):")
    print(f"    {'Axis':<10} {'Base':>12} {'Residual':>12}")
    print(f"    {'─'*10} {'─'*12} {'─'*12}")
    for ax in axis_names:
        print(f"    {ax:<10} {base_av[ax]:>12.6f} {res_av[ax]:>12.6f}")

    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="outputs/trajectory_analysis")
    p.add_argument("--tasks", type=str, nargs="+", default=["lift", "pnp", "stack"])
    args = p.parse_args()

    print("=" * 72)
    print("  TRAJECTORY ANALYSIS: Base vs Residual Policy")
    print("=" * 72)

    all_results = {}
    for task in args.tasks:
        base_data = load_data(args.data_dir, task, "base")
        res_data = load_data(args.data_dir, task, "residual")
        if base_data is None or res_data is None:
            print(f"\n  [SKIP] {task}: missing data files")
            continue
        print_comparison(task, base_data, res_data)
        all_results[task] = {"base": base_data, "residual": res_data}

    # Cross-task summary
    if all_results:
        print("=" * 72)
        print("  CROSS-TASK SUMMARY")
        print("=" * 72)
        print(f"\n  {'Task':<10} {'Base SR':>10} {'Res SR':>10} {'Δ SR':>10} {'Base Jerk':>12} {'Res Jerk':>12} {'Δ Jerk':>12}")
        print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12} {'─'*12}")
        for task in args.tasks:
            if task not in all_results:
                continue
            bd = all_results[task]["base"]
            rd = all_results[task]["residual"]
            bsr = float(bd["sr"])
            rsr = float(rd["sr"])
            bj = compute_smoothness(bd["actions"])["jerk_mean"]
            rj = compute_smoothness(rd["actions"])["jerk_mean"]
            print(f"  {task:<10} {bsr:>10.1%} {rsr:>10.1%} {rsr-bsr:>+10.1%} {bj:>12.5f} {rj:>12.5f} {rj-bj:>+12.5f}")
        print()


if __name__ == "__main__":
    main()

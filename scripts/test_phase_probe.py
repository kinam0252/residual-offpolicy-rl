#!/usr/bin/env python3
"""
Test phase probe accuracy on offline dataset.

Loads the frozen linear probe (2048→3) and evaluates it against ground-truth
phase labels derived from env state (cube_height, finger_cube_dist, has_contact).

Usage:
    python scripts/test_phase_probe.py \
        --probe_path assets/configs/phase_probe.pt \
        --data_dir  assets/lerobot_data/pickMushroom_train_dense_clipped_3cm_vlm

    # With custom thresholds:
    python scripts/test_phase_probe.py \
        --probe_path assets/configs/phase_probe.pt \
        --data_dir  assets/lerobot_data/pickMushroom_train_dense_clipped_3cm_vlm \
        --lift_height_thresh 0.005 --grasp_dist_thresh 0.05
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PHASE_NAMES = ["approach", "grasp", "lift"]


def compute_gt_phase(
    cube_height: float,
    finger_cube_dist: float,
    has_contact: float,
    *,
    lift_h: float = 0.005,
    grasp_d: float = 0.05,
) -> int:
    """Ground truth phase label (same logic as training).
    0=approach, 1=grasp, 2=lift.
    """
    if cube_height >= lift_h and has_contact > 0.5:
        return 2  # lift
    elif finger_cube_dist < grasp_d or has_contact > 0.5:
        return 1  # grasp
    else:
        return 0  # approach


def main():
    parser = argparse.ArgumentParser(description="Test phase probe on offline data")
    parser.add_argument("--probe_path", type=str, required=True,
                        help="Path to phase_probe.pt")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to LeRobot-format offline dataset")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to test (None=all)")
    parser.add_argument("--lift_height_thresh", type=float, default=0.005,
                        help="Cube height threshold for lift phase (m)")
    parser.add_argument("--grasp_dist_thresh", type=float, default=0.05,
                        help="Finger-cube distance threshold for grasp phase (m)")
    parser.add_argument("--show_errors", type=int, default=10,
                        help="Number of error examples to print per class")
    args = parser.parse_args()

    # ── Load probe ──
    probe_ckpt = torch.load(args.probe_path, map_location="cpu")
    weight = probe_ckpt["weight"]  # (3, 2048)
    bias = probe_ckpt["bias"]      # (3,)
    probe = torch.nn.Linear(weight.shape[1], weight.shape[0])
    probe.weight.data.copy_(weight)
    probe.bias.data.copy_(bias)
    probe.eval()
    probe.requires_grad_(False)

    in_dim = weight.shape[1]
    print(f"Probe: Linear({in_dim} → 3)")
    if "test_accuracy" in probe_ckpt:
        print(f"  Saved test accuracy: {probe_ckpt['test_accuracy']*100:.1f}%")
    print()

    # ── Load data ──
    data_dir = Path(args.data_dir)
    parquet_files = sorted(glob.glob(str(data_dir / "data" / "chunk-000" / "episode_*.parquet")))
    if args.max_episodes:
        parquet_files = parquet_files[:args.max_episodes]
    print(f"Testing on {len(parquet_files)} episodes from {data_dir}")
    print(f"GT thresholds: lift_h={args.lift_height_thresh}m, grasp_d={args.grasp_dist_thresh}m")
    print()

    # ── Run inference ──
    all_gt = []
    all_pred = []
    all_probs = []
    all_meta = []  # (episode, step, cube_h, finger_dist, contact_force)

    for ep_idx, pq in enumerate(parquet_files):
        df = pd.read_parquet(pq)
        ep_name = Path(pq).stem

        if "vlm_latent" not in df.columns:
            print(f"  SKIP {ep_name}: no vlm_latent column")
            continue

        vlm = np.stack(df["vlm_latent"].values).astype(np.float32)  # (T, 2048)
        cube_h = df["cube_height"].values.astype(np.float32)
        finger_d = df["finger_cube_dist"].values.astype(np.float32)
        has_contact = df["has_contact"].values.astype(np.float32) if "has_contact" in df.columns else np.zeros(len(df), dtype=np.float32)
        contact_f = df["contact_force"].values.astype(np.float32) if "contact_force" in df.columns else np.zeros(len(df), dtype=np.float32)

        # GT labels
        gt_labels = np.array([
            compute_gt_phase(cube_h[t], finger_d[t], has_contact[t],
                             lift_h=args.lift_height_thresh, grasp_d=args.grasp_dist_thresh)
            for t in range(len(df))
        ])

        # Probe inference
        with torch.no_grad():
            logits = probe(torch.from_numpy(vlm))  # (T, 3)
            probs = F.softmax(logits, dim=-1).numpy()  # (T, 3)
        pred_labels = probs.argmax(axis=1)

        all_gt.append(gt_labels)
        all_pred.append(pred_labels)
        all_probs.append(probs)
        for t in range(len(df)):
            all_meta.append((ep_name, t, cube_h[t], finger_d[t], contact_f[t], has_contact[t]))

        if (ep_idx + 1) % 50 == 0:
            print(f"  Processed {ep_idx + 1}/{len(parquet_files)} episodes...")

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    all_probs = np.concatenate(all_probs)
    correct = (all_gt == all_pred)

    # ── Overall accuracy ──
    print()
    print("=" * 65)
    print(f"  PHASE PROBE TEST RESULTS  ({len(all_gt)} timesteps)")
    print("=" * 65)
    print(f"  Overall accuracy: {correct.mean()*100:.1f}%")
    print()

    # ── Per-class metrics ──
    print(f"{'Phase':<12} {'Count':>7} {'Correct':>9} {'Accuracy':>10} {'Precision':>11} {'Recall':>8}")
    print("-" * 65)
    for cls_id, cls_name in enumerate(PHASE_NAMES):
        gt_mask = (all_gt == cls_id)
        pred_mask = (all_pred == cls_id)
        tp = (gt_mask & pred_mask).sum()
        gt_count = gt_mask.sum()
        pred_count = pred_mask.sum()
        recall = tp / max(gt_count, 1)
        precision = tp / max(pred_count, 1)
        acc = correct[gt_mask].mean() if gt_count > 0 else 0.0
        print(f"  {cls_name:<10} {gt_count:>7} {tp:>9} {acc*100:>9.1f}% {precision*100:>10.1f}% {recall*100:>7.1f}%")

    print("-" * 65)
    print()

    # ── Confusion matrix ──
    print("Confusion Matrix (rows=GT, cols=Pred):")
    print(f"{'':>12}", end="")
    for name in PHASE_NAMES:
        print(f" {name:>10}", end="")
    print()
    for gt_id, gt_name in enumerate(PHASE_NAMES):
        print(f"  {gt_name:<10}", end="")
        gt_mask = (all_gt == gt_id)
        for pred_id in range(3):
            count = ((all_gt == gt_id) & (all_pred == pred_id)).sum()
            pct = count / max(gt_mask.sum(), 1) * 100
            print(f" {count:>6}({pct:>4.1f}%)", end="")
        print()
    print()

    # ── Confidence analysis ──
    print("Confidence (mean max-prob):")
    for cls_id, cls_name in enumerate(PHASE_NAMES):
        gt_mask = (all_gt == cls_id)
        if gt_mask.sum() > 0:
            conf_correct = all_probs[gt_mask & correct].max(axis=1).mean() if (gt_mask & correct).sum() > 0 else 0
            conf_wrong = all_probs[gt_mask & ~correct].max(axis=1).mean() if (gt_mask & ~correct).sum() > 0 else 0
            print(f"  {cls_name:<10} correct: {conf_correct:.3f}  wrong: {conf_wrong:.3f}")
    print()

    # ── Phase transition analysis ──
    # Check probe accuracy at phase boundaries (most interesting)
    print("Phase transition accuracy (±2 steps around GT phase change):")
    boundary_correct = []
    nonboundary_correct = []
    for i in range(1, len(all_gt)):
        if all_gt[i] != all_gt[i - 1]:
            # Boundary: ±2 steps
            for j in range(max(0, i - 2), min(len(all_gt), i + 3)):
                boundary_correct.append(correct[j])
        else:
            nonboundary_correct.append(correct[i])
    if boundary_correct:
        print(f"  At boundaries:     {np.mean(boundary_correct)*100:.1f}% ({len(boundary_correct)} steps)")
    print(f"  Away from boundary: {np.mean(nonboundary_correct)*100:.1f}% ({len(nonboundary_correct)} steps)")
    print()

    # ── Error examples ──
    if args.show_errors > 0:
        print(f"Error examples (up to {args.show_errors} per class):")
        for cls_id, cls_name in enumerate(PHASE_NAMES):
            errors = np.where((all_gt == cls_id) & ~correct)[0]
            if len(errors) == 0:
                print(f"  {cls_name}: no errors!")
                continue
            print(f"  {cls_name} (GT) — {len(errors)} total errors:")
            for idx in errors[:args.show_errors]:
                ep, step, ch, fd, cf, hc = all_meta[idx]
                pred_id = all_pred[idx]
                p = all_probs[idx]
                print(f"    {ep} t={step:>3d}: pred={PHASE_NAMES[pred_id]} "
                      f"probs=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}] "
                      f"h={ch*100:.1f}cm dist={fd*100:.1f}cm cf={cf:.2f}N contact={'Y' if hc>0.5 else 'N'}")
            print()


if __name__ == "__main__":
    main()

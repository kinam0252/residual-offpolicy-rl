# Object State Augmentation — Ablation Study

> **Goal**: Test whether adding noise/dropout to object state observations during online RL training
> improves robustness of the learned residual policy.

## Motivation

GR00T base policies receive object state (pose, position) from an external perception system.
In real deployment, this signal is noisy and can drop out. Training with clean sim states may
cause the residual policy to overfit to perfect observations, leading to sim-to-real degradation.

This ablation tests 4 conditions:
1. **clean** — no augmentation (baseline)
2. **noise** — uniform noise σ ~ U(0, 0.01) per dim
3. **drop** — 10% probability of zeroing entire object rows
4. **both** — noise + dropout combined

---

## Results Summary

### PNP (Jobs 747-750, 400-432K/500K steps)

| Condition | Base SR | Peak SR | Final SR | Avg Last 10 Evals |
|-----------|---------|---------|----------|--------------------|
| clean     | 55%     | 75%     | 65%      | 41.0%              |
| **noise** | 60%     | **80%** ⭐ | 20%  | 39.5%              |
| drop      | 65%     | 70%     | 60%      | 49.5%              |
| both      | 65%     | 70%     | 35%      | 34.5%              |

### Stack (Jobs 763-766, 239-245K/500K steps)

| Condition | Base SR | Peak SR | Final SR | Avg Last 10 Evals |
|-----------|---------|---------|----------|--------------------|
| clean     | 45%     | 70%     | 30%      | 48.0%              |
| **noise** | 45%     | **75%** | 45%      | 48.5%              |
| **drop**  | 55%     | **75%** ⭐ | 65%  | 49.0%              |
| both      | 60%     | 70%     | 55%      | 52.0%              |

### Drawer (Jobs 751-754, completed at 50K/50K)

| Condition | Base SR | Peak SR | Final SR | Avg Last 10 Evals |
|-----------|---------|---------|----------|--------------------|
| clean     | 100%    | 100%    | 100%     | 100.0%             |
| noise     | 100%    | 100%    | 100%     | 100.0%             |
| drop      | 100%    | 100%    | 100%     | 100.0%             |
| both      | 100%    | 100%    | 100%     | 100.0%             |

> Drawer task is too easy (100% baseline) to differentiate augmentation effects.

---

## Peak SR Comparison (Base → Peak)

| Task  | clean       | noise        | drop         | both         |
|-------|-------------|--------------|--------------|--------------|
| PNP   | 55→75 (+20) | 60→**80** (+20) | 65→70 (+5)  | 65→70 (+5)  |
| Stack | 45→70 (+25) | 45→**75** (+30) | 55→**75** (+20) | 60→70 (+10) |

---

## Observations

1. **Noise augmentation shows highest peak SR** in both PNP (80%) and Stack (75%).

2. **High variance across all conditions** — latest SR often far below peak. This is
   characteristic of residual RL with small eval sets (20 episodes).

3. **Dropout alone doesn't consistently help** — PNP drop (70%) ≤ clean (75%).
   Stack drop peaks at 75% but starts from a higher base (55%).

4. **Combined (both) doesn't beat individual augmentations** — both PNP and Stack
   show "both" ≤ best single augmentation. Possible interference.

5. **Drawer uninformative** — 100% SR across all conditions; task too easy for ablation.

6. **avg_last10 is similar across conditions** (~34-52%), suggesting the augmentation
   effect is more about peak capability than stable improvement.

---

## Caveats

- **Single seed per condition** — no statistical significance can be claimed.
- **PNP stopped at ~80-86%** of training; Stack at ~48%. Results may shift with full runs.
- **Base SR varies between conditions** (55-65% for PNP) due to eval stochasticity at step=0.
  This makes direct Δ comparisons imprecise.
- **eval_num_episodes=1 per position** (20 positions) — high eval noise.

---

## Files

- `configs.md` — Full training configuration, augmentation implementation details
- `data/pnp_sr_history.csv` — PNP eval SR at every 2K steps (4 conditions)
- `data/stack_sr_history.csv` — Stack eval SR at every 2K steps (4 conditions)
- `data/drawer_sr_history.csv` — Drawer eval SR at every 2K steps (4 conditions)

---

## Job Reference

| Job ID | Task   | Condition | SLURM QoS | Duration    | Status    |
|--------|--------|-----------|-----------|-------------|-----------|
| 747    | PNP    | clean     | core-own  | 1d 12h      | cancelled |
| 748    | PNP    | noise     | core-own  | 1d 12h      | cancelled |
| 749    | PNP    | drop      | core-own  | 1d 12h      | cancelled |
| 750    | PNP    | both      | core-own  | 1d 12h      | cancelled |
| 751    | Drawer | clean     | core-own  | 2h 39m      | completed |
| 752    | Drawer | noise     | core-own  | 2h 50m      | completed |
| 753    | Drawer | drop      | extra     | 2h 46m      | completed |
| 754    | Drawer | both      | extra     | 2h 55m      | completed |
| 763    | Stack  | clean     | extra     | 1d 12h      | cancelled |
| 764    | Stack  | noise     | extra     | 1d 12h      | cancelled |
| 765    | Stack  | drop      | extra     | 1d 12h      | cancelled |
| 766    | Stack  | both      | extra     | 1d 12h      | cancelled |

> PNP/Stack cancelled early (not failed) — checkpoints and wandb logs are intact.

---

*Generated: 2026-05-11*

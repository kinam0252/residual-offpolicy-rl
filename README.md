# Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement

This repository contains the code for training and evaluating **object-centric residual RL policies** that enhance Vision-Language-Action (VLA) models via zero-shot sim-to-real transfer.

## Overview

The framework has three stages:

1. **Stage 1 — Paired VLA Training**: Train sim/real VLAs (GR00T-N1.5) on replayed teleoperation data
2. **Stage 2 — Residual RL Training**: Train a lightweight residual TD3 policy in MuJoCo using object-centric observations (6-DoF pose + proprioception + base VLA action)
3. **Stage 3 — Zero-Shot Deployment**: Deploy the sim-trained residual on a real FR3 robot without adaptation

**Tasks**: Cube Lift, Pick-and-Place, Stack Cube, Close Drawer, Stand Cup Up

## Repository Structure

```
resfit/
├── rl_finetuning/           # Residual RL training pipeline
│   ├── scripts/             # Training & evaluation scripts
│   ├── off_policy/          # TD3 implementation
│   ├── wrappers/            # MuJoCo environments & residual wrapper
│   ├── rewards/             # Task reward definitions
│   ├── configs/             # Task registry (task_configs.py)
│   ├── config/              # Algorithm hyperparameters
│   ├── rendering/           # Realistic MuJoCo rendering
│   └── utils/               # Utilities (offline buffer, evaluation)
├── lerobot/                 # Base VLA (GR00T) fine-tuning
│   ├── scripts/             # BC training scripts
│   ├── policies/            # Policy architectures
│   └── configs/             # BC training configs
configs/                     # Task-specific eval/position configs
├── tasks/                   # Per-task JSON configs
offline_data_generation/     # Teleoperation replay for sim data
workspace/                   # GR00T workspace setup
scripts/                     # Evaluation & diagnostic scripts
```

## Getting Started

### 1. Environment Setup

```bash
conda create -n residual python=3.10 -y
conda activate residual
./resfit/rl_finetuning/setup_rlpd_robosuite.sh
pip install wandb draccus==0.10.0 torchrl==0.9.2 hydra-core serial deepdiff matplotlib
```

### 2. Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
python scripts/check_render.py  # Verify MuJoCo rendering
```

### 3. Prepare Base VLA Checkpoints

Download or train GR00T-N1.5 checkpoints for each task and place them under `checkpoints/`:

```
checkpoints/
├── groot_lift_sim/checkpoint-100000
├── groot_pnp_sim/checkpoint-100000
├── groot_stack_sim/checkpoint-100000
├── groot_drawer_sim/checkpoint-100000
└── groot_cup_sim/checkpoint-100000
```

## Training

### Stage 1: Base VLA Fine-Tuning

Fine-tune GR00T-N1.5 on sim teleoperation data:

```bash
python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset <YOUR_DATASET> \
    --policy act \
    --steps 100000 \
    --batch_size 256 \
    --wandb_project groot-sim-vla
```

### Stage 2: Residual RL Training

Train the residual TD3 policy (example: Cube Lift):

```bash
python resfit/rl_finetuning/scripts/train_residual_td3_unified.py \
    --task lift \
    --groot_checkpoint checkpoints/groot_lift_sim/checkpoint-100000 \
    --reward_type dense \
    --use_obs_noise --obs_noise_max 0.005 \
    --use_obs_dropout --obs_dropout_prob 0.1
```

Available tasks: `lift`, `pnp`, `stack`, `drawer`, `cup`

### Evaluation

```bash
python resfit/rl_finetuning/scripts/eval_async_unified.py \
    --task lift \
    --groot_checkpoint checkpoints/groot_lift_sim/checkpoint-100000 \
    --residual_checkpoint <RESIDUAL_CHECKPOINT_PATH>
```

## Key Hyperparameters

| Parameter | Lift | PnP | Stack | Drawer | Cup |
|-----------|------|-----|-------|--------|-----|
| Actor LR | 1e-5 | 3e-4 | 3e-4 | 1e-5 | 1e-5 |
| Critic LR | 1e-4 | 3e-4 | 3e-4 | 1e-4 | 1e-4 |
| γ | 0.99 | 0.99 | 0.95 | 0.99 | 0.95 |
| Max episode steps | 300 | 500 | 500 | 500 | 500 |
| Pose noise σ_max | 0.005 (5mm) | | | | |
| Pose dropout ρ | 0.1 | | | | |

## License

See [LICENSE](./LICENSE) for details.

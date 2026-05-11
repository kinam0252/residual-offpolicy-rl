# Object State Augmentation — Experiment Configuration

## 1. Augmentation Implementation

### Noise (`--use_obs_noise --obs_noise_max σ_max`)
- Per timestep: σ ~ U(0, σ_max), then `obj_state += U(-σ, σ)` per dimension
- Uses `np.random.uniform` for σ, `torch.empty_like().uniform_()` for per-dim noise
- Guard: `if self.obs_noise_max > 0.0` — completely no-op when 0.0

### Dropout (`--use_obs_dropout --obs_dropout_prob p`)
- `torch.rand(N, 1) >= p` mask zeros **entire rows** with probability p
- Each object (row) is independently dropped
- Guard: `if self.obs_dropout_prob > 0.0` — completely no-op when 0.0

### Application Order & Scope
- **Order**: Noise first, then dropout
- **Scope**: Online rollouts only (wrapper `step()` and `reset()`)
- **NOT applied to**: Offline replay data, eval subprocess
- **Eval**: Always clean — `eval_async_unified.py` defaults to `use_obs_noise=False`, `use_obs_dropout=False`
- **Code location**: `mujoco_residual_wrapper_unified.py` L544-567, called at L234 (reset) and L298 (step)

### Object State Dimensions
| Task    | Dim  | Contents                                    |
|---------|------|---------------------------------------------|
| PNP     | 10D  | cube_pos(3) + cube_quat(4) + bowl_pos(3)   |
| Stack   | 10D  | white_pos(3) + white_quat(4) + green_pos(3)|
| Drawer  | 5D   | slide(1) + active_idx(1) + face_center(3)  |

---

## 2. Ablation Conditions

| Condition | `OBS_NOISE_MAX` | `OBS_DROPOUT_PROB` | Description        |
|-----------|-----------------|--------------------|--------------------|
| clean     | —               | —                  | No augmentation    |
| noise     | 0.01            | —                  | Noise only         |
| drop      | —               | 0.1                | Dropout only       |
| both      | 0.01            | 0.1                | Noise + Dropout    |

---

## 3. Per-Task Training Configuration

### PNP (Jobs 747-750)
| Parameter            | Value                          | Source          |
|----------------------|--------------------------------|-----------------|
| task                 | pnp                            | JSON            |
| action_scale         | **0.1**                        | env var override (JSON default: 0.2) |
| reward_type          | dense_v3                       | JSON            |
| gamma                | 0.99                           | JSON            |
| action_l2_reg        | 0.01                           | JSON            |
| offline_fraction     | 0.5                            | JSON            |
| total_timesteps      | 500,000                        | JSON            |
| num_envs             | 30                             | JSON            |
| eval_num_envs        | 20                             | JSON            |
| eval_interval        | 2,000                          | CLI             |
| groot_checkpoint     | `groot_pnp_sim_66ep/checkpoint-100000` | JSON    |
| offline_data         | `outputs/offline_data/pnp_30pos_states` (300 files) | JSON |
| no_action_clamp      | true                           | CLI             |
| use_action_scaler    | true                           | CLI             |
| chunk_sync           | true                           | CLI             |

### Stack (Jobs 763-766)
| Parameter            | Value                          | Source          |
|----------------------|--------------------------------|-----------------|
| task                 | stack                          | JSON            |
| action_scale         | **0.05**                       | env var override (JSON default: 0.1) |
| reward_type          | dense                          | JSON            |
| gamma                | 0.95                           | JSON            |
| action_l2_reg        | 0.0                            | JSON            |
| offline_fraction     | 0.5                            | JSON            |
| total_timesteps      | 500,000                        | JSON            |
| num_envs             | 30                             | JSON            |
| eval_num_envs        | 20                             | JSON            |
| eval_interval        | 2,000                          | CLI             |
| groot_checkpoint     | `groot_stack_sim_66ep/checkpoint-75000` | JSON     |
| offline_data         | `outputs/offline_data/stack_75k` (19,800 files) | JSON |
| offline_reward_relabel | computed (reward_config: `configs/reward_stack.yaml`) | JSON |
| no_action_clamp      | true                           | CLI             |
| use_action_scaler    | true                           | CLI             |
| chunk_sync           | true                           | CLI             |

### Drawer (Jobs 751-754, completed)
| Parameter            | Value                          | Source          |
|----------------------|--------------------------------|-----------------|
| task                 | drawer                         | JSON            |
| action_scale         | **0.05**                       | env var override (JSON default: 0.1) |
| reward_type          | **delta**                      | env var override (JSON default: dense) |
| gamma                | 0.99                           | JSON            |
| action_l2_reg        | 10.0                           | JSON            |
| offline_fraction     | 0.3                            | JSON            |
| total_timesteps      | 50,000                         | JSON            |
| num_envs             | 10                             | JSON            |
| eval_num_envs        | 10                             | JSON            |
| eval_interval        | 2,000                          | CLI             |
| groot_checkpoint     | `groot_drawer_sim_33ep/checkpoint-100000` | JSON  |
| offline_data         | `outputs/offline_drawer_zgate_100k` (90 files) | JSON |
| no_action_clamp      | true                           | CLI             |
| use_action_scaler    | true                           | CLI             |
| chunk_sync           | true                           | CLI             |

---

## 4. SLURM Configuration

| Parameter       | Value                                    |
|-----------------|------------------------------------------|
| GPU             | 1 × GPU (`--gres=gpu:1`)                |
| CPU             | 14 cores (`--cpus-per-task=14`)          |
| Memory          | 219 GB                                   |
| Time limit      | 7 days                                   |
| QoS             | `core-own` (PNP 747-750, Drawer 751-752), `extra` (Drawer 753-754, Stack 763-766) |
| Rendering       | `MUJOCO_GL=egl` with GPU-matched EGL device |

---

## 5. Environment Variable → CLI Mapping

The sweep script (`scripts/slurm/sweep_obs_augment_all.sh`) sets env vars before `sbatch`.
`train_td3.sh` converts them to CLI args:

```
OBS_NOISE_MAX=0.01  →  --use_obs_noise --obs_noise_max 0.01
OBS_DROPOUT_PROB=0.1 →  --use_obs_dropout --obs_dropout_prob 0.1
ACTION_SCALE=0.1     →  --action_scale 0.1
REWARD_TYPE=delta    →  --reward_type delta
```

Per-task deploy overrides (matching `REAL_DEPLOY_CHECKPOINTS.md`):
- **PNP**: `ACTION_SCALE=0.1`
- **Stack**: `ACTION_SCALE=0.05`
- **Drawer**: `ACTION_SCALE=0.05 REWARD_TYPE=delta`

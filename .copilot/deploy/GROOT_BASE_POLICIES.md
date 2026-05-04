# GR00T Base Policy Checkpoints

Base: `~/DATA/INTERN/training/`

## Per-Task

| Task | Sim | Real |
|------|-----|------|
| Lift | `gr00t_sim_100ep/` | ❌ 없음 |
| PnP | `groot_pnp_sim_66ep/` | `groot_pnp_real_100ep/` |
| | `groot_pnp_sim_100ep/` | `groot_pnp_real_66ep/` |
| | `groot_pnp_sim_33ep/` | `groot_pnp_real_33ep/` |
| | `groot_pnp_sim_20ep/` | `groot_pnp_real_20ep/` |
| | `groot_pnp_sim_10ep/` | `groot_pnp_real_10ep/` |
| Stack | `groot_stack_sim_66ep/` | `groot_stack_real_66ep/` |
| Cup | `groot_cup_sim_27ep/` | `groot_cup_real_27ep/` |
| Drawer | `groot_drawer_sim_33ep/` | `groot_drawer_real_33ep/` |

## RL 학습에 사용된 Sim Base Policy

| Task | Base Policy | Checkpoint Step |
|------|-------------|-----------------|
| Lift | `gr00t_sim_100ep/` | 불명 |
| PnP | `groot_pnp_sim_66ep/` | 100000 |
| Stack | `groot_stack_sim_66ep/` | 75000 |
| Cup | `groot_cup_sim_27ep/` | 100000 |
| Drawer | `groot_drawer_sim_33ep/` | 100000 |

## 참고

- Lift는 task-specific GR00T가 없고 범용 `gr00t_sim_100ep/` 사용
- PnP는 real 데이터 양 별로 5가지 버전 존재
- 각 폴더 안에 `checkpoint-25000` ~ `checkpoint-300000` 존재

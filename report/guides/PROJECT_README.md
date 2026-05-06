# .copilot — Residual TD3 프로젝트 문서

> Residual Off-Policy RL (TD3) + GR00T VLA base policy on MuJoCo

## 프로젝트 개요

GR00T VLA를 frozen base policy로 사용하고, Residual TD3로 fine-tuning하는 프로젝트.
MuJoCo 환경에서 4개 task를 학습합니다.

## 코드 구조

```
resfit/rl_finetuning/
├── off_policy/rl/
│   ├── q_agent.py         # 🌐 TD3 agent (공유)
│   ├── actor.py           # 🌐 Actor network (공유)
│   └── critic.py          # 🌐 Critic network (공유)
├── utils/
│   └── normalization.py   # 🌐 ActionScaler (공유)
├── config/
│   ├── residual_td3_mujoco.py         # Lift config
│   ├── residual_td3_mujoco_drawer.py  # Drawer config
│   └── residual_td3_mujoco_pnp.py    # PnP config
├── wrappers/
│   ├── mujoco_residual_wrapper.py         # 📁 Lift wrapper (원본, 접미사 없음)
│   ├── mujoco_residual_wrapper_drawer.py  # 📁 Drawer wrapper
│   ├── mujoco_residual_wrapper_stack.py   # 📁 Stack wrapper
│   ├── mujoco_residual_wrapper_pnp.py     # 📁 PnP wrapper
│   └── mujoco_residual_wrapper_cup.py     # 📁 Cup (Stand Cup) wrapper
└── scripts/
    ├── train_residual_td3_mujoco.py         # 📁 Lift train (원본, 접미사 없음)
    ├── train_residual_td3_mujoco_drawer.py  # 📁 Drawer train
    ├── train_residual_td3_mujoco_stack.py   # 📁 Stack train
    ├── train_residual_td3_mujoco_pnp.py     # 📁 PnP train
    ├── eval_async_mujoco.py                 # 📁 Lift eval (원본)
    ├── eval_async_mujoco_drawer.py          # 📁 Drawer eval
    ├── eval_async_mujoco_stack.py           # 📁 Stack eval
    └── eval_async_mujoco_pnp.py             # 📁 PnP eval

scripts/                    # SLURM 스크립트, standalone eval 등
.copilot/                   # 이 문서 폴더
```

- 🌐 = 공유 코드 (수정 시 모든 task에 영향)
- 📁 = Task-specific (해당 task에만 영향)

## 문서 목록

| 문서 | 내용 | 언제 보나 |
|------|------|----------|
| [TASK_CONFIGS.md](TASK_CONFIGS.md) | Task별 하이퍼파라미터 비교표 | 새 task 시작할 때 |
| [TASK_SWITCHING_CHECKLIST.md](TASK_SWITCHING_CHECKLIST.md) | Task 전환 시 체크리스트 | **task 바꿀 때마다** |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 알려진 버그 + 수정 상태 | 이상 현상 발생 시, task 전환 시 |
| [SLURM_GUIDE.md](SLURM_GUIDE.md) | SLURM 제출 방법 + 파티션 | 잡 제출할 때 |
| [MODIFICATIONS.md](MODIFICATIONS.md) | 원본 대비 수정사항 | 코드 변경/커밋 시 |

## Quick Start — Task 전환

1. **`TASK_SWITCHING_CHECKLIST.md`** 열기
2. 체크리스트 항목 하나씩 확인
3. 의심되면 **`KNOWN_ISSUES.md`**에서 해당 task 버그 상태 확인
4. SLURM 제출 전 **`SLURM_GUIDE.md`** 참조
5. 하이퍼파라미터는 **`TASK_CONFIGS.md`** 비교표 참조

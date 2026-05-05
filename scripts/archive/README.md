# ⚠️ ARCHIVED — Per-Task SLURM Scripts (Deprecated)

구버전 per-task 코드용 SLURM 스크립트들.
신규 실험은 unified script로 제출하세요:

```bash
TASK=stack sbatch --job-name=stk_u1 scripts/slurm_train_td3.sh
TASK=drawer sbatch --job-name=drawer_u1 scripts/slurm_train_td3.sh
```

지원 task: cup, pnp, lift, stack, drawer

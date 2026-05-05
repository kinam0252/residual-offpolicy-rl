# ⚠️ ARCHIVED — Per-Task Scripts (Deprecated)

이 폴더의 스크립트들은 **구버전 per-task 코드**입니다.

## 왜 deprecated?

1. **Replay buffer 버그**: residual action만 저장 → critic/actor action space 불일치
2. **ActionScaler 미적용**: 고정 physical scale + tanh clamp만 사용
3. **코드 중복**: task별로 거의 동일한 코드가 복사됨

## 대신 사용할 것

```bash
# Unified script (모든 task 지원: cup, pnp, lift, stack, drawer)
python resfit/rl_finetuning/scripts/train_residual_td3_unified.py --task {cup,pnp,lift,stack,drawer}
python resfit/rl_finetuning/scripts/eval_async_unified.py --task {cup,pnp,lift,stack,drawer}

# SLURM 제출
TASK=stack sbatch --job-name=stk_u1 scripts/slurm_train_td3.sh
```

- ActionScaler 사용: `--use_action_scaler`
- Replay에 normalized combined action 저장 (수정됨)
- 모든 task 공통 코드 → 버그 수정 시 한 곳만 변경


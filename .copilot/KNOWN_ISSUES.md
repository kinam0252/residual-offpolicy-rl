# 알려진 이슈 및 버그

> 마지막 업데이트: 2026-05-02

## 🔴 Critical (성능에 직접 영향)

### 1. Replay Buffer Clamp 버그
- **상태**: 🟡 Drawer만 수정 (uncommitted), Stack 미확인
- **영향 task**: Drawer (수정됨), Stack (미확인), Lift/원본 wrapper (미확인, `mujoco_residual_wrapper.py`에도 같은 패턴), PnP (ActionScaler 미사용이라 해당 없음)
- **설명**: ActionScaler 모드에서 replay buffer에 저장하는 action이 clamp 전 값 (pre-clamp)이었음. 실제 env가 실행한 action은 clamp 후 값 (post-clamp). → critic이 실제와 다른 action에 대해 학습
- **수정**: `_combine_actions()`에서 executed physical action을 다시 `scaler.scale()`해서 clamped normalized 값 저장
- **파일**: `mujoco_residual_wrapper_drawer.py` (line ~310-322)
- **TODO**: Stack wrapper에도 같은 패턴 있는지 확인 필요 (line ~421-425)

### 2. open_loop_horizon 불일치 (Eval)
- **상태**: ✅ 해결됨
- **영향 task**: 모든 task
- **설명**: eval script에서 `open_loop_horizon=1`로 하면 GR00T base policy가 매 step 새로 추론 → 학습 시 horizon=16과 완전히 다른 행동 → SR 0%
- **수정**: eval 시 반드시 `open_loop_horizon=16` 사용
- **교훈**: standalone eval script (`scripts/standalone_eval_drawer.py`) 참조

### 3. GR00T Chunking Non-Markov State
- **상태**: 🔴 미해결 (연구 과제)
- **영향 task**: 모든 task
- **설명**: GR00T는 16-step action chunk를 사용하지만, 현재 chunk의 몇 번째 step인지 (chunk phase 0~15)가 observation에 포함되지 않음. → TD3 critic이 Markov 가정 위반 → Q-value 진동
- **근거**: Rubber-duck 분석에서 최우선 이슈로 지목
- **잠재적 해결**: observation에 chunk_phase (0~15) 추가

## 🟡 Important (학습 안정성 관련)

### 4. Residual Saturation
- **상태**: 🟡 관찰됨, 미해결
- **영향 task**: 모든 task (특히 action_scale이 작을 때)
- **설명**: actor가 항상 action_scale 최대치로 포화. "미세 보정" 대신 "항상 최대로 밀기"를 학습
- **관련 지표**: wandb `train/train/residual_abs_max` ≈ action_scale 상한

### 5. Exploration Noise 비율
- **상태**: 🟡 관찰됨, 미해결
- **영향 task**: 모든 task
- **설명**: `stddev=0.05`는 `action_scale=0.05`일 때 **100%** 비율 (너무 큼). action_scale에 비례해서 noise를 조절해야 함
- **권장**: `stddev = 0.2 * action_scale` 정도로 스케일링

### 6. Q-Value Explosion (fix1 관찰)
- **상태**: 🟡 관찰됨
- **영향 task**: Drawer (fix1 실험에서 발생)
- **설명**: critic warmup 단계에서 loss spike가 회복되지 않으면 cascade → Qt 2→10까지 폭발
- **원인**: random variance. bug fix 자체와는 무관 (J도 같은 spike가 있었으나 회복)
- **대응**: wandb에서 초기 critic_loss 추이 모니터링

## 🟢 Resolved

### 7. q_agent state_only dim assert
- **상태**: ✅ 수정됨 (committed)
- **설명**: `q_agent.py`에서 `state.shape[1] == 10` assert가 state_only=True task (Drawer)에서 실패
- **수정**: `if not self.state_only` 조건 추가
- **파일**: `resfit/rl_finetuning/off_policy/rl/q_agent.py` (line ~940)

### 8. Eval object_state_dim 불일치
- **상태**: ✅ 수정됨 (committed)
- **설명**: Eval에서 `object_state_dim=10`으로 QAgent 생성했으나 학습은 7 (cube_pos 3D + cube_quat 4D) → shape mismatch
- **파일**: commit `bb63bb5`

### 9. ActionScaler attribute access 버그
- **상태**: ✅ 수정됨 (committed)
- **파일**: commit `785b9fc`

### 10. ActionScaler eval subprocess 전달
- **상태**: ✅ 수정됨 (committed)
- **파일**: commit `c938641`

### 11. Replay buffer 공간 불일치 (original)
- **상태**: ✅ 수정됨 (committed)
- **설명**: replay에 residual만 저장하는데 actor loss는 base+residual=combined으로 계산 → 공간 불일치
- **파일**: commit `740f7cb`

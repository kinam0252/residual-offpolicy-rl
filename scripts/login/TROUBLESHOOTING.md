# Login Node Troubleshooting Guide

로그인 노드에서 디버깅/테스트 시 발생하는 에러와 해결법 모음.

---

## EGL_BAD_ACCESS (멀티스레드 렌더링)

**증상**: `eglMakeCurrent failed with EGL_BAD_ACCESS`

**원인**: EGL context는 **thread-affine**. 다음 상황에서 발생:
1. `async_prefetch=True` → background thread에서 GR00T 추론 시 `get_groot_obs()` → EGL rendering → main thread와 충돌
2. `render_parallel=True` → ThreadPoolExecutor에서 여러 env 동시 렌더링
3. Critic warmup의 heavy CUDA ops가 EGL display state를 corrupt (드문 케이스)

**해결** (2026-05-06 확인):
```python
# train_residual_td3_unified.py & eval_async_unified.py:
wrapper = MuJoCoResidualWrapperUnified(..., async_prefetch=False)

# critic warmup 후 renderer 재생성 (train script에 이미 적용):
env.vec_env.rebuild_renderers()
```

**핵심**: `async_prefetch`는 성능 최적화이지만, EGL이 thread-safe하지 않아 training에서 사용 불가. Eval subprocess에서도 동일하게 `async_prefetch=False` 필수.

**이력**:
- Job 8560 (성공): `async_prefetch` 기능이 존재하지 않던 코드 버전
- Job 8835~10635 (실패): `async_prefetch=True` 기본값이 적용됨
- Job 10658+ (성공): `async_prefetch=False` 명시 + `rebuild_renderers()`

---

## MuJoCo 렌더링 검은 화면 / segfault

**증상**: 렌더링 결과가 전부 검은색이거나 segfault 발생

**원인**: `MUJOCO_GL` 미설정 또는 EGL 라이브러리 경로 누락

**해결**:
```bash
export MUJOCO_GL=egl
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/.local/lib:$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cublas/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nccl/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}
```

---

## SubprocVecEnv 데드락

**증상**: 환경 생성 또는 reset 시 무한 대기

**원인**: `multiprocessing.spawn` context + EGL 충돌. Worker가 EGL 초기화하면 main process와 충돌.

**해결**: Worker는 반드시 `MUJOCO_GL=osmesa` (렌더링 안함), main process만 EGL 사용. 스크립트에 `if __name__ == "__main__":` 가드 필수 (spawn이 모듈 재실행).

---

## PnP cube-bowl 겹침 (수정됨)

**증상**: PnP 환경에서 cube와 bowl이 같은 위치에 겹쳐서 시작

**원인**: SubprocVecEnv에서 double-reset race condition — worker가 설정한 위치와 main process가 설정한 위치가 다르고, `_sync_qpos_all`이 `model.body_pos` (bowl)를 sync하지 않음.

**해결**: `be724bd` 커밋에서 수정됨.
- `_reset_single_env()`: parallel 모드에서 위치 재설정 스킵
- Worker `get_qpos_all`: bowl_pos도 전송
- `_sync_qpos_all`: bowl body_pos도 sync

---

## Gripper가 cube를 안잡음 (lift)

**증상**: Lift task에서 base policy가 gripper를 닫지 않음

**원인**: `GRIPPER_CLOSE_THRESHOLD=0.5`이 training data의 gripper 최솟값(~0.49)과 거의 동일. GR00T 출력에 약간의 노이즈만 있어도 threshold 미만으로 못내려감.

**상태**: 미수정. 해결 옵션:
1. Threshold를 0.6~0.7로 올리기
2. Gripper latch 추가 (cup처럼)

---

## Gripper 열림/닫힘 반복 (PnP)

**증상**: PnP에서 grasp 후 gripper가 열렸다 닫혔다 반복

**원인**: `GRIPPER_CLOSE_THRESHOLD=0.65`에서 GR00T 출력이 threshold 근처에서 요동. Latch가 없어서 매 스텝 grasp 상태 전환.

**상태**: 미수정. 해결 옵션: gripper latch 활성화 (`use_gripper_latch=True` in task_configs.py)

---

## Gripper actuator ID = -1

**증상**: Gripper가 아예 동작하지 않음. 모든 에피소드에서 gripper 닫힌 채로 시작.

**원인**: FR3 XML에 `"gripper"` tendon actuator가 없고 `finger_joint1`/`finger_joint2` 개별 actuator만 존재. `mj_name2id("gripper")` → -1 반환 → 모든 gripper 로직 스킵.

**해결**: 개별 finger actuator fallback 로직 추가. 현재 모든 vec env에 적용됨.

---

## 유용한 디버그 스크립트

| 스크립트 | 용도 |
|----------|------|
| `scripts/login/test_egl.py` | EGL 렌더링이 되는지 빠르게 확인 |
| `scripts/login/test_parallel_vecenv.py` | SubprocVecEnv 생성/reset/step 동작 확인 |
| `scripts/login/render_env_grids.py` | N개 환경의 초기 상태를 그리드 이미지로 렌더링 |
| `debug/render_pnp_grid.py` | PnP 30env 그리드 렌더링 (cube-bowl 겹침 확인용) |

---

## 로그인 노드 디버깅 제한

- GPU 1개만 사용 (DDP 테스트 시 최대 2개)
- 100 step 이하, 10분 이내
- 그 이상은 반드시 `sbatch`로 제출

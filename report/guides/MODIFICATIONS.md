# 코드 수정사항 (원본 대비)

> 마지막 업데이트: 2026-05-15

## 요약

공유 코드 (모든 task 영향)와 task-specific 코드를 구분합니다.

## 🌐 공유 코드 수정 (Committed)

### 1. ActionScaler 기능 추가 — `normalization.py`
- **커밋**: `1faf05b` feat: add ActionScaler mode for data-driven residual scaling
- **파일**: `resfit/rl_finetuning/utils/normalization.py`
- **설명**: 하드코딩된 물리 스케일 대신 offline data 기반으로 action 범위를 자동 계산
- **영향**: ActionScaler를 사용하는 모든 task

### 2. ActionScaler no_clamp 옵션 — `normalization.py`
- **커밋**: `6475377` ActionScaler: add --no_action_clamp to preserve base policy
- **파일**: `resfit/rl_finetuning/utils/normalization.py`
- **설명**: `no_clamp=True`일 때 scale/unscale이 순수 선형 변환. base policy action이 offline 범위 밖이어도 잘리지 않음
- **영향**: ActionScaler를 사용하는 모든 task

### 3. q_agent state_only 분기 — `q_agent.py`
- **커밋**: `205d297` Stack Cube: offline data, ActionScaler smoke test, sweep scripts, base eval
- **파일**: `resfit/rl_finetuning/off_policy/rl/q_agent.py` (line ~940)
- **설명**: `state_only=True`일 때 state dim == 10 assert 스킵 (Drawer는 state dim이 다름)
- **영향**: state_only=True인 모든 task

### 4. Replay Buffer 공간 불일치 수정
- **커밋**: `740f7cb` fix: store normalized combined action in replay for ActionScaler mode
- **파일**: `mujoco_residual_wrapper.py` (원본 Lift wrapper) — 다른 task wrapper에도 같은 패턴 반영 필요
- **설명**: replay에 residual만 저장 → combined (base+residual) 저장으로 변경. critic/actor 공간 일치

### 5. Eval object_state_dim 수정
- **커밋**: `bb63bb5`
- **설명**: eval QAgent 생성 시 object_state_dim 10→7

### 6. ActionScaler attribute access 수정
- **커밋**: `785b9fc`

### 7. ActionScaler eval subprocess 전달
- **커밋**: `c938641`

## 📁 Task-Specific 수정

### Cup (Stand Cup) — Committed

| 파일 | 변경 | 커밋 |
|------|------|------|
| `mujoco_residual_wrapper_cup.py` | **chunk_sync 구현** — `--chunk_sync` 시 16스텝마다 batch GR00T 호출, 나머지는 캐시 사용. `_get_base_actions_sync()`, `_get_current_pose()`, `_query_groot_batch()` 추가 | `26eced4` |
| `train_residual_td3_mujoco_cup.py` | `--chunk_sync` CLI arg 추가, wrapper에 전달 | `26eced4` |
| `eval_async_mujoco_cup.py` | `--chunk_sync` CLI arg 추가, wrapper에 전달 | `26eced4` |
| `slurm_train_cup_td3.sh` | `CUDA_VISIBLE_DEVICES=0` → `${SLURM_JOB_GPUS:-0}` (sub 파티션 GPU 충돌 수정) | `f3b20f6` |

### Drawer (Uncommitted)

| 파일 | 변경 | 상태 |
|------|------|------|
| `mujoco_residual_wrapper_drawer.py` | **Replay clamp 버그 수정** — pre-clamp → post-clamp 값 저장 | ⚠️ uncommitted |
| `eval_async_mujoco_drawer.py` | `--eval_stddev` arg 추가 | ⚠️ uncommitted |
| `train_residual_td3_mujoco_drawer.py` | `--eval_stddev` arg + eval subprocess에 전달 | ⚠️ uncommitted |
| `slurm_train_drawer_td3_d2only.sh` | `EVAL_STDDEV` 환경변수 지원 | ⚠️ uncommitted |

### Stack (Uncommitted)

| 파일 | 변경 | 상태 |
|------|------|------|
| `mujoco_residual_wrapper_stack.py` | `render_mode` "rl_only" → "none" (불필요 렌더링 제거) | ⚠️ uncommitted |
| `mujoco_vec_env_stack.py` | render_mode 관련 수정 | ⚠️ uncommitted |
| `eval_async_mujoco_stack.py` | 수정사항 있음 | ⚠️ uncommitted |
| `train_residual_td3_mujoco_stack.py` | ActionScaler 관련 수정 | ⚠️ uncommitted |
| `slurm_sweep_stack_*.sh` | Stack sweep scripts (4개) | ⚠️ uncommitted |

### PnP / Lift
- **수정 없음** — 원본 코드 그대로

## 🏗️ Unified Architecture (2026-05 ~)

### 커밋: `4660514` ~ `a816db8`

Task별 분산되어 있던 train/eval/wrapper를 통합 아키텍처로 리팩터링.

### 1. Unified Train/Eval Scripts
| 파일 | 설명 |
|------|------|
| `scripts/train_residual_td3_unified.py` | 모든 task (cup/pnp/lift/stack/drawer) 통합 학습 |
| `scripts/eval_async_unified.py` | 모든 task 통합 평가 |
| `scripts/eval_distill_debug.py` | Distillation 모델 디버그 평가 |
| `scripts/collect_offline_data_with_images.py` | Realistic 이미지 포함 offline 데이터 수집 |

### 2. Unified Wrapper — `mujoco_residual_wrapper_unified.py`
- **커밋**: `4660514`, `0e40b74`
- `use_images`: VecEnv 이미지를 obs에 포함 (distillation용)
- `realistic`: `RealisticImageRenderer`로 sim 이미지를 realistic 이미지로 교체
- `_inject_realistic_images()`: 매 step마다 realistic 렌더링 → obs dict 교체
- `reinit_realistic_renderer()`: fork 후 EGL 컨텍스트 재생성

### 3. Task Config Registry — `configs/task_configs.py`
- **커밋**: `4660514`
- 각 task의 `vec_env_module`, `vec_env_class`, `camera_keys`, `rl_image_keys` 등 중앙 관리
- Drawer: `camera_keys = ["back", "wrist"]` (다른 task와 통일)

## 🎨 Realistic Rendering Pipeline (2026-05 ~)

### 커밋: `0e40b74`, `72ad541`, `a816db8` + uncommitted

Sim 이미지를 실제 환경과 유사하게 렌더링하는 파이프라인. `scene_realistic.py` (공용 라이브러리)를 활용.

### 1. Realistic Renderer — `rendering/realistic_renderer.py`
- **커밋**: `0e40b74` (PnP/Lift/Stack), `a816db8` (Drawer)
- 별도 MuJoCo 모델(`make_model_generic()`)을 생성하여 realistic 렌더링
- VecEnv → realistic model로 qpos/qvel 복사 → `RealisticRenderHelper`로 렌더링
- Base cam: depth compositing + real background 합성
- Wrist cam: textured table + depth-based background replacement

### 2. Drawer Realistic 지원
- **커밋**: `a816db8` + uncommitted 수정

| 항목 | 내용 |
|------|------|
| Cabinet XML | `_build_drawer_config()` — `cabinet_placement.json`에서 동적 생성 |
| Camera calib | Interactive calib (`Mujoco_Franka_Drawer/config/camera_info.yaml`) |
| Background | `real_background_sam_fixed.png` (SAM으로 추출한 drawer 전용 배경) |
| Face texture | Homography 기반 `_overlay_face_textures()` — drawer face에 실제 나무 텍스처 오버레이 |
| Table | `table_visible=True` (wrist cam에서 직접 렌더링), size=0.55×0.45 |
| Cabinet placement | `--realistic` 시에만 `Mujoco_Franka_Drawer/output/cabinet_placement.json` 사용, 미사용 시 MSRA 기본값 유지 |

### 3. Camera Key 통일
- **커밋**: `a816db8`
- Drawer의 `cam_base`/`cam_wrist` → `back`/`wrist`로 통일 (다른 task와 동일)
- `CAMERA_MAP` 및 `_render_cameras()` 수정
- `_inject_realistic_images()`가 task 구분 없이 동작

### 4. 외부 에셋 경로

| 경로 | 용도 |
|------|------|
| `~/Repos/Intern/Mujoco_Franka/src/scene_realistic.py` | 공용 realistic 렌더링 라이브러리 |
| `~/Repos/Intern/Mujoco_Franka/src/assets/table_texture.png` | 테이블 텍스처 |
| `~/Repos/Intern/Mujoco_Franka_Drawer/config/camera_info.yaml` | Drawer interactive calib |
| `~/Repos/Intern/Mujoco_Franka_Drawer/output/cabinet_placement.json` | Realistic용 cabinet 위치 |
| `~/Repos/Intern/Mujoco_Franka_Drawer/output/real_background_sam_fixed.png` | Drawer 배경 |
| `~/Repos/Intern/Mujoco_Franka_Drawer/output/face_tex_d{3,4}.png` | Drawer face 텍스처 |

## 🧪 DAgger Distillation (2026-05 ~)

### 커밋: `69a7715`

State-only teacher → Image student distillation (DAgger 방식).

### 1. Distill Script — `scripts/distill_dagger.py`
- Teacher: state-only RL checkpoint (critic frozen)
- Student: ResNet18 image encoder → MLP policy
- DAgger: student rollout → teacher labels → supervised update
- `--realistic` 플래그로 realistic 이미지 사용 가능

### 2. Distill 결과 (Lift/PnP/Stack)
| Task | Teacher SR | Best Student SR | 비고 |
|------|-----------|----------------|------|
| PnP | 95% | 80% | 안정적 |
| Lift | 95% | 75% | 진동 |
| Stack | 75% | 65% | 높은 분산 |

## 🆕 새로 만든 파일 (Untracked)

| 파일 | 용도 |
|------|------|
| `scripts/standalone_eval_drawer.py` | Drawer 독립 eval script (correct horizon=16) |
| `scripts/sbatch_oneshot_eval.sh` | One-shot eval SLURM script |
| `scripts/quick_eval_noise.py` | Quick eval noise test (삭제 가능) |
| `scripts/slurm_sweep_stack_l2_{1,2,3,4}.sh` | Stack L2 sweep scripts |

## Git 상태

```bash
# Committed (pushed까지 된 것)
git log origin/mujoco..HEAD  # local only commits 확인

# Uncommitted 확인
git status --short

# 전체 diff 확인
git diff --stat
```

## ⚠️ 주의: Cross-Task 영향

1. **공유 코드 (`q_agent.py`, `normalization.py`)** 수정은 모든 task에 영향
2. **각 task wrapper/train/eval**은 task별 독립 — 다른 task에 영향 없음
3. Drawer의 replay clamp 수정이 Stack에도 필요한지 **미확인** (같은 패턴 코드가 Stack wrapper line 421-425에 존재)

## 🚀 성능 최적화 (Cup — 2026-05-03)

학습/수집/평가 모든 경로에 자동 적용됨 (vec_env + wrapper 레벨 변경).

### 1. 렌더링 스킵 (16스텝 중 15번)
- **파일**: `mujoco_residual_wrapper_cup.py` step(), `mujoco_vec_env_cup.py` _build_obs_dict()
- **내용**: GR00T 추론은 16스텝마다 1회. 나머지 15스텝에선 `render_mode="none"`으로 카메라 렌더링 스킵
- **효과**: 127ms/step → 68ms/step (**1.86x speedup**)
- **검증**: reward diff = 0.0 (이미지는 reward 계산에 무관)

### 2. obs 버퍼 프리얼로케이트
- **파일**: `mujoco_vec_env_cup.py` __init__, _build_obs_dict()
- **내용**: 매 스텝 `list → np.stack → torch` 할당 대신 `__init__`에서 버퍼 사전 할당, in-place 쓰기
- **효과**: GC 압력 감소, 메모리 할당 오버헤드 제거

### 3. 렌더링 `.copy()` 제거
- **파일**: `mujoco_vec_env_cup.py` _render_cameras_inplace()
- **내용**: 렌더러 출력을 복사 없이 직접 버퍼에 기록
- **효과**: 렌더링 스텝에서 불필요한 복사 제거

### 4. uprightness 직접 계산
- **파일**: `mujoco_vec_env_cup.py` _get_uprightness()
- **내용**: `Rotation.from_quat().as_matrix()` 대신 `1 - 2(x² + y²)` 직접 계산
- **효과**: reward/success 체크 시 Rotation 객체 생성 오버헤드 제거 (diff < 2e-15)

### 5. _combine_actions / _augment_obs 벡터화
- **파일**: `mujoco_residual_wrapper_cup.py`
- **내용**: per-env Python 루프 → batch `Rotation.from_quat/euler` 한 번 호출
- **효과**: env 수가 많을수록 효과 커짐 (27 envs 기준 유의미)

### 6. grasp 탐지 최적화
- **파일**: `mujoco_vec_env_cup.py` _update_grasp()
- **내용**: `list.index()` O(n) → `set` O(1) lookup, bilateral 확인 시 early break
- **효과**: contact 수가 많을 때 루프 조기 종료

### 7. SubprocVecEnv 병렬 물리 스테핑
- **파일**: `mujoco_vec_env_cup.py` _cup_env_worker_loop(), __init__()
- **내용**: `multiprocessing.Pipe` + `spawn` 컨텍스트로 워커 프로세스 분산 물리 시뮬레이션
- **기본값**: `parallel_envs=True, num_workers=8`
- **효과**: 27 envs 기준 154.8ms/step → 29.3ms/step (**5.29x speedup**)
- **검증**: reward diff = 0.0 (exact match)
- **주의**: 워커에서 `MUJOCO_GL=osmesa` 설정 (EGL 충돌 방지), `spawn` 컨텍스트라 `if __name__ == "__main__"` 가드 필요

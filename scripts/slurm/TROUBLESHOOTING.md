# SLURM Troubleshooting Guide

Worker 노드에서 발생하는 에러와 해결법 모음.

---

## CUDA not available

**증상**: `torch.cuda.is_available()` → `False`, 또는 `RuntimeError: No CUDA GPUs are available`

**원인**: SLURM cgroup이 GPU 1개만 노출하지만 `CUDA_VISIBLE_DEVICES`가 물리 인덱스(e.g. 7)로 설정됨. torch는 GPU 7을 찾으려 하지만 cgroup 안에는 index 0만 보임.

**해결**:
```bash
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)
```

---

## cuDNN SDPA: libnvrtc == nullptr

**증상**: `RuntimeError: cuDNN error ... libnvrtc == nullptr`

**원인**: Worker에 nvrtc (JIT 컴파일러)가 설치되어 있지 않아 cuDNN SDPA 백엔드가 실패.

**해결**:
```python
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)
```

---

## source: not found (--wrap)

**증상**: `source: not found` 에러

**원인**: `sbatch --wrap="..."` 는 `/bin/sh`을 사용. `source`는 bash 전용.

**해결**: `. /path/to/activate` 사용하거나 별도 `#!/bin/bash` 스크립트 파일 작성.

---

## CUDA 13.0 라이브러리 없음

**증상**: `libcudart.so.13: cannot open shared object file`

**원인**: Worker에 `/usr/local/cuda-13.0/` 미설치. SLURM이 자동으로 LD_LIBRARY_PATH에 추가하지만 실제 파일 없음.

**해결**: PyTorch pip 패키지에 bundled된 nvidia 라이브러리 경로를 LD_LIBRARY_PATH에 추가:
```bash
NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:${LD_LIBRARY_PATH:-}
```

---

## libstdc++ CXXABI_1.3.15 없음

**증상**: `ImportError: ... version 'CXXABI_1.3.15' not found`

**원인**: 시스템 libstdc++가 오래됨.

**해결**: miniconda3에서 최신 libstdc++ 복사:
```bash
cp ~/miniconda3/lib/libstdc++.so.6* ~/lib-compat/
export LD_LIBRARY_PATH=$HOME/lib-compat:${LD_LIBRARY_PATH:-}
```

---

## NAS hang on replay_buffer.dumps()

**증상**: `ReplayBuffer.dumps()` 호출 시 무한 대기 (NAS I/O 병목)

**원인**: 500K 용량의 LazyTensorStorage를 NAS에 한번에 직렬화.

**해결**: 버퍼 사이즈 줄이거나, 저장 시 `/tmp`(로컬 SSD) 사용 후 NAS로 복사.

---

## EGL_BAD_ACCESS (RL Training 크래시)

**증상**: Critic warmup 후 `env.reset()` 또는 training loop 중 `renderer.render()` 에서:
```
EGLError(err = EGL_BAD_ACCESS, baseOperation = eglMakeCurrent, ...)
```

**원인**: `async_prefetch=True` (기본값)이 background thread에서 GR00T render+inference를 실행.
EGL context는 **thread-affine** — 한 thread에서 `eglMakeCurrent()` 하면 다른 thread에서 같은 display 접근 시 `EGL_BAD_ACCESS`.

추가로 critic warmup의 heavy CUDA 연산 (CUBLAS matmul, backprop)이 EGL display state를 corrupt할 수 있음.

**해결** (commit `a56017a`, 2026-05-06):
```python
# train_residual_td3_unified.py & eval_async_unified.py:
env = MuJoCoResidualWrapperUnified(..., async_prefetch=False)

# critic warmup 후 (train script에 이미 적용):
env.vec_env.rebuild_renderers()  # 새로운 EGL context 생성
```

**SLURM script**:
```bash
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}  # GPU 매칭
```

**확인 방법**: 3분 이내 crash → `async_prefetch` 문제 의심. Job 로그에 `Rebuilding EGL renderers` 출력 후에도 crash → 다른 원인.

**이력**: Job 8560 (성공)에는 `async_prefetch` 기능 자체가 없었음. 이후 추가되면서 모든 RL training이 crash.

---

## SLURM 환경 변수 템플릿

모든 SLURM 잡에 필요한 최소 환경 설정:
```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --mem=80G --cpus-per-task=8

. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NV=$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$HOME/.local/lib:$HOME/lib-compat:$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib:$NV/cuda_nvrtc/lib:$NV/nccl/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)

# 워크로드 실행
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
python3 scripts/workloads/your_script.py
```

---

## Lift Base Eval SR=0% (큐브 위치 불일치)

**증상**: Lift base eval에서 GR00T가 gripper close를 전혀 예측하지 않음. 모든 스텝에서 `raw_grip ≈ 0.99` (fully open). SR=0%.

**원인**: `MuJoCoVecEnv` 기본 큐브 위치 `[0.45, -0.05, 0.02]`가 학습 데이터(Lift_32ep_sim)의 큐브 위치와 다름. GR00T에 입력되는 카메라 이미지에 큐브가 안 보이거나 다른 위치에 있어서, 학습 시 본 observation 분포와 맞지 않음.

**검증**: 데이터셋 observation을 GR00T에 직접 넣으면 close 정상 출력 → 모델 자체는 정상, 입력 분포 문제.

**해결**: 학습 데이터 수집 시 사용한 큐브 위치를 로드:
```python
vec_env = MuJoCoVecEnv(..., cube_positions=cube_positions)
```
```bash
python scripts/eval_base_lift.py --cube_positions_json configs/lift_66ep_positions.json
```
위치 적용 후 reward 60x 증가 (130 → 7700+), GR00T가 close 예측 시작.

---

## Lift Gripper Close 신호 너무 짧음 (Latch 필요)

**증상**: 큐브 위치 수정 후 GR00T가 close를 예측하지만, 1 action chunk (16스텝) 동안만 close → 다음 chunk에서 즉시 open (0.99) → 큐브를 잡았다 바로 놓음. lift=0.02m 정도에서 drop.

**원인**: GR00T action chunk (horizon=16)에서 close 구간이 짧음. Lift 데이터셋의 close ratio가 27%로 낮아서(PnP는 46%), close를 유지하는 패턴을 충분히 학습하지 못함. chunk 경계에서 grip 값이 0.99로 복귀.

**해결**: gripper latch 활성화. 한번 close하면 일정 시간 유지:
```python
# mujoco_residual_wrapper_unified.py
_GRIP_CLOSE_LATCH_THRESH = 0.8    # raw_grip < 0.8 → latch ON
_GRIP_OPEN_LATCH_THRESH = 0.95   # raw_grip > 0.95 for N steps → unlatch
_GRIP_LATCH_OPEN_STEPS = 150     # 150스텝 연속 open이어야 unlatch

env = MuJoCoResidualWrapperUnified(..., use_gripper_latch=True)
```
⚠️ 위 threshold는 Lift용 (grip 범위 0~1). Cup (grip 범위 0~0.04)에서는 원래 값 (0.015/0.035) 사용. 현재 module-level 상수이므로 task 전환 시 주의.

**결과**: SR 0% → 60% (20 env 기준).

---

## Eval 스크립트 Success 미감지 (terminated vs info["success"])

**증상**: env가 성공적으로 lift하고 auto-reset까지 되지만, eval 스크립트에서 SR=0%으로 집계됨. 로그에서 `G lift=0.040` 후 다음 스텝에서 `- lift=0.000` (reset됨).

**원인**: eval 스크립트가 `info["success"]`를 체크했지만, `MuJoCoResidualWrapperUnified`는 해당 키를 info에 넣지 않음. success 시 `terminated=True`를 반환하고 자동으로 env를 reset (line 293: `self.vec_env.reset_envs(done_ids)`).

**해결**: `terminated=True`로 success 감지:
```python
# 잘못된 코드
if "success" in info:
    if info["success"][i]: ep_successes[i] = True

# 올바른 코드
if terminated[i]: ep_successes[i] = True
```

**참고**: `terminated=True`는 `_is_success()` (grasped + lift ≥ 4cm)에서만 발생. `truncated=True`는 max_episode_steps 도달 시 (실패).

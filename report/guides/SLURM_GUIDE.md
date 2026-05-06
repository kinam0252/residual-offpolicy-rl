# SLURM 사용 가이드

> 마지막 업데이트: 2026-05-04

## 파티션 및 QoS

SLURM은 `--partition` + `--qos`를 함께 사용합니다.

| QoS | partition | qos | GPU 제한 | 용도 |
|-----|-----------|-----|---------|------|
| **core-own** | `--partition=core` | `--qos=core-own` | 8 GPU | 주 학습용 (우선순위 높음) |
| **core-on-sub** | `--partition=sub` | `--qos=core-on-sub` | 별도 | 추가 학습 |
| **core-extra** | `--partition=core` | `--qos=core-extra` | 별도 | 여유 GPU 활용 |

### 현재 사용량 확인
```bash
# 내 잡 목록
squeue -u kinamkim

# GPU 사용량 요약 (잡 수)
squeue -u kinamkim | wc -l

# 특정 노드 상태
sinfo -N -l | grep worker-
```

## 기본 SLURM 스크립트 구조

```bash
#!/bin/bash
#SBATCH --job-name=JOBNAME
#SBATCH --partition=core           # core (own/extra) 또는 sub (on-sub)
#SBATCH --qos=core-own             # core-own / core-extra / core-on-sub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/%x_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/%x_%j.err
#SBATCH --nodelist=worker-N        # 특정 노드 지정 시

export PYTHONUNBUFFERED=1          # 필수! 없으면 stdout 버퍼링됨
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}  # 필수! sub 파티션 GPU 충돌 방지
export WANDB_MODE=online           # online / offline / disabled

# 경로 설정
GROOT_CKPT=~/DATA/INTERN/training/groot_{task}_sim_{ep}ep/checkpoint-100000
OFFLINE_DIR=outputs/offline_{task}_{details}

cd ~/Repos/Intern/residual-offpolicy-rl

python resfit/rl_finetuning/scripts/train_residual_td3_mujoco_{task}.py \
    --groot_checkpoint $GROOT_CKPT \
    --offline_data_dir $OFFLINE_DIR \
    --num_envs N \
    --action_scale 0.1 \
    --action_l2_reg 10.0 \
    --total_timesteps 50000 \
    ... (task별 파라미터)
```

## Task별 제출 예시

### Drawer (D2-only)
```bash
sbatch scripts/slurm_train_drawer_td3_d2only.sh
```
- 주요 환경변수: `AS` (action_scale), `L2` (L2_reg), `EVAL_STDDEV`, `REWARD` (reward_type), `CW` (critic_warmup)
- SLURM default: action_scale=0.1, **L2=0.0**, reward=**dense**
- argparse default과 다름에 주의 (argparse L2=10.0)

### Stack Cube
```bash
sbatch scripts/slurm_sweep_stack_r05a1.sh  # 예시
```
- Stack은 argparse default가 PnP와 다름에 주의 (LR=3e-4, L2=1.0 등)
- num_envs=30으로 큰 값 사용

### PnP
```bash
sbatch scripts/slurm_pnp_rl_v2.sh
```
- difficulty별 offline data 경로가 다름

## 주의사항

### PYTHONUNBUFFERED=1 필수
Python stdout은 SLURM 파일 리다이렉트 시 fully buffered됨. 이것 없으면 로그가 실시간으로 안 찍힘.

### /tmp는 worker별 독립
Login node의 `/tmp/`와 worker node의 `/tmp/`는 다름. NAS 경로 사용할 것.

### Worker별 GPU 배정 ⚠️ 중요

> **마지막 검증: 2026-05-04** (worker-2, 9, 10에서 실측)

**SLURM은 batch script에 `CUDA_VISIBLE_DEVICES`를 자동 설정하지 않습니다.**
대신 `SLURM_JOB_GPUS` 환경변수에 할당된 물리 GPU 인덱스가 들어있습니다.

```bash
# 올바른 패턴 (모든 sbatch script에 필수)
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}
```

#### core vs sub 파티션 GPU 격리 차이

| | core 파티션 | sub 파티션 |
|---|---|---|
| **하드웨어 격리** | ✅ 있음 (할당된 GPU만 보임, index 0으로 리매핑) | ❌ 없음 (같은 유저의 모든 할당 GPU가 물리 인덱스 그대로 보임) |
| **유저 간 격리** | ✅ 다른 유저 GPU 안 보임 | ✅ 다른 유저 GPU 안 보임 |
| **CUDA_VISIBLE_DEVICES 미설정 시** | `cuda:0` = 할당된 GPU (안전) | `cuda:0` = 물리 GPU 0 (충돌 가능!) |

**sub 파티션에서 `CUDA_VISIBLE_DEVICES=0` 하드코딩하면?**
→ 같은 노드의 내 잡들이 전부 물리 GPU 0에 몰림 → GPU 충돌 + 30-40% 성능 저하
→ 할당된 다른 GPU(1,2,...)는 완전히 유휴 상태

**검증 방법** (잡 실행 중):
```bash
# SLURM이 할당한 GPU 확인
scontrol show job $JOBID -d | grep "GRES=gpu"  # IDX:N 값 확인

# 실제 프로세스가 쓰는 GPU 확인
srun --jobid=$JOBID --overlap nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

> **⚠️ 클러스터 가이드(`~/.codex/AGENTS.md`)의 §10과 다릅니다.**
> 클러스터 가이드는 "CUDA_VISIBLE_DEVICES is set by Slurm — do not override"라고 하지만,
> 실측 결과 batch script에는 설정되지 않습니다. 반드시 `${SLURM_JOB_GPUS:-0}`을 사용하세요.
> (클러스터 가이드는 root 소유 읽기전용이라 수정 불가)

### Login Node에서 Quick Eval
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/standalone_eval_drawer.py \
    outputs/.../best.pt --episodes 10 --action_scale 0.05
```
- GPU 부하가 적은 시간에만 사용
- 다른 사용자에게 영향 줄 수 있으므로 짧게만

### 잡 관리
```bash
# 잡 취소
scancel JOB_ID

# 특정 노드의 잡만 보기
squeue -u kinamkim -w worker-4

# 잡 상세 정보
scontrol show job JOB_ID
```

### Stand Cup — 배치 오프라인 수집
```bash
# 8개 array job으로 병렬 수집 (core-on-sub)
sbatch scripts/cup_collect_batch.sh
```
- **스크립트**: `scripts/cup_collect_batch.sh`
- **array=0-7**: 8잡 × 13 ep/env × 27 positions = 총 2,808 episodes
- **출력**: `outputs/offline_cup_batch/chunk_{0..7}/`
- **GR00T**: `~/DATA/INTERN/training/groot_cup_sim_27ep/checkpoint-200000`
- **최적화 적용**: SubprocVecEnv(8 workers) + render-skip(16스텝 중 15번) 자동 적용
- **주의**: `python3` 사용 필수 (`python`은 worker에서 없음), `six` 모듈이 `~/.local`에 설치돼있어야 함
- **수집 후 merge**: 학습 전 chunk들을 하나로 합쳐야 함 (또는 학습 스크립트에서 다중 디렉토리 지원)

### Stand Cup — RL 학습
```bash
# 기본 제출 (core)
sbatch scripts/slurm_train_cup_td3.sh

# chunk_sync 모드 (GR00T batch inference, ~2x 속도 향상)
sbatch scripts/slurm_train_cup_td3.sh --chunk_sync

# sub 파티션으로 제출
sbatch --partition=sub --qos=core-on-sub scripts/slurm_train_cup_td3.sh --chunk_sync

# 하이퍼파라미터 오버라이드
ACTION_SCALE=0.05 ACTION_L2=0.0 sbatch scripts/slurm_train_cup_td3.sh --chunk_sync
```
- 학습 스크립트: `resfit/rl_finetuning/scripts/train_residual_td3_mujoco_cup.py`
- offline data: `outputs/offline_cup_batch` (배치 수집 결과)
- reward_type: dense (approach + uprightness + contact grasp)
- `--chunk_sync`: 16스텝마다 batch GR00T 호출 (env_step ~415ms 일정 vs staggered ~440→960ms 증가)
- `"$@"` 지원: sbatch 뒤에 추가 인자를 붙이면 python 스크립트에 전달됨

## wandb 설정

- **online 모드**: 실시간 모니터링 가능 (권장)
- **offline 모드**: 네트워크 불안정할 때
- 프로젝트명: wandb config에서 설정 (보통 train script의 argparse로)
- API로 메트릭 조회 시 키 이름이 `train/train/...` 형태 (이중 prefix 주의)

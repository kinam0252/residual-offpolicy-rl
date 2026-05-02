# SLURM 사용 가이드

> 마지막 업데이트: 2026-05-02

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

### Worker별 GPU 배정
SLURM이 GPU를 할당하면 `CUDA_VISIBLE_DEVICES`가 자동 설정됨. 코드에서 `cuda:0`을 쓰면 할당된 GPU를 자동으로 사용.

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

## wandb 설정

- **online 모드**: 실시간 모니터링 가능 (권장)
- **offline 모드**: 네트워크 불안정할 때
- 프로젝트명: wandb config에서 설정 (보통 train script의 argparse로)
- API로 메트릭 조회 시 키 이름이 `train/train/...` 형태 (이중 prefix 주의)

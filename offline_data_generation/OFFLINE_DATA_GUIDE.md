# Offline Data Generation Guide (CSV → LeRobot Format)

## Overview

Isaac Lab 시뮬레이터에서 CSV 데모 데이터를 replay하여  
LeRobot format의 offline RL 학습 데이터를 생성합니다.

### 생성되는 데이터 구조
```
output_dir/
├── data/chunk-000/
│   ├── episode_000000.parquet   # 에피소드별: state, action, reward, vlm_latent
│   ├── episode_000001.parquet
│   └── ...
├── videos/chunk-000/
│   ├── image/episode_000000.mp4        # front camera
│   ├── right_image/episode_000000.mp4  # right camera
│   └── wrist_image/episode_000000.mp4  # wrist camera
└── worker_*_summary.json
```

### Parquet 컬럼
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `state` | float32[10] | EEF pos(3) + quat_xyzw(4) + gripper(2) + contact_force(1) |
| `actions` | float32[7] | normalized delta action (pos/0.02, rot/0.2, gripper*2-1) |
| `vlm_latent` | float32[2048] | GR00T backbone mean-pooled features |
| `reward` | float32 | dense_clipped reward [0, 1] |
| `cube_height` | float32 | 큐브 lift 높이 (meters) |
| `contact_force` | float32 | 손가락 접촉력 (N) |
| `has_contact` | float32 | 접촉 유무 (0/1, threshold 0.1N) |
| `finger_cube_dist` | float32 | 손가락↔큐브 거리 (meters) |

---

## 필요 파일

| 항목 | 경로 예시 |
|------|---------|
| **CSV 데모 데이터** | `Data/pickMushroom/pickMushroom_*` (174개 폴더) |
| **GR00T checkpoint** | `Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000/` |
| **코드** | `residual-offpolicy-rl/workspace/replay_all_worker_with_groot.py` |
| | `residual-offpolicy-rl/resfit/rl_finetuning/wrappers/iface_env_wrapper.py` (scene config 참조) |
| | `Isaac-GR00T/` (gr00t.policy 패키지) |
| **Docker/환경** | Isaac Sim 4.5.0 + Isaac Lab 2.0.1 컨테이너 |

### CSV 데모 데이터 구조
각 `pickMushroom_*` 폴더에:
```
franka_joint_states.csv      # 로봇 관절 상태 (7 DOF)
gripper_joint_states.csv     # 그리퍼 상태
isaac_deltaEEF_gripper.csv   # delta EEF action (6 DOF + gripper)
isaac_absEEF_gripper_convert314.csv  # (optional)
object_pos.csv               # 큐브 초기 위치 [x, y, z, yaw]
```

---

## 실행 방법

### 방법 1: VLM latent 포함 (권장, v21 학습용)

```bash
CONTAINER="isaaclab_resfit"
N_WORKERS=4
CSV_BASE="/path/to/Data/pickMushroom/pickMushroom_20251209_081825_758"
OUTPUT_DIR="/path/to/Data/lerobot/pickMushroom_train_dense_clipped_3cm_vlm"
GROOT_MODEL="/path/to/Isaac-GR00T/outputs/pickMushroom_train/checkpoint-100000"
WORKER_SCRIPT="/path/to/residual-offpolicy-rl/workspace/replay_all_worker_with_groot.py"

for WID in $(seq 0 $((N_WORKERS - 1))); do
  LAUNCHER="/tmp/_worker_vlm_${WID}.sh"
  cat > "$LAUNCHER" << EOF
#!/usr/bin/env bash
set -e
export TERM=xterm PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace/isaaclab/workspace/.pydeps_resfit_train:/path/to/residual-offpolicy-rl:/path/to/Isaac-GR00T"
cd /workspace/isaaclab
exec ./isaaclab.sh -p $WORKER_SCRIPT \\
  --headless --enable_cameras --num_envs 1 \\
  --csv_base_dir "$CSV_BASE" \\
  --output_dir "$OUTPUT_DIR" \\
  --groot_model_path "$GROOT_MODEL" \\
  --groot_embodiment_tag new_embodiment \\
  --groot_policy_device cuda:0 \\
  --language_override "pick up mushroom" \\
  --worker_id $WID --num_workers $N_WORKERS \\
  --friction 50000 \\
  --success_threshold 0.03 \\
  --reward_type dense_clipped \\
  --vlm_inference_interval 16
EOF
  chmod +x "$LAUNCHER"
  docker exec -d "$CONTAINER" bash -c "bash '$LAUNCHER' > '$OUTPUT_DIR/worker_${WID}.log' 2>&1"
  echo "Worker $WID launched"
done
```

### 방법 2: VLM latent 없이 (v16 학습용, 빠름)

`run_replay_all.sh` 사용:
```bash
WORKER_SCRIPT="/path/to/residual-offpolicy-rl/workspace/replay_all_worker.py"
# replay_all_worker_with_groot.py 대신 replay_all_worker.py 사용
# --groot_model_path 등 groot 관련 인자 불필요
```

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--num_workers` | 4 | 병렬 Isaac Sim 인스턴스 수 (GPU 메모리에 따라) |
| `--friction` | 50000 | 로봇/큐브 마찰 계수 (높을수록 잡기 쉬움) |
| `--success_threshold` | 0.03 | 성공 기준 큐브 높이 (3cm) |
| `--reward_type` | dense_clipped | sparse / dense / dense_clipped |
| `--vlm_inference_interval` | 16 | VLM latent 추출 간격 (CSV row 단위) |

### Dense Clipped Reward 계산 (reward_type=dense_clipped)
```
distance_reward = (1 - tanh(dist/0.1)) * 0.1     # 손가락↔큐브 거리
contact_reward  = grasp_gate * 0.2                # 잡기 성공
height_reward   = tanh(h/0.1) * 0.5 * grasp_gate  # 높이 비례
success_reward  = (h >= 3cm) * 1.0 * grasp_gate   # 성공 보너스
reward = clamp(sum, 0, 1)

grasp_gate = (양손 < 5cm) AND (그리퍼 닫힘) AND (접촉력 > 0.1N)
```

---

## 핵심 코드 파일

### 1. `replay_all_worker_with_groot.py` (메인 워커)
- Isaac Lab 시뮬레이터에서 CSV 궤적 replay
- 매 step: joint position target 설정 → sim step → reward 계산
- 매 `vlm_inference_interval` step: GR00T inference → backbone_features mean pool → 2048D
- 에피소드 단위로 parquet + video 저장

### 2. `replay_all_worker.py` (VLM 없는 버전)
- `replay_all_worker_with_groot.py`와 동일하되 GR00T 관련 코드 없음
- vlm_latent 컬럼 생성 안 함

### 3. `replay_all_csv_with_reward.py` (단일 프로세스 버전)
- Worker 분할 없이 모든 에피소드를 순차 처리
- 디버깅/소규모 테스트용

### 4. `iface_env_wrapper.py` (scene config)
- `_make_scene_cfg()` 함수: Isaac Lab scene 구성 (로봇, 큐브, 카메라, 접촉센서)
- `_yaw_to_quat_wxyz()`: yaw angle → quaternion 변환
- 워커에서 import하여 사용

---

## 모니터링

```bash
# 개별 워커 로그
tail -f $OUTPUT_DIR/worker_0.log

# 전체 진행 확인
for i in $(seq 0 $((N_WORKERS - 1))); do
  echo "=== W$i ===" && tail -1 $OUTPUT_DIR/worker_${i}.log
done

# 완료된 에피소드 수
ls $OUTPUT_DIR/data/chunk-000/*.parquet | wc -l

# 전체 종료
docker exec $CONTAINER pkill -9 -f replay_all_worker
```

---

## 예상 시간

| 설정 | 174 episodes |
|------|-------------|
| 4 workers, no VLM | ~30분 |
| 4 workers, VLM | ~2시간 |
| 10 workers, VLM | ~50분 (GPU 메모리 주의) |

---

## 주의사항

1. **GPU 메모리**: 각 worker가 ~7GB 사용 (Isaac Sim + GR00T). A100 80GB에서 최대 ~10 worker
2. **마찰 계수**: `friction=50000`으로 높게 설정해야 CSV replay 시 큐브가 미끄러지지 않음
3. **CSV row → sim step**: 1 CSV row = 5 sim steps (smoothing), `max_joint_step=0.05`로 제한
4. **VLM 캐싱**: `vlm_inference_interval=16`이면 16 CSV row마다 GR00T inference 1회. 중간 step은 마지막 결과 재사용
5. **에피소드 길이**: 성공(cube_height >= threshold) 시 조기 종료

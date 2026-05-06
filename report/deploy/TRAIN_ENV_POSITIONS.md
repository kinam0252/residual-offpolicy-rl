# Training Environment Positions

각 태스크의 RL 학습 시 환경 포지션 설정.

## Per-Task Summary

| Task | Train Envs | Position Source | Assignment 방식 |
|------|------------|-----------------|-----------------|
| **Cup (Stand Cup)** | 27 | `configs/cup_positions.json` (27개) | 각 env = 고정 1포지션 |
| **PnP (Pick & Place)** | 30 | `configs/pnp_train_30.json` (30개) | reset 시 풀에서 랜덤 선택 |
| **Stack (Stack Cube)** | 30 | `configs/stack_train_30.json` (30개) | env에 순차 할당 (env0=pos0, ...) |
| **Lift (Cube Lift)** | 30 | 기본위치 + `random_cube_range` | 고정 위치 기반 랜덤 perturbation |
| **Drawer** | 1 | 고정 (XML 기본 위치) | 단일 env, 변화 없음 |

## 상세

### Cup
- 원본: `MSRA/stand_cup/cup_positions.json` → `configs/cup_positions.json`
- 27개 에피소드 데모에서 추출한 cup position + orientation(quaternion)
- 각 env가 1개 고정 포지션 담당 (27 envs = 27 positions)
- Eval도 동일 27개 포지션 사용

### PnP
- 원본: `configs/pnp_66ep_positions.json` (66개, cube_pos + bowl_pos + cube_yaw_deg)
- Train: `configs/pnp_train_30.json` (앞 30개 추출)
- Eval: `configs/pnp_eval_20.json` (앞 20개 추출, train subset)
- 학습 시 reset마다 풀에서 **랜덤** 선택 (순차 아님)
- cube-bowl XY 거리: 0.20~0.46m (baseline 0.085m 대비 훨씬 어려움)

### Stack
- 원본: `configs/stack_cube_positions.json` (66개, white_cube_pos + green_cube_pos + yaw)
- Train: `configs/stack_train_30.json` (앞 30개 추출)
- Eval: `configs/stack_eval_20.json` (앞 20개 추출)
- 학습 시 env에 **순차 할당**: env 0 = position 0, env 1 = position 1, ...
- white-green XY 거리: 0.16~0.50m, mean 0.33m

### Lift
- 기본 위치: `[0.45, -0.05, 0.02]`
- `random_cube_range`: `{"dx":[-12,16]cm, "dy":[-20,20]cm, "yaw":[0,0]}`
- 각 reset 시 기본 위치에서 랜덤 perturbation 적용
- Position file 미사용, 실시간 랜덤화
- Eval: `configs/eval_table_hard_v2.json` (고정 perturbation 테이블)

### Drawer
- 고정 위치 (MuJoCo XML scene 기본값)
- 단일 env (num_envs=1)
- 포지션 다양성 없음

## Position 파일 형식

```json
// Cup: configs/cup_positions.json
[{"episode": 0, "cup_position": [x,y,z], "cup_orientation_wxyz": [w,x,y,z]}, ...]

// PnP: configs/pnp_train_30.json
[{"cube_pos": [x,y,z], "bowl_pos": [x,y,z], "cube_yaw_deg": float, "src_episode": int}, ...]

// Stack: configs/stack_train_30.json
[{"episode": 0, "white_cube_pos": [x,y,z], "green_cube_pos": [x,y,z], "white_yaw_deg": float, "green_yaw_deg": float}, ...]
```

## 원본 데이터 출처

모든 position 파일은 실제 로봇 데모 에피소드 녹화에서 초기 오브젝트 위치를 추출한 것:
- 66ep 데이터: PnP, Stack, Lift 공통 (동일 66 에피소드 세트)
- 27ep 데이터: Cup 전용
- 33ep 데이터: Drawer, PnP sim subset

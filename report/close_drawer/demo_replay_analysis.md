# Demo Replay 분석 결과

**날짜**: 2026-05-01
**스크립트**: `resfit/rl_finetuning/scripts/debug_demo_replay_full.py`
**데이터**: `~/DATA/INTERN/datasets/CloseDrawer_sim_33ep/data/chunk-000/` (33 episodes)

## 핵심 발견

### 1. active_drawer 전환 버그 수정
- `env._active_drawers[0] = d` 만으로는 부족 → `env["active_drawer"]`, `drawer_z_min/max`도 같이 업데이트 필요
- 이전 결과(D3=17%, D4=0%)는 항상 같은 서랍(D2)만 열린 상태였음
- 수정 후: 33/33 = 100% 성공 (no_gate)

### 2. DEMO_CONTACT_MEAN 업데이트
기존 값은 GR00T policy eval 기준 → 실제 demo parquet replay로 재측정

| 서랍 | OLD y_norm | NEW y_norm | OLD z_norm | NEW z_norm |
|------|-----------|-----------|-----------|-----------|
| D2   | -0.0150   | **-0.0795** | 0.2750   | **0.9794** |
| D3   | -0.0450   | **-0.1100** | 0.2680   | **0.9844** |
| D4   | -0.0690   | **-0.1518** | -0.2620  | **0.9713** |

z_norm 차이가 0.7~1.2로 매우 큼. 특히 D4는 부호까지 반대(-0.262 → +0.971).

### 3. Threshold=1.0 데모 성공률

**gate_1.0 (DEMO_CONTACT_MEAN 업데이트 후):**

| 서랍 | SR | closed_mean | ndist p50 | ndist p90 | ≤1.0 비율 |
|------|-----|------------|-----------|-----------|----------|
| D2   | 10/10=100% | 91.7% | 0.28 | 0.78 | 93.6% |
| D3   | 12/12=100% | 91.6% | 0.35 | 0.94 | 90.4% |
| D4   | 11/11=100% | 91.7% | 0.32 | 1.17 | 85.2% |
| **Total** | **33/33=100%** | **91.7%** | | | |

### 결론
- **contact_threshold=1.0** + 새 DEMO_CONTACT_MEAN으로 데모 33개 전부 성공
- 이 설정으로 residual RL 학습 진행

## 상세 JSON
- `outputs/demo_replay/demo_replay_final_thr1.0.json`

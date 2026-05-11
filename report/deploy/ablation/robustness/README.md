# Robustness Evaluation Results (Simulation)

## Overview

> ⚠️ **본 실험은 MuJoCo 시뮬레이션 환경에서 수행된 결과입니다.**
> 실제 로봇 배포(real-world deploy) 환경과는 차이가 있을 수 있습니다.

Object state observation에 noise/dropout augmentation을 적용해 학습한 체크포인트가
실제로 noisy observation에 더 강건한지 평가하는 실험.

**환경:** MuJoCo simulation (sim-to-sim robustness test)

**평가 조건:**
- Noise: `noise_max=0.01` (uniform), `dropout_prob=0.1`
- 각 체크포인트를 2가지 조건에서 평가: Clean (augmentation 없음) vs Noisy (noise + dropout)
- 60 episodes per condition (20 envs × 3 episodes)
- SLURM Jobs: 1260-1267

**체크포인트:**
- 4가지 학습 조건의 best.pt 사용 (ablation sweep에서 학습)
- PNP: jobs 747-750, Stack: jobs 763-766
- ⚠️ **학습 미완료 체크포인트** — PNP ~80%, Stack ~48% 진행 시점에서 중단

---

## Results

### PNP (Pick-and-Place)

| 학습 조건 | Clean SR | Noisy SR | SR Drop | 비고 |
|----------|----------|----------|---------|------|
| clean    | 36.67%   | 58.33%   | **-21.67%** | noisy에서 오히려 상승 |
| noise    | 46.67%   | 43.33%   | 3.33%   | 가장 안정적 drop |
| drop     | 33.33%   | 48.33%   | **-15.00%** | noisy에서 오히려 상승 |
| **both** | **53.33%** | 45.00% | 8.33%   | **clean SR 최고** |

### Stack

| 학습 조건 | Clean SR | Noisy SR | SR Drop | 비고 |
|----------|----------|----------|---------|------|
| clean    | **55.00%** | 46.67% | 8.33%   | 가장 큰 drop |
| noise    | 45.00%   | 50.00%  | -5.00%  | noisy에서 오히려 상승 |
| drop     | 50.00%   | 55.00%  | -5.00%  | noisy에서 오히려 상승 |
| **both** | **60.00%** | **58.33%** | **1.67%** | **최고 성능 + 최소 drop** |

---

## Analysis

### 1. "both" 체크포인트가 전반적으로 가장 우수
- PNP: clean SR 53.33% (1위)
- Stack: clean SR 60.00% (1위), noisy SR 58.33% (1위), SR drop 1.67% (최소)
- noise+dropout 동시 augmentation이 regularization 효과 → 일반화 성능 향상

### 2. Noisy eval에서 오히려 SR 상승하는 현상
- PNP clean/drop, Stack noise/drop에서 noisy SR > clean SR
- 가능한 원인:
  - **60 episodes의 높은 variance** — 95% CI ≈ ±12% (binomial)
  - **Stochasticity가 exploration 역할** — noise가 local optima 탈출 도움
  - **Dropout이 일부 noisy feature 차단** — 잘못된 object state 정보 무시

### 3. 통계적 유의성 부족
- 60 episodes × binary outcome → standard error ≈ √(p(1-p)/60) ≈ 6-7%
- 대부분의 차이(3-8%)가 1 SE 이내 → **통계적으로 유의미하지 않음**
- 결론을 내리려면 최소 200+ episodes 필요

### 4. 학습 미완료 체크포인트의 한계
- PNP ~80%, Stack ~48% 학습 → 최종 성능이 아님
- 특히 Stack은 절반도 안 학습 → 조건 간 차이가 수렴 후와 다를 수 있음

---

## Key Takeaway

| 관찰 | 신뢰도 |
|------|--------|
| "both" augmentation이 clean SR을 해치지 않고 오히려 높임 | ⭐⭐⭐ (일관됨) |
| "both"가 noisy eval에서 가장 안정적 (최소 SR drop) | ⭐⭐ (Stack에서 명확) |
| 개별 noise/drop이 특별히 유리하지 않음 | ⭐⭐ (일관됨) |
| 절대적 SR 차이가 통계적으로 유의미 | ⭐ (episode 수 부족) |

**권장사항:**
1. **"both" augmentation 채택** — 손해 없이 잠재적 이득
2. 실배포 전 real-world에서도 robustness 재평가 필요
3. 실배포 전 더 많은 episodes로 재평가 (200+)
3. 더 큰 noise level (0.05, 0.1)에서 robustness 차이 확인

---

## Eval Configuration

```
eval_robustness_standalone.py
├── noise_max: 0.01 (uniform noise on object state)
├── dropout_prob: 0.1 (10% chance of zeroing object state)
├── eval_num_episodes: 3 per env
├── num_envs: 20
├── total_episodes: 60 per condition
└── conditions: clean, noisy (noise + dropout)
```

## File Structure

```
robustness/
├── README.md              ← this file
└── data/
    ├── pnp_robustness.csv
    └── stack_robustness.csv
```

Raw JSON results: `outputs/robustness_eval/`

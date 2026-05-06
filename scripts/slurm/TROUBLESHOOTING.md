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

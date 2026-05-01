#!/bin/bash
# Submit remaining 4 video rollout jobs
# Run: bash scripts/submit_remaining.sh
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
TEMPLATE=~/Reports/slurm/template.sh
OUT="outputs/trajectory_videos"
NV="\$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia"
NV_LD="${NV}/cu13/lib:${NV}/cuda_runtime/lib:${NV}/cublas/lib:${NV}/cudnn/lib:${NV}/cufft/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib:${NV}/cuda_nvrtc/lib:${NV}/nccl/lib"
ENV=". ~/.venvs/groot/bin/activate && export LD_LIBRARY_PATH=${NV_LD}:\$HOME/lib-compat:~/.local/lib/gl:\${LD_LIBRARY_PATH:-} && export MUJOCO_GL=egl DS_BUILD_OPS=0 HF_HUB_OFFLINE=1 CUDA_HOME=~/fake_cuda && export PYTHONPATH=$(pwd):\${PYTHONPATH:-} && cd $(pwd)"
C="--partition=core --qos=core-own --account=core --gres=gpu:1 --cpus-per-task=14 --mem=200G --time=2:00:00 --comment=pytorch"

for job in \
  "vid_pnp_base|--task pnp --mode base --positions 1" \
  "vid_pnp_res|--task pnp --mode residual --positions 1" \
  "vid_stack_base_eff|--task stack --mode base --positions 3,4,7,14,15" \
  "vid_stack_res_eff|--task stack --mode residual --positions 3,4,7,14,15"; do
  NAME="${job%%|*}"
  ARGS="${job##*|}"
  echo "Submitting $NAME..."
  sbatch --job-name="$NAME" $C --output="slurms/${NAME}_%j.out" "$TEMPLATE" \
    "${ENV} && python scripts/rollout_video.py $ARGS --num_rounds 3 --output_dir $OUT"
  sleep 5
done
echo "Done. Check: squeue -u \$USER"

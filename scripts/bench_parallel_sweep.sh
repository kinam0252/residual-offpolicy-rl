#!/bin/bash
# Benchmark sweep: parallel worker count optimization
# Submits 6 jobs to sub partition with core-own QoS
# Each runs 1000 training steps to measure mujoco_step timing

set -e
cd ~/Repos/Intern/residual-offpolicy-rl

COMMON_ARGS=(
    --groot_checkpoint ~/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-100000
    --use_action_scaler --no_action_clamp
    --eval_num_envs 1
    --offline_data_dir outputs/offline_stack_66ep
    --offline_fraction 0.75
    --num_envs 30
    --episode_positions_file configs/stack_cube_positions.json
    --total_timesteps 1000
    --eval_interval 99999 --checkpoint_interval 99999
    --batch_size 256 --buffer_size 50000
    --actor_lr 3e-4 --critic_lr 3e-4
    --gamma 0.95 --target_tau 0.005
    --max_episode_steps 300
    --random_action_noise_scale 0.05
    --action_l2_reg 0.1
    --critic_warmup_steps 200
    --n_step 3
    --no_offline_cache
    --actor_hidden_dim 256 --critic_hidden_dim 256
    --wandb_mode disabled
    --reward_type dense
    --offline_reward_relabel none
    --residual_rot_scale 0.05
    --action_scale 0.05
    --seed 42
)

# Worker counts to test
WORKERS=(0 2 4 6 8 10)

for W in "${WORKERS[@]}"; do
    if [ "$W" -eq 0 ]; then
        PARALLEL=0
        WNAME="seq"
        LABEL="sequential"
    else
        PARALLEL=1
        WNAME="w${W}"
        LABEL="${W}_workers"
    fi

    OUTDIR="outputs/stack_rl/bench_parallel_${WNAME}"
    JOBNAME="bench_${WNAME}"

    sbatch --job-name="$JOBNAME" \
           --partition=sub \
           --qos=core-on-sub \
           --gres=gpu:1 \
           --cpus-per-task=14 \
           --mem=200G \
           --time=01:00:00 \
           --output="/home/nas_main/kinamkim/slurms/${JOBNAME}_%j.out" \
           --error="/home/nas_main/kinamkim/slurms/${JOBNAME}_%j.err" \
           --wrap="
set -e
. ~/.venvs/groot/bin/activate
export MUJOCO_GL=egl
NV=\$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=\$HOME/lib-compat:\$NV/cuda_runtime/lib:\$NV/cublas/lib:\$NV/cudnn/lib:\$NV/cufft/lib:\$NV/cusolver/lib:\$NV/cusparse/lib:\$NV/nvjitlink/lib:\$NV/cuda_nvrtc/lib:\$NV/nccl/lib:\$HOME/.local/lib/gl:\${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DS_BUILD_OPS=0 CUDA_HOME=~/fake_cuda
export PATH=~/bin:\$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0

cd ~/Repos/Intern/residual-offpolicy-rl
echo '=== Benchmark: ${LABEL} (parallel=${PARALLEL}, workers=${W}) ==='
echo 'CPUs available:' \$(nproc)
date

python3 resfit/rl_finetuning/scripts/train_residual_td3_mujoco_stack.py \\
    ${COMMON_ARGS[*]} \\
    --parallel_envs ${PARALLEL} \\
    --num_workers ${W} \\
    --output_dir ${OUTDIR}

echo 'Done.'
date
"

    echo "Submitted: $JOBNAME (parallel=$PARALLEL, workers=$W)"
done

echo ""
echo "All 6 benchmark jobs submitted. Check results with:"
echo "  grep 'mujoco_step' ~/slurms/bench_*.out"
echo "  grep 'wall_time\|steps/s' ~/slurms/bench_*.out"

#!/bin/bash
# State ablation evaluation for Stack Cube Residual TD3
# 8 jobs: test how RL policy uses observation components
set -e
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

TEMPLATE=~/Reports/slurm/template.sh
CKPT="outputs/mujoco_td3/gr00t_sim_66ep/easy_v2_s42/checkpoints/best.pt"
GROOT_CKPT="$HOME/DATA/INTERN/training/groot_stack_sim_66ep/checkpoint-75000"
POSITIONS="configs/stack_cube_positions.json"

NV="\$HOME/.venvs/groot/lib/python3.10/site-packages/nvidia"
NV_LD="${NV}/cu13/lib:${NV}/cuda_runtime/lib:${NV}/cublas/lib:${NV}/cudnn/lib:${NV}/cufft/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib:${NV}/cuda_nvrtc/lib:${NV}/nccl/lib"
ENV_CMD=". ~/.venvs/groot/bin/activate && export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,) && export LD_LIBRARY_PATH=${NV_LD}:\$HOME/lib-compat:~/.local/lib/gl:\${LD_LIBRARY_PATH:-} && export MUJOCO_GL=egl DS_BUILD_OPS=0 HF_HUB_OFFLINE=1 CUDA_HOME=~/fake_cuda && export PYTHONPATH=$(pwd):\${PYTHONPATH:-} && cd $(pwd)"

C="--partition=core --qos=core-own --account=core --gres=gpu:1 --cpus-per-task=14 --mem=200G --time=2:00:00 --comment=pytorch"

COMMON_ARGS="--groot_checkpoint ${GROOT_CKPT} --eval_positions_file ${POSITIONS} --asymmetric_critic --num_episodes 3 --action_scale 0.1 --max_positions 10"

# 8 experiments:
# 1. normal (control) - residual with correct state
# 2. zero_obj - zero out object_state (cube poses)
# 3. rand_obj - random object_state
# 4. noise_obj_5 - 5cm noise on object position
# 5. noise_obj_20 - 20cm noise on object position
# 6. zero_state - zero out proprio state (eef pos/quat/grip)
# 7. rand_state - random proprio state
# 8. zero_ba - zero out base_action

MODES=("normal" "zero_obj" "rand_obj" "noise_obj_5" "noise_obj_20" "zero_state" "rand_state" "zero_ba")

for mode in "${MODES[@]}"; do
    echo "Submitting: ${mode}"
    sbatch --job-name="abl_${mode}" $C \
        --output="slurms/abl_${mode}_%j.out" \
        "$TEMPLATE" \
        "${ENV_CMD} && python scripts/eval_state_ablation.py --checkpoint ${CKPT} --noise_mode ${mode} ${COMMON_ARGS}"
    sleep 2
done

echo "Done. Check: squeue -u \$USER"

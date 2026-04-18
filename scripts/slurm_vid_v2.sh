#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=vid_v2
#SBATCH --output=/home/nas_main/kinamkim/slurms/vid_v2_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/vid_v2_%j.err
#SBATCH --time=2:00:00
#SBATCH --exclude=worker-6

set -e
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

CKPT="/home/nas_main/kinamkim/DATA/INTERN/training/groot_pnp_sim_33ep/checkpoint-100000"

python3 scripts/eval_pnp_base_sr.py \
    --groot_checkpoint "$CKPT" \
    --difficulty easy \
    --episodes 10 \
    --seed 42 \
    --out_dir outputs/pnp_vid_v2 \
    --max_episode_steps 500 \
    --scene_xml /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml \
    --positions_file configs/pnp_common5_positions.json \
    --ema_alpha 0.0

echo "Done at $(date)"

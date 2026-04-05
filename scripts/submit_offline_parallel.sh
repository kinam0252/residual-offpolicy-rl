#!/bin/bash
# Submit 10 parallel offline data collection jobs (one per env position)
set -e
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

CKPT="~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000"
SCENE="~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml"
TABLE="configs/mujoco_cube_perturb_table.json"
EPISODES=50
OUTDIR="outputs/offline_data/batch_$(date +%Y%m%d_%H%M%S)"

for ENV_ID in 0 1 2 3 4 5 6 7 8 9; do
    sbatch --job-name="off_e${ENV_ID}" \
           --partition=core \
           --qos=core-extra \
           --gres=gpu:1 \
           --cpus-per-task=8 \
           --mem=64G \
           --time=4:00:00 \
           --output="/home/nas_main/kinamkim/slurms/mj_offdata_e${ENV_ID}_%j.out" \
           --error="/home/nas_main/kinamkim/slurms/mj_offdata_e${ENV_ID}_%j.err" \
           --wrap="
source /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:\${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export DS_BUILD_OPS=0
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
export PYTHONPATH=\${PWD}:\${PYTHONPATH:-}
echo \"Collecting env_id=${ENV_ID} | \$(hostname) | CUDA=\${CUDA_VISIBLE_DEVICES}\"
python3 resfit/rl_finetuning/scripts/collect_offline_data.py \
    --groot_checkpoint ${CKPT} \
    --scene_xml ${SCENE} \
    --perturb_table ${TABLE} \
    --num_episodes_per_env ${EPISODES} \
    --max_episode_steps 300 \
    --output_dir ${OUTDIR}/env${ENV_ID} \
    --env_ids ${ENV_ID}
"
    echo "Submitted env_id=${ENV_ID}"
done

echo ""
echo "All 10 jobs submitted. Output: ${OUTDIR}"

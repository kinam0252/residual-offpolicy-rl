#!/bin/bash
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --job-name=noise_rot_30deg
#SBATCH --output=/home/nas_main/kinamkim/slurms/noise_rot_30deg_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/noise_rot_30deg_%j.err

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:$LD_LIBRARY_PATH
export DS_BUILD_OPS=0
export PYTHONUNBUFFERED=1

python -m resfit.rl_finetuning.scripts.eval_checkpoint     --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000     --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml     --perturb_table configs/mujoco_cube_perturb_table.json     --checkpoint outputs/mujoco_td3/v57_dense_s123/checkpoints/best.pt     --eval_num_episodes 2     --asymmetric_critic     --action_scale 0.2 --actor_hidden_dim 512 --critic_hidden_dim 1024     --seed 42 \
    --obj_rot_noise 30.0 \
    --output_json outputs/eval_results/noise_rot_30deg.json \
    --label noise_rot_30deg

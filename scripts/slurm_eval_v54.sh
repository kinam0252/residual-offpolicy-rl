#!/bin/sh
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/eval_v54_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/eval_v54_%j.err
#SBATCH --job-name=eval_v54

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:$LD_LIBRARY_PATH
export DS_BUILD_OPS=0
export PYTHONUNBUFFERED=1

python -m resfit.rl_finetuning.scripts.eval_checkpoint --checkpoint outputs/mujoco_td3/v54_dense_nol2_s02/checkpoints/best.pt --action_scale 0.2 --actor_hidden_dim 512 --critic_hidden_dim 1024 --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml --perturb_table configs/mujoco_cube_perturb_table.json --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 --eval_num_episodes 2 --seeds 42 123 456 --max_episode_steps 500 --reward_type dense --asymmetric_critic --label v54_best --output_json outputs/eval_results/v54_best.json

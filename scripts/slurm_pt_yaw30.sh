#!/bin/sh
#SBATCH --partition=core
#SBATCH --qos=core-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nas_main/kinamkim/slurms/pt_yaw30_%j.out
#SBATCH --error=/home/nas_main/kinamkim/slurms/pt_yaw30_%j.err
#SBATCH --job-name=pt_yaw30

cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
. /home/nas_main/kinamkim/.venvs/groot/bin/activate
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=~/.local/lib/gl:$LD_LIBRARY_PATH
export DS_BUILD_OPS=0
export PYTHONUNBUFFERED=1

python -m resfit.rl_finetuning.scripts.eval_checkpoint --checkpoint none --perturb_table configs/perturb_test/perturb_yaw30.json --cube_yaw 30 --scene_xml ~/Repos/Intern/residual-offpolicy-rl_v1/mujoco_menagerie/franka_fr3/fr3_with_hand.xml --groot_checkpoint ~/DATA/INTERN/training/gr00t_groot_v2_30k_backup/checkpoint-30000 --eval_num_episodes 2 --max_episode_steps 500 --reward_type dense --asymmetric_critic --action_scale 0.2 --actor_hidden_dim 512 --critic_hidden_dim 1024 --seed 42 --label pt_yaw30 --output_json outputs/eval_results/pt_yaw30.json

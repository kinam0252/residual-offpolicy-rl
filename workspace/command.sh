cd /home/kinam/Desktop/Repos/VLA_RL/residual-offpolicy-rl/workspace
POLICY_DEVICE=cuda bash run_franka_gr00t_iface_workspace.sh --headless


cd /home/kinam/Desktop/Repos/VLA_RL/residual-offpolicy-rl
HEADLESS=1 \
NUM_ENVS=1 \
MAX_STEPS=1000 \
CSV_BASE_DIR=/home/kinam/Desktop/DATA/dataset_from_Namiko/0_Raw_dataset/pickMushroom/pickMushroom_20251209_081744_174 \
GROOT_MODEL_PATH=/home/kinam/Desktop/Repos/VLA_RL/Isaac-GR00T/outputs/checkpoint-100000 \
GROOT_EMBODIMENT_TAG=new_embodiment \
GROOT_POLICY_DEVICE=cuda \
resfit/rl_finetuning/scripts/run_rollout_groot_residual_zero_isaaclab.sh


# 1) resfit 헤드리스 롤아웃 (GPU)
cd /home/kinam/Desktop/Repos/VLA_RL/residual-offpolicy-rl
NUM_ENVS=1 \
MAX_STEPS=1000 \
CSV_BASE_DIR=/home/kinam/Desktop/DATA/dataset_from_Namiko/0_Raw_dataset/pickMushroom/pickMushroom_20251209_081744_174 \
GROOT_MODEL_PATH=/home/kinam/Desktop/Repos/VLA_RL/Isaac-GR00T/outputs/checkpoint-100000 \
GROOT_EMBODIMENT_TAG=new_embodiment \
GROOT_POLICY_DEVICE=cuda \
resfit/rl_finetuning/scripts/run_rollout_groot_residual_zero_isaaclab.sh
cd <YOUR_REPO_PATH>
POLICY_DEVICE=cuda bash run_franka_gr00t_iface_workspace.sh --headless


cd <YOUR_REPO_PATH>
HEADLESS=1 \
NUM_ENVS=1 \
MAX_STEPS=1000 \
CSV_BASE_DIR=<YOUR_DATA_PATH> \
GROOT_MODEL_PATH=<YOUR_REPO_PATH> \
GROOT_EMBODIMENT_TAG=new_embodiment \
GROOT_POLICY_DEVICE=cuda \
resfit/rl_finetuning/scripts/run_rollout_groot_residual_zero_isaaclab.sh


# 1) resfit 헤드리스 롤아웃 (GPU)
cd <YOUR_REPO_PATH>
NUM_ENVS=1 \
MAX_STEPS=1000 \
CSV_BASE_DIR=<YOUR_DATA_PATH> \
GROOT_MODEL_PATH=<YOUR_REPO_PATH> \
GROOT_EMBODIMENT_TAG=new_embodiment \
GROOT_POLICY_DEVICE=cuda \
resfit/rl_finetuning/scripts/run_rollout_groot_residual_zero_isaaclab.sh
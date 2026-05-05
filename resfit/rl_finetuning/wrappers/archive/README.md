# Archive — Per-Task Residual Wrappers

**DEPRECATED**: These per-task residual wrappers are replaced by the unified wrapper:
`resfit/rl_finetuning/wrappers/mujoco_residual_wrapper_unified.py`

Use `train_residual_td3_unified.py --task {cup,pnp,lift,stack,drawer}` for all training.

## Archived files
- `mujoco_residual_wrapper.py` — lift (original)
- `mujoco_residual_wrapper_cup.py` — stand cup
- `mujoco_residual_wrapper_pnp.py` — pick and place
- `mujoco_residual_wrapper_stack.py` — stack cube
- `mujoco_residual_wrapper_drawer.py` — close drawer

## Note
`mujoco_vec_env_*.py` files remain in the parent directory — they are still used by the unified pipeline.

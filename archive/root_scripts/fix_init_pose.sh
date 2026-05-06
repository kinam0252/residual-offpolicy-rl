#!/bin/bash
set -e
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl

PYBIN=/home/nas_main/kinamkim/.venvs/groot/bin/python

# 1. Update JSON: rename cube_yaw -> cube_yaw_deg with +90 offset
$PYBIN -c "
import json

for fn in ['configs/pnp_common5_positions.json', 'configs/pnp_sim33ep_positions.json']:
    with open(fn) as f:
        data = json.load(f)
    for ep in data:
        if 'cube_yaw' in ep:
            ep['cube_yaw_deg'] = ep.pop('cube_yaw') + 90.0
    with open(fn, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Updated {fn}: {len(data)} entries')
"

# 2. Patch eval script: replace HOME_QPOS with real joint position
EVAL=scripts/eval_pnp_base_sr.py

# Replace the HOME_QPOS import and usage
sed -i 's/from resfit.rl_finetuning.wrappers.mujoco_vec_env_pnp import HOME_QPOS/# Real teleop initial joint position (idx=2, sorted by joint name)\n    REAL_INIT_QPOS = np.array([-1.567889, -1.287458, 1.495029, -2.485614, 1.351407, 1.725887, -0.180199])/' "$EVAL"

sed -i 's/data\.qpos\[model\.jnt_qposadr\[jid\]\] = HOME_QPOS\[j\]/data.qpos[model.jnt_qposadr[jid]] = REAL_INIT_QPOS[j]/' "$EVAL"

# 3. Patch eval script: read cube_yaw_deg from positions for easy difficulty
# Replace "yaw_deg = 0.0" with reading from ep
sed -i 's/yaw_deg = 0\.0/yaw_deg = ep.get("cube_yaw_deg", 0.0)/' "$EVAL"

echo "=== Verifying patches ==="
grep -n 'REAL_INIT_QPOS\|cube_yaw_deg' "$EVAL"
echo "=== Done ==="

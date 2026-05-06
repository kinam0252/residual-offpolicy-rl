#!/bin/bash
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
/home/nas_main/kinamkim/.venvs/groot/bin/python -c "
import json
# Revert cube_yaw_deg to eef_yaw only (remove the +90 offset)
for fn in ['configs/pnp_common5_positions.json', 'configs/pnp_sim33ep_positions.json']:
    with open(fn) as f:
        data = json.load(f)
    for ep in data:
        if 'cube_yaw_deg' in ep:
            ep['cube_yaw_deg'] -= 90.0
    with open(fn, 'w') as f:
        json.dump(data, f, indent=2)
    vals = [round(ep['cube_yaw_deg'], 2) for ep in data[:5]]
    print(f'{fn}: first 5 = {vals}')
"
rm -rf outputs/pnp_ema_eval_v2 outputs/pnp_vid_v2
echo "Done"

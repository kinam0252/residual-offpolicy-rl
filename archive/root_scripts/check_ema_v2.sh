#!/bin/bash
cd /home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
/home/nas_main/kinamkim/.venvs/groot/bin/python -c "
import json, glob
base = 'outputs/pnp_ema_eval_v2'
results = {}
for f in sorted(glob.glob(f'{base}/*/seed_*/results.json')):
    parts = f.split('/')
    ema = parts[-3]
    seed = parts[-2]
    d = json.load(open(f))
    sr = d['success_rate']
    if ema not in results:
        results[ema] = []
    results[ema].append((seed, sr))

print(f'Completed: {sum(len(v) for v in results.values())}/18')
print()
for ema in sorted(results.keys(), key=lambda x: float(x.split('_')[1])):
    srs = [s[1] for s in results[ema]]
    avg = sum(srs)/len(srs) if srs else 0
    detail = ', '.join(f'{s[0]}={s[1]:.0%}' for s in results[ema])
    print(f'{ema}: AVG={avg:.0%}  [{detail}]')
"

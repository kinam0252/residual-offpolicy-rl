import numpy as np, sys, glob, os
base = sys.argv[1] if len(sys.argv) > 1 else "outputs/offline_data/pnp_easy_v3"
eps = sorted(glob.glob(os.path.join(base, "*/episode.npz")) + glob.glob(os.path.join(base, "ep_*.npz")))
print(f"Found {len(eps)} episodes")
for p in eps[:5]:
    d = np.load(p)
    name = os.path.basename(p)
    succ = d["success"][0]
    r = d["reward"].sum() if "reward" in d else d.get("reward_uniform4", np.zeros(1)).sum()
    steps = len(d["reward"]) if "reward" in d else len(d.get("reward_uniform4", [0]))
    term = d["terminated"].any()
    print(f"{name}: succ={succ} term={term} steps={steps} r_sum={r:.1f}")
# Aggregate
all_succ = []
for p in eps:
    d = np.load(p)
    all_succ.append(d["success"][0])
sr = sum(all_succ) / len(all_succ) * 100 if all_succ else 0
print(f"\nTotal: {len(eps)} eps, SR={sr:.1f}% ({sum(all_succ)}/{len(all_succ)})")

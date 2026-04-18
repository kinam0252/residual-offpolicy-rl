"""Aggregate PnP base eval results into a summary table."""
import os, json, glob, sys


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/pnp_base_eval"
    
    results = {}
    for results_file in sorted(glob.glob(os.path.join(base_dir, "*/results.json"))):
        with open(results_file) as f:
            data = json.load(f)
        
        # Extract key from directory name (e.g., sim_33ep_100k_easy)
        dir_name = os.path.basename(os.path.dirname(results_file))
        parts = dir_name.rsplit("_", 1)  # split off difficulty
        difficulty = parts[1] if len(parts) == 2 else "unknown"
        ckpt_key = parts[0] if len(parts) == 2 else dir_name
        
        if ckpt_key not in results:
            results[ckpt_key] = {}
        results[ckpt_key][difficulty] = data
    
    if not results:
        print(f"No results found in {base_dir}/")
        return
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"PnP Base Policy Evaluation Summary")
    print(f"{'='*70}")
    print(f"{'Checkpoint':<25} {'Easy':>8} {'Normal':>8} {'Hard':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    
    for ckpt_key in sorted(results.keys()):
        diffs = results[ckpt_key]
        easy = diffs.get("easy", {})
        normal = diffs.get("normal", {})
        hard = diffs.get("hard", {})
        
        easy_sr = f"{easy.get('success_rate', 0)*100:.0f}%" if easy else "—"
        normal_sr = f"{normal.get('success_rate', 0)*100:.0f}%" if normal else "—"
        hard_sr = f"{hard.get('success_rate', 0)*100:.0f}%" if hard else "—"
        
        print(f"{ckpt_key:<25} {easy_sr:>8} {normal_sr:>8} {hard_sr:>8}")
    
    print(f"{'='*70}")
    
    # Detailed per-episode info
    print(f"\n{'='*70}")
    print(f"Detailed Results (min_dist)")
    print(f"{'='*70}")
    for ckpt_key in sorted(results.keys()):
        for diff in ["easy", "normal", "hard"]:
            data = results[ckpt_key].get(diff)
            if not data:
                continue
            eps = data.get("episodes", [])
            dists = [e["min_dist"] for e in eps]
            avg_dist = sum(dists) / len(dists) if dists else 0
            sr = data.get("success_rate", 0)
            print(f"  {ckpt_key}_{diff}: SR={sr*100:.0f}%, "
                  f"avg_min_dist={avg_dist:.3f}m, "
                  f"n={len(eps)}")
    print()


if __name__ == "__main__":
    main()

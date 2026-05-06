#!/bin/bash
# Kill any previous submit script, create scripts, submit first job
pkill -f submit_dv2 2>/dev/null

REPO=/home/nas_main/kinamkim/Repos/Intern/residual-offpolicy-rl
SRC=$REPO/scripts/slurm_pnp_rl_v2_dv2.sh

# Fix base script to core-own
sed -i 's/--partition=sub/--partition=core/' $SRC
sed -i 's/--qos=core-on-sub/--qos=core-own/' $SRC
sed -i 's/pnp_v2_s02_/pnp_v2_dv2_s02_/g' $SRC

# Create s05 (core-own, scale 0.5)
sed -e 's/action_scale 0.2/action_scale 0.5/g' \
    -e 's/pnp_v2_dv2_s02_/pnp_v2_dv2_s05_/g' \
    -e 's|outputs/pnp_rl_v2_dv2/|outputs/pnp_rl_v2_dv2_s05/|g' \
    $SRC > $REPO/scripts/slurm_pnp_rl_v2_dv2_s05.sh

# Create s10 (core-own, scale 1.0)
sed -e 's/action_scale 0.2/action_scale 1.0/g' \
    -e 's/pnp_v2_dv2_s02_/pnp_v2_dv2_s10_/g' \
    -e 's|outputs/pnp_rl_v2_dv2/|outputs/pnp_rl_v2_dv2_s10/|g' \
    $SRC > $REPO/scripts/slurm_pnp_rl_v2_dv2_s10.sh

# Clean output dirs
for s in dv2 dv2_s05 dv2_s10; do
  for d in easy normal hard; do
    rm -rf $REPO/outputs/pnp_rl_v2_${s}/${d} 2>/dev/null
  done
done

echo "=== Verify core-own ==="
grep -E 'partition|qos' $SRC | head -2
grep -E 'partition|qos' $REPO/scripts/slurm_pnp_rl_v2_dv2_s05.sh | head -2
grep -E 'partition|qos' $REPO/scripts/slurm_pnp_rl_v2_dv2_s10.sh | head -2

echo "=== Submit s02 easy ==="
sbatch --export=ALL,DIFFICULTY=easy $SRC

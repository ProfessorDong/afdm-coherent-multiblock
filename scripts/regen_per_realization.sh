#!/usr/bin/env bash
# Full regeneration under per_realization_veff=True (Eq. 18 exactly).
# Sequential to avoid CPU contention. Each script saves its own JSON.
set -u
cd "$(dirname "$0")/.."
LOG=runs/REGEN.log
: > $LOG
run () {
  local name=$1; local to=$2
  echo "=== [$(date +%H:%M:%S)] START $name (timeout ${to}s)" | tee -a $LOG
  local t0=$SECONDS
  if timeout "$to" python3 -u "scripts/$name.py" >> "runs/regen_$name.log" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] OK    $name  ($((SECONDS-t0))s)" | tee -a $LOG
  else
    echo "=== [$(date +%H:%M:%S)] FAIL  $name  rc=$? ($((SECONDS-t0))s)" | tee -a $LOG
  fi
}
run scaling_B_v2      3600
run ablation_v2       2400
run fair_pilots_v2    3600
run convergence_v3    3600
run aperture_ablation 2400
run crb_vs_B          3600
run pilot_design_v2   1800
run reviewer_response 3600
run robustness_v2     2400
run hp_robustness     3600
run tdlc_evaluation   3600
run ber_vs_snr_v2     7200
run highorder_sweep   5400
run phase_coherence   5400
echo "=== [$(date +%H:%M:%S)] ALL DONE" | tee -a $LOG

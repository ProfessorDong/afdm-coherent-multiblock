"""Reconstructed tuning sweep for the D-GESBL-style baseline (Table I caption).

The baseline is an ADAPTATION and must be reported at its best, so its two
free hyperparameters are swept over a 20-point grid at the HARD operating
point (P=5, N_p=16), B=1, 15 dB -- the point where the comparison is decided.
The winner is then evaluated at the full 5-seed x 8-batch x 32 protocol by
run_dgesbl_{hard,easy,fairpilots}.py.

Tuning protocol (deliberately cheaper than the final eval): 3 seeds x 4
batches x 32 realizations, seeds k*137+42 matching every other script.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from afdm.experiments import ExperimentConfig
from dgesbl_baseline import eval_dgesbl

T_EM_GRID = [10, 15, 20, 30, 40]
GRID_LR_GRID = [0.0, 0.05, 0.1, 0.2]
SNR, N_SEEDS, N_BATCHES, BATCH = 15.0, 3, 4, 32

cfg = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=5, N_p=16, P_max=8)
out = {"snr_db": SNR, "operating_point": "HARD (P=5, N_p=16), B=1",
       "protocol": {"N_seeds": N_SEEDS, "N_batches": N_BATCHES, "batch_size": BATCH},
       "grid": {"T_em": T_EM_GRID, "grid_lr": GRID_LR_GRID}, "results": {}}
p = Path("runs/dgesbl_tuning.json")

print(f"D-GESBL-style tuning: {len(T_EM_GRID)*len(GRID_LR_GRID)} configurations", flush=True)
best = (1e9, None)
for T_em in T_EM_GRID:
    for glr in GRID_LR_GRID:
        t0 = time.time()
        v = [eval_dgesbl(cfg, SNR, B_block=1, seed=k * 137 + 42, n_batches=N_BATCHES,
                         batch_size=BATCH, T_em=T_em, grid_lr=glr) for k in range(N_SEEDS)]
        a = np.array(v)
        key = f"T_em={T_em},grid_lr={glr}"
        out["results"][key] = {"T_em": T_em, "grid_lr": glr,
                               "mean": float(a.mean()), "std": float(a.std())}
        if a.mean() < best[0]:
            best = (float(a.mean()), key)
        print(f"  {key:26s} {a.mean()*100:6.2f}% +/- {a.std()*100:4.2f}   ({time.time()-t0:.0f}s)", flush=True)
        out["best"] = {"key": best[1], "mean": best[0]}
        p.write_text(json.dumps(out, indent=1))
print(f"\nBEST: {best[1]} at {best[0]*100:.2f}%")

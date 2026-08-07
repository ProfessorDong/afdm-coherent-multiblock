"""Re-evaluate the D-GESBL-style baseline at its re-tuned optimum.

The extended 32-point sweep (tune_dgesbl.py + tune_dgesbl_ext.py) moved the
optimum from T_em=40 (a grid-boundary value) to T_em=160, grid_lr=0.1. This
re-runs every reported baseline number at that setting under the full
5 seeds x 8 batches x 32 protocol used by all other receivers.

Saves incrementally after every entry.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from afdm.experiments import ExperimentConfig
from dgesbl_baseline import eval_dgesbl

BEST = dict(T_em=160, grid_lr=0.1)
SNR, N_SEEDS, NB, BS = 15.0, 5, 8, 32
out = {"config": BEST, "snr_db": SNR, "N_seeds": N_SEEDS,
       "N_batches": NB, "batch_size": BS, "results": {}}
p = Path("runs/dgesbl_retuned.json")

def go(tag, cfg, B, **kw):
    t0 = time.time()
    v = [eval_dgesbl(cfg, SNR, B_block=B, seed=k * 137 + 42, n_batches=NB,
                     batch_size=BS, **BEST, **kw) for k in range(N_SEEDS)]
    a = np.array(v)
    out["results"][tag] = {"mean": float(a.mean()), "std": float(a.std())}
    print(f"  {tag:24s} {a.mean()*100:6.2f}% +/- {a.std()*100:4.2f}   ({time.time()-t0:.0f}s)", flush=True)
    p.write_text(json.dumps(out, indent=1))

hard = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=5, N_p=16, P_max=8)
easy = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=3, N_p=32, P_max=6)
fp3  = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=3, N_p=64, P_max=6)
fp5  = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=5, N_p=64, P_max=8)

print("[fair pilots, N_p=64, B=1]", flush=True)
go("fair_P3", fp3, 1); go("fair_P5", fp5, 1)
print("[Hard (P=5, N_p=16)]", flush=True)
for B in [1, 2, 4, 8]: go(f"hard_B{B}", hard, B)
print("[Easy (P=3, N_p=32)]", flush=True)
for B in [1, 2, 4, 8]: go(f"easy_B{B}", easy, B)
print("done")

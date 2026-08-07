"""Does the D-GESBL-style baseline still saturate in B when given many more
EM iterations? The tuning grid's optimum sat at its T_em=40 edge, so the
reported baseline may be under-tuned. Compares T_em=40 vs 160 across B at the
HARD point under one common (cheaper) protocol so the T_em effect is isolated.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from afdm.experiments import ExperimentConfig
from dgesbl_baseline import eval_dgesbl

cfg = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=5, N_p=16, P_max=8)
SEEDS, NB, BS = 3, 4, 32
out = {"operating_point": "HARD (P=5,N_p=16)", "snr_db": 15.0,
       "protocol": {"N_seeds": SEEDS, "N_batches": NB, "batch_size": BS},
       "grid_lr": 0.1, "results": {}}
p = Path("runs/dgesbl_Tem_check.json")
for T in [40, 160]:
    for B in [1, 2, 4, 8]:
        t0 = time.time()
        v = [eval_dgesbl(cfg, 15.0, B_block=B, seed=k * 137 + 42, n_batches=NB,
                         batch_size=BS, T_em=T, grid_lr=0.1) for k in range(SEEDS)]
        a = np.array(v)
        out["results"][f"T{T}_B{B}"] = {"T_em": T, "B": B,
                                        "mean": float(a.mean()), "std": float(a.std())}
        print(f"  T_em={T:3d} B={B}: {a.mean()*100:6.2f}% +/- {a.std()*100:4.2f}  ({time.time()-t0:.0f}s)", flush=True)
        p.write_text(json.dumps(out, indent=1))
r = out["results"]
print("\n  B :  T=40    T=160")
for B in [1, 2, 4, 8]:
    print(f"  {B} : {r[f'T40_B{B}']['mean']*100:6.2f}  {r[f'T160_B{B}']['mean']*100:6.2f}")

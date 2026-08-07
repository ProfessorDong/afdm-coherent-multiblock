"""Full evaluation of the tuned D-GESBL-style baseline at the paper's protocol."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from afdm.experiments import ExperimentConfig
from dgesbl_baseline import eval_dgesbl
BEST = dict(T_em=40, grid_lr=0.1)          # tuned; baseline reported at its best
SNR, N_SEEDS, N_BATCHES, BATCH = 15.0, 5, 8, 32
def avg(cfg,B,**kw):
    v=[eval_dgesbl(cfg,SNR,B_block=B,seed=k*137+42,n_batches=N_BATCHES,batch_size=BATCH,**kw) for k in range(N_SEEDS)]
    a=np.array(v); return float(a.mean()), float(a.std())
out={"config":BEST,"snr_db":SNR,"N_seeds":N_SEEDS,"N_batches":N_BATCHES,"batch_size":BATCH,"results":{}}
for name,cfg in [("Easy (P=3, N_p=32)",ExperimentConfig(N=128,kappa_max=5.,ell_max=10.,P=3,N_p=32,P_max=6)),
                 ("Hard (P=5, N_p=16)",ExperimentConfig(N=128,kappa_max=5.,ell_max=10.,P=5,N_p=16,P_max=8))]:
    out["results"][name]={}
    print(f"\n[{name}]",flush=True)
    for B in [1,2,4,8]:
        t0=time.time(); m,s=avg(cfg,B,**BEST)
        out["results"][name][str(B)]={"mean":m,"std":s}
        print(f"  B={B}: {m*100:.2f}% +/- {s*100:.2f}   ({time.time()-t0:.0f}s)",flush=True)
p=Path("runs/dgesbl_eval.json"); json.dump(out,open(p,"w"),indent=2); print(f"\nSaved: {p}")

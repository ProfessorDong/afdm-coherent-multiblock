"""D-GESBL-style at the Table III operating point (aggregate 64 pilots, B=1, N_p=64)."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from afdm.experiments import ExperimentConfig
from dgesbl_baseline import eval_dgesbl
BEST=dict(T_em=40, grid_lr=0.1); SNR,N_SEEDS,N_BATCHES,BATCH=15.0,5,8,32
out={"config":BEST,"snr_db":SNR,"N_p":64,"B":1,"N_seeds":N_SEEDS,"results":{}}
for P in (3,5):
    cfg=ExperimentConfig(N=128,kappa_max=5.,ell_max=10.,P=P,N_p=64,P_max=max(P+3,4))
    t0=time.time()
    v=[eval_dgesbl(cfg,SNR,B_block=1,seed=k*137+42,n_batches=N_BATCHES,batch_size=BATCH,**BEST) for k in range(N_SEEDS)]
    a=np.array(v); out["results"][f"P={P}"]={"mean":float(a.mean()),"std":float(a.std())}
    print(f"  P={P}, N_p=64: {a.mean()*100:.2f}% +/- {a.std()*100:.2f}  ({time.time()-t0:.0f}s)",flush=True)
    json.dump(out,open("runs/dgesbl_fairpilots.json","w"),indent=2)
print("Saved: runs/dgesbl_fairpilots.json")

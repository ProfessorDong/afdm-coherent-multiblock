import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from afdm.experiments import ExperimentConfig
from dgesbl_baseline import eval_dgesbl
BEST=dict(T_em=40, grid_lr=0.1)
SNR,N_SEEDS,N_BATCHES,BATCH=15.0,5,8,32
cfg=ExperimentConfig(N=128,kappa_max=5.,ell_max=10.,P=5,N_p=16,P_max=8)
out={"config":BEST,"snr_db":SNR,"N_seeds":N_SEEDS,"results":{}}
print("[Hard (P=5, N_p=16)]",flush=True)
for B in [1,2,4,8]:
    t0=time.time()
    v=[eval_dgesbl(cfg,SNR,B_block=B,seed=k*137+42,n_batches=N_BATCHES,batch_size=BATCH,**BEST) for k in range(N_SEEDS)]
    a=np.array(v); out["results"][str(B)]={"mean":float(a.mean()),"std":float(a.std())}
    print(f"  B={B}: {a.mean()*100:.2f}% +/- {a.std()*100:.2f}   ({time.time()-t0:.0f}s)",flush=True)
    json.dump(out,open("runs/dgesbl_hard.json","w"),indent=2)
print("Saved: runs/dgesbl_hard.json")

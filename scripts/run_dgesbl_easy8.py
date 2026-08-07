import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from afdm.experiments import ExperimentConfig
from dgesbl_baseline import eval_dgesbl
BEST=dict(T_em=40, grid_lr=0.1); SNR,N_SEEDS,N_BATCHES,BATCH=15.0,5,8,32
cfg=ExperimentConfig(N=128,kappa_max=5.,ell_max=10.,P=3,N_p=32,P_max=6)
print("[Easy (P=3,N_p=32)] B=8",flush=True)
t0=time.time()
v=[eval_dgesbl(cfg,SNR,B_block=8,seed=k*137+42,n_batches=N_BATCHES,batch_size=BATCH,**BEST) for k in range(N_SEEDS)]
a=np.array(v)
print(f"  B=8: {a.mean()*100:.2f}% +/- {a.std()*100:.2f}  ({time.time()-t0:.0f}s)",flush=True)
json.dump({"config":BEST,"B8_mean":float(a.mean()),"B8_std":float(a.std())},open("runs/dgesbl_easy8.json","w"),indent=2)
print("Saved: runs/dgesbl_easy8.json")

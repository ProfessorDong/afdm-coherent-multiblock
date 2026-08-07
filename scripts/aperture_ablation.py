"""Isolate what actually produces the multi-block gain.

The paper attributes the gain primarily to aperture-synthesis coherent kappa
refinement.  That is a causal claim about a receiver with several simultaneous
changes relative to B=1 (more observations, pilot-data interference averaging,
phase-corrected stacked LS, and the coherent fine search).  This ablation turns
the coherent fine search off while holding everything else fixed, so the claim
can be tested rather than asserted.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiblock_dasbl import multiblock_dasbl_receiver

SNR, N_SEEDS, N_BATCHES, BATCH = 15.0, 5, 8, 32

def ser(cfg, B, seed, use_aperture):
    system=cfg.system(); ch=cfg.channel(); const=cfg.constellation()
    pp,pv=PILOT_DESIGNS["hopping"](N=cfg.N,N_p=cfg.N_p,B=B,constellation=const,device=cfg.device,seed=42)
    g=torch.Generator(device=cfg.device); g.manual_seed(seed); acc=0.
    for _ in range(N_BATCHES):
        batch=sample_multiblock(system,ch,const,pp,pv,batch_size=BATCH,snr_db=SNR,generator=g)
        with torch.no_grad():
            hard,_,_,_=multiblock_dasbl_receiver(system,batch,const,cfg,use_aperture=use_aperture)
        m=batch.pilot_mask; acc+=float(((hard!=batch.labels)*m).float().sum()/m.float().sum())
    return acc/N_BATCHES

def avg(fn):
    a=np.array([fn(k*137+42) for k in range(N_SEEDS)]); return float(a.mean()), float(a.std())

def main():
    out={}
    for name,cfg in [("Hard (P=5,N_p=16)",ExperimentConfig(N=128,kappa_max=5.,ell_max=10.,P=5,N_p=16,P_max=8))]:
        print(f"\n{'='*74}\n{name}  @ {SNR} dB, per-block pilots fixed\n{'='*74}")
        print(f"{'B':>3s} {'no aperture':>18s} {'with aperture':>18s} {'gain from aperture':>20s}")
        res={}
        for B in [1,2,4,8]:
            t0=time.time()
            n_m,n_s=avg(lambda s: ser(cfg,B,s,False))
            a_m,a_s=avg(lambda s: ser(cfg,B,s,True))
            res[str(B)]={"no_aperture":{"mean":n_m,"std":n_s},"with_aperture":{"mean":a_m,"std":a_s}}
            print(f"{B:>3d} {n_m*100:>15.2f}%  {a_m*100:>15.2f}%  {n_m/max(a_m,1e-9):>17.2f}x   ({time.time()-t0:.0f}s)",flush=True)
        out[name]=res
    p=Path("runs/aperture_ablation.json"); p.parent.mkdir(parents=True,exist_ok=True)
    json.dump({"snr_db":SNR,"N_seeds":N_SEEDS,"results":out},open(p,"w"),indent=2)
    print(f"\nSaved: {p}")

if __name__=="__main__": main()

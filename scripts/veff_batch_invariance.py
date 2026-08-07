"""Batch-invariance check for the effective variance.

Eq. (18) defines v_eff as one scalar per realization. The batched implementation
pooled it into a batch mean for the CG ridge, which couples independent Monte
Carlo realizations: a deployed receiver processing one frame cannot see the
residuals of the other frames in its batch. This compares the pooled and
per-realization rules, and sweeps the computational batch size, which must not
affect a correct single-realization algorithm.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
from multiblock_dasbl import multiblock_dasbl_receiver

SNR, SEEDS = 15.0, [k * 137 + 42 for k in range(5)]
CFG = {"Hard": dict(P=5, N_p=16, P_max=8), "Easy": dict(P=3, N_p=32, P_max=6),
       "Hard-Np32": dict(P=5, N_p=32, P_max=8)}

def run(cfgkw, B, per_real, nb, bs):
    cfg = ExperimentConfig(N=128, kappa_max=5., ell_max=10., **cfgkw)
    system, ch, const = cfg.system(), cfg.channel(), cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B,
                                      constellation=const, device=cfg.device, seed=42)
    vals = []
    for sd in SEEDS:
        g = torch.Generator(device=cfg.device); g.manual_seed(sd); acc = 0.
        for _ in range(nb):
            b = sample_multiblock(system, ch, const, pp, pv, batch_size=bs, snr_db=SNR, generator=g)
            with torch.no_grad():
                hard, _, _, _ = multiblock_dasbl_receiver(system, b, const, cfg,
                                                          per_realization_veff=per_real)
            m = b.pilot_mask
            acc += float(((hard != b.labels) * m).float().sum() / m.float().sum())
        vals.append(acc / nb)
    a = np.array(vals); return float(a.mean()), float(a.std())

out = {"snr_db": SNR, "N_seeds": len(SEEDS), "pooled_vs_perrealization": {}, "batch_invariance": {}}
p = Path("runs/veff_batch_invariance.json")
print("[pooled batch-mean ridge  vs  per-realization ridge]  (5 seeds x 8 x 32)", flush=True)
for name, kw in CFG.items():
    for B in [1, 2, 4, 8]:
        t0 = time.time()
        m0, s0 = run(kw, B, False, 8, 32)
        m1, s1 = run(kw, B, True, 8, 32)
        out["pooled_vs_perrealization"][f"{name}_B{B}"] = {
            "pooled": {"mean": m0, "std": s0}, "per_realization": {"mean": m1, "std": s1},
            "delta_pp": (m1 - m0) * 100}
        print(f"   {name:10s} B={B}: pooled {m0*100:6.2f}  per-real {m1*100:6.2f}  "
              f"delta {(m1-m0)*100:+5.2f} pp  ({time.time()-t0:.0f}s)", flush=True)
        p.write_text(json.dumps(out, indent=1))
print("[batch-size invariance, per-realization rule, Hard B=4, 256 realizations/seed]", flush=True)
for bs, nb in [(1, 256), (8, 32), (32, 8)]:
    t0 = time.time(); m, s = run(CFG["Hard"], 4, True, nb, bs)
    out["batch_invariance"][str(bs)] = {"batch_size": bs, "n_batches": nb, "mean": m, "std": s}
    print(f"   batch_size={bs:<3d} {m*100:6.2f}% +/- {s*100:4.2f}  ({time.time()-t0:.0f}s)", flush=True)
    p.write_text(json.dumps(out, indent=1))
print("done")

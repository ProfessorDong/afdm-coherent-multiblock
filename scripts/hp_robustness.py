"""+/-30% hyperparameter perturbation sweep for the MB-IDAR receiver.

Substantiates the robustness claim in Table V of the manuscript. Each
hyperparameter is scaled to 0.7x and 1.3x its default in turn (integer-valued
ones rounded), with all others held at their defaults, at the ablation
operating point (P=5, N_p=32), B=4, 15 dB.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
from multiblock_dasbl import multiblock_dasbl_receiver

DEFAULTS = dict(n_outer=6, n_lm_per_outer=3, rho_min=0.5, lambda_ridge=1e-3,
                gamma_lr=0.5, max_step=0.15, slack=1e-4,
                kappa_window=0.30, kappa_step=0.003, K_cg=30)
INTEGER = {"n_outer", "n_lm_per_outer", "K_cg"}


def run(cfg, B_block, snr_db, seeds, n_batches, batch_size, **over):
    system = cfg.system(); ch = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_block,
                                      constellation=const, device=cfg.device, seed=42)
    kw = dict(DEFAULTS); kw.update(over)
    per_seed = []
    for sd in seeds:
        g = torch.Generator(device=cfg.device); g.manual_seed(sd); acc = 0.0
        for _ in range(n_batches):
            batch = sample_multiblock(system, ch, const, pp, pv, batch_size=batch_size,
                                      snr_db=snr_db, generator=g)
            with torch.no_grad():
                hard, _, _, _ = multiblock_dasbl_receiver(system, batch, const, cfg,
                                                          use_reacq=True, **kw)
            m = batch.pilot_mask
            acc += float(((hard != batch.labels) * m).float().sum() / m.float().sum())
        per_seed.append(acc / n_batches)
    mean = sum(per_seed) / len(per_seed)
    var = sum((v - mean) ** 2 for v in per_seed) / max(len(per_seed) - 1, 1)
    return mean, var ** 0.5


def main():
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)
    seeds = [k * 137 + 42 for k in range(5)]   # same draw as ablation_v2.py
    nb, bs, B, snr = 8, 32, 4, 15.0
    out = {"snr_db": snr, "B": B, "P": cfg.P, "N_p": cfg.N_p,
           "N_seeds": len(seeds), "N_batches": nb, "batch_size": bs,
           "defaults": DEFAULTS, "results": {}}
    p = Path("runs/hp_robustness.json")

    t0 = time.time()
    m, s = run(cfg, B, snr, seeds, nb, bs)
    out["baseline"] = {"mean": m, "std": s}
    print(f"baseline               {m*100:6.2f} +/- {s*100:4.2f}%   ({time.time()-t0:.0f}s)", flush=True)
    p.write_text(json.dumps(out, indent=1))

    for name, dv in DEFAULTS.items():
        for tag, f in (("0.7x", 0.7), ("1.3x", 1.3)):
            v = max(1, round(dv * f)) if name in INTEGER else dv * f
            t0 = time.time()
            mm, ss = run(cfg, B, snr, seeds, nb, bs, **{name: v})
            out["results"][f"{name}@{tag}"] = {"value": v, "mean": mm, "std": ss,
                                               "delta_pp": (mm - m) * 100}
            print(f"{name:16s} {tag}  {mm*100:6.2f} +/- {ss*100:4.2f}%  "
                  f"delta {(mm-m)*100:+5.2f} pp   ({time.time()-t0:.0f}s)", flush=True)
            p.write_text(json.dumps(out, indent=1))

    d = [abs(v["delta_pp"]) for v in out["results"].values()]
    out["max_abs_delta_pp"] = max(d)
    p.write_text(json.dumps(out, indent=1))
    print(f"\nmax |delta| = {max(d):.2f} pp over {len(d)} perturbations")


if __name__ == "__main__":
    main()

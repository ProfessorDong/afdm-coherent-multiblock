"""Inter-block phase-coherence stress test for the slow-time aperture.

The multi-block gain comes from a deterministic inter-block phase ramp, so any
impairment that randomizes block-to-block phase attacks the mechanism directly.
Two impairments are injected in the TIME domain, where an oscillator acts:

    r_b[n] <- r_b[n] * exp( j ( phi_b + 2 pi eps t_b(n) / N ) ),
    t_b(n) = b(N + N_cp) + N_cp + n,

with phi_b a Wiener block-phase walk (per-block increment std sigma_phi) and
eps a residual CFO normalized to subcarrier spacing. The DAFT-domain
observation y is recomputed from the impaired r, so both receiver inputs are
consistent.

This checks two claims made in the manuscript's synchronization remark:
 (i)  coherence needs accumulated phase deviation over the aperture well below
      one radian (accumulated std at B blocks is sigma_phi * sqrt(B-1));
 (ii) a residual CFO is benign for communication because it is common to all
      paths and CG-MMSE depends only on the composite operator.
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

SNR, NB, BS = 15.0, 8, 32
SEEDS = [k * 137 + 42 for k in range(3)]
cfg = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=5, N_p=16, P_max=8)
N = cfg.N; N_cp = int(cfg.ell_max)


def impair(batch, system, B_block, sigma_phi, eps, gen, device):
    r = batch.r
    Bb = r.shape[0]
    n = torch.arange(N, device=device, dtype=torch.float32)
    # Wiener block-phase walk, independent per realization
    if sigma_phi > 0:
        inc = torch.randn(Bb, B_block, device=device, generator=gen) * sigma_phi
        inc[:, 0] = 0.0
        phi = torch.cumsum(inc, dim=1)                       # (Bb, B_block)
    else:
        phi = torch.zeros(Bb, B_block, device=device)
    ph = torch.empty(Bb, B_block, N, device=device)
    for b in range(B_block):
        t_b = b * (N + N_cp) + N_cp + n                      # absolute time index
        ph[:, b, :] = phi[:, b:b+1] + 2 * torch.pi * eps * t_b.unsqueeze(0) / N
    r2 = r * torch.exp(1j * ph).to(r.dtype)
    y2 = system.daft(r2.reshape(-1, N)).reshape(Bb, B_block, N)
    batch.r = r2; batch.y = y2
    return batch


def run(B_block, sigma_phi, eps):
    system, ch, const = cfg.system(), cfg.channel(), cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=N, N_p=cfg.N_p, B=B_block,
                                      constellation=const, device=cfg.device, seed=42)
    vals = []
    for sd in SEEDS:
        g = torch.Generator(device=cfg.device); g.manual_seed(sd)
        gi = torch.Generator(device=cfg.device); gi.manual_seed(sd + 9999)
        acc = 0.
        for _ in range(NB):
            b = sample_multiblock(system, ch, const, pp, pv, batch_size=BS,
                                  snr_db=SNR, generator=g)
            b = impair(b, system, B_block, sigma_phi, eps, gi, cfg.device)
            with torch.no_grad():
                hard, _, _, _ = multiblock_dasbl_receiver(system, b, const, cfg)
            m = b.pilot_mask
            acc += float(((hard != b.labels) * m).float().sum() / m.float().sum())
        vals.append(acc / NB)
    a = np.array(vals)
    return float(a.mean()), float(a.std())


out = {"snr_db": SNR, "operating_point": "HARD (P=5,N_p=16)", "N_seeds": len(SEEDS),
       "N_batches": NB, "batch_size": BS, "phase_noise": {}, "cfo": {}}
p = Path("runs/phase_coherence.json")

print("[phase noise] sigma_phi rad per block; accumulated std = sigma_phi*sqrt(B-1)", flush=True)
for sp in [0.0, 0.05, 0.1, 0.2, 0.4]:
    for B in [1, 2, 4, 8]:
        t0 = time.time(); m, s = run(B, sp, 0.0)
        out["phase_noise"][f"sp{sp}_B{B}"] = {"sigma_phi": sp, "B": B, "mean": m, "std": s,
                                              "accum_rad": sp * (B - 1) ** 0.5}
        print(f"   sigma_phi={sp:<5} B={B}: {m*100:6.2f}% +/- {s*100:4.2f}  "
              f"(accum {sp*(B-1)**0.5:.2f} rad, {time.time()-t0:.0f}s)", flush=True)
        p.write_text(json.dumps(out, indent=1))

print("[CFO] normalized to subcarrier spacing, B=8", flush=True)
for eps in [0.0, 1e-4, 1e-3, 1e-2]:
    t0 = time.time(); m, s = run(8, 0.0, eps)
    out["cfo"][f"eps{eps}"] = {"eps": eps, "B": 8, "mean": m, "std": s}
    print(f"   eps={eps:<7} B=8: {m*100:6.2f}% +/- {s*100:4.2f}  ({time.time()-t0:.0f}s)", flush=True)
    p.write_text(json.dumps(out, indent=1))
print("done")

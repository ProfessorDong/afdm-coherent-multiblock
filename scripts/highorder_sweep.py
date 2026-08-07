"""Higher-order modulation: SNR sweep with a genie reference.

The paper previously reported a single 16-QAM point (82.7% SER at 15 dB) and
attributed the failure to the hard-decision pseudo-pilot mechanism.  A genie
check (true theta AND true h) shows 22.6% at the same point, so the dominant
limitation is NOT the data-aided loop: linear MMSE equalization of this
doubly-dispersive channel cannot support the 3.2x smaller minimum distance of
16-QAM at 15 dB even with a perfect channel.

This script separates the two effects across SNR:
  genie   : true theta, true h            -> equalizer/operating-point limit
  hard    : receiver, hard pseudo-pilots  -> the receiver as previously published
  soft    : receiver, posterior-mean feedback + pilot-calibrated output

so that the SNR at which 16-QAM becomes usable, and the receiver-to-genie gap,
are both quantified rather than asserted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from afdm.classical import cg_solve
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, block_doppler_phase, sample_multiblock
from afdm.operators import FastAFDMOperator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiblock_dasbl import multiblock_dasbl_receiver

SNRS = [15.0, 20.0, 25.0, 30.0, 35.0]
B_BLOCK = 4
N_SEEDS = 3
N_BATCHES = 4
BATCH_SIZE = 32


def _mask(batch, b):
    m = batch.pilot_mask
    return m[:, b, :] if m.dim() == 3 else m


def genie_ser(cfg, snr, seed):
    system = cfg.system(); ch = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_BLOCK,
                                      constellation=const, device=cfg.device, seed=42)
    g = torch.Generator(device=cfg.device); g.manual_seed(seed)
    acc, n = 0.0, 0
    for _ in range(N_BATCHES):
        batch = sample_multiblock(system, ch, const, pp, pv, batch_size=BATCH_SIZE,
                                  snr_db=snr, generator=g)
        for b in range(B_BLOCK):
            ph = block_doppler_phase(batch.theta_true[..., 1], b, cfg.N, int(cfg.ell_max))
            op = FastAFDMOperator(system=system, ell=batch.theta_true[..., 0],
                                  kappa=batch.theta_true[..., 1], h=batch.h_true * ph)
            sw = batch.sigma_w2_block
            z = cg_solve(lambda v: op.rmatvec(op.matvec(v)) + sw * v,
                         op.rmatvec(batch.y[:, b, :]), max_iter=30)
            hard = ((z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2).argmin(-1)
            mb = _mask(batch, b)
            acc += float(((hard != batch.labels[:, b, :]) * mb).float().sum() / mb.float().sum())
            n += 1
    return acc / n


def rx_ser(cfg, snr, seed, soft, cal):
    system = cfg.system(); ch = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_BLOCK,
                                      constellation=const, device=cfg.device, seed=42)
    g = torch.Generator(device=cfg.device); g.manual_seed(seed)
    acc = 0.0
    for _ in range(N_BATCHES):
        batch = sample_multiblock(system, ch, const, pp, pv, batch_size=BATCH_SIZE,
                                  snr_db=snr, generator=g)
        with torch.no_grad():
            hard, _, _, _ = multiblock_dasbl_receiver(system, batch, const, cfg,
                                                      soft_symbols=soft,
                                                      calibrate_output=cal)
        m = batch.pilot_mask
        acc += float(((hard != batch.labels) * m).float().sum() / m.float().sum())
    return acc / N_BATCHES


def avg(fn):
    v = [fn(k * 137 + 42) for k in range(N_SEEDS)]
    a = np.array(v)
    return float(a.mean()), float(a.std())


def main():
    out = {"snrs": SNRS, "B": B_BLOCK, "P": 5, "N_p": 32,
           "N_seeds": N_SEEDS, "N_batches": N_BATCHES, "batch_size": BATCH_SIZE,
           "results": {}}
    for kind in ["qpsk", "qam16"]:
        cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32,
                               P_max=8, constellation_kind=kind)
        print(f"\n{'='*82}\n{kind.upper()}  (P=5, N_p=32, B={B_BLOCK})\n{'='*82}")
        print(f"{'SNR':>6s} {'genie':>16s} {'rx hard':>16s} {'rx soft+cal':>16s}")
        per = {"genie": [], "hard": [], "soft": [],
               "genie_std": [], "hard_std": [], "soft_std": []}
        for snr in SNRS:
            t0 = time.time()
            gm, gs = avg(lambda s: genie_ser(cfg, snr, s))
            hm, hs = avg(lambda s: rx_ser(cfg, snr, s, soft=False, cal=False))
            sm, ss = avg(lambda s: rx_ser(cfg, snr, s, soft=True, cal=True))
            for k, v in [("genie", gm), ("hard", hm), ("soft", sm),
                         ("genie_std", gs), ("hard_std", hs), ("soft_std", ss)]:
                per[k].append(v)
            print(f"{snr:>5.0f}dB {gm*100:>13.2f}%  {hm*100:>13.2f}%  {sm*100:>13.2f}%"
                  f"   ({time.time()-t0:.0f}s)", flush=True)
        out["results"][kind] = per
    p = Path("runs/highorder_sweep.json"); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()

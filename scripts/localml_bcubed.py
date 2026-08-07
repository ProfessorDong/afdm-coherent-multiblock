"""Attainability of the cubic Doppler bound (Theorem 2).

Computing a CRB and observing it fall as 1/B^3 only verifies the formula. This
script shows the bound is ATTAINABLE: a local maximum-likelihood Doppler
estimator, given the correct coarse Nyquist cell and an unknown complex gain,
achieves RMSE(kappa) ~ B^{-3/2}.

Setup (deliberately clean, to isolate the aperture effect):
  * P = 1 path, pilot-only, no data-aided iteration, no pseudo-pilot error
  * true (ell, kappa) drawn at random; coarse cell given as kappa_0 = kappa + U(-0.3, 0.3)
    i.e. inside one Nyquist cell (1/beta ~ 0.93) but far coarser than the main lobe
  * complex gain unknown (never used by the estimator)
  * estimator: argmax over fine grid of the coherent slow-time statistic
        S(dk) = | sum_b z_b exp(-j 2 pi dk b beta) |^2
    with z_b the per-block correlation at the (known) delay -- exactly Corollary 1's
    array-factor statistic.

Reports RMSE(kappa_hat) vs B and the fitted log-log slope, which should approach
-1.5 (Theorem 2) rather than -0.5 (classical Fisher additivity).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, block_doppler_phase, sample_multiblock

BS = [1, 2, 4, 8, 16, 32]
SNR = 20.0
N_TRIALS = 400
COARSE_ERR = 0.30      # coarse-cell error half-width (inside one Nyquist cell)
FINE_HALF = 0.35       # local search half-window
FINE_STEP = 2e-4       # fine grid step (<< 1/(B beta) even at B=32)


def run_B(cfg, B, seed):
    """Return array of kappa estimation errors over N_TRIALS single-path trials."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    N = cfg.N; N_cp = system.ell_max; beta = (N + N_cp) / N
    device = cfg.device
    pp, pv = PILOT_DESIGNS["repeated"](N=N, N_p=cfg.N_p, B=B,
                                       constellation=const, device=device, seed=42)
    gen = torch.Generator(device=device); gen.manual_seed(seed)

    batch = sample_multiblock(system, channel, const, pp, pv,
                              batch_size=N_TRIALS, snr_db=SNR, generator=gen)
    ell_t = batch.theta_true[..., 0]          # (T,1)
    kap_t = batch.theta_true[..., 1]

    # pilot-only reference waveform per block
    x_p = torch.zeros(N_TRIALS, B, N, dtype=batch.r.dtype, device=device)
    for b in range(B):
        x_p[:, b, pp[b]] = pv[b].unsqueeze(0)
    s_list = [system.idaft(x_p[:, b, :]) for b in range(B)]

    n = torch.arange(N, device=device, dtype=torch.float32)
    ell_int = ell_t[:, 0].round().long().clamp(0, N - 1)

    # coarse cell: true kappa perturbed inside one Nyquist cell
    u = (2.0 * torch.rand(N_TRIALS, device=device, generator=gen) - 1.0)
    kap0 = kap_t[:, 0] + COARSE_ERR * u

    # per-block statistic z_b at the known delay, demodulated at kap0.
    # Both phase terms must be removed: the intra-block ramp exp(j2pi kap (n+N_cp)/N)
    # and the inter-block term exp(j2pi kap b beta) contributed by D_b(kappa).
    # After demodulating both at kap0, z_b ~ alpha * exp(j2pi dkappa b beta).
    z = torch.zeros(N_TRIALS, B, dtype=batch.r.dtype, device=device)
    for b in range(B):
        shift = (n.unsqueeze(0) - ell_int.unsqueeze(1)) % N
        s_sh = torch.gather(s_list[b], 1, shift.long())
        demod = torch.exp(-1j * 2 * torch.pi * kap0.unsqueeze(1)
                          * (n.unsqueeze(0) + N_cp) / N).to(batch.r.dtype)
        inter = torch.exp(-1j * 2 * torch.pi * kap0 * b * beta).to(batch.r.dtype)
        z[:, b] = (batch.r[:, b, :] * s_sh.conj() * demod).sum(dim=-1) * inter

    # local ML: argmax_dk | sum_b z_b exp(-j 2 pi dk b beta) |^2
    n_fine = int(2 * FINE_HALF / FINE_STEP) + 1
    dk = torch.linspace(-FINE_HALF, FINE_HALF, n_fine, device=device, dtype=torch.float32)
    bidx = torch.arange(B, device=device, dtype=torch.float32)
    # phase[f, b]
    ph = torch.exp(-1j * 2 * torch.pi * dk.unsqueeze(1) * bidx.unsqueeze(0) * beta).to(z.dtype)
    score = (z.unsqueeze(1) * ph.unsqueeze(0)).sum(dim=-1).abs() ** 2   # (T, n_fine)
    best = score.argmax(dim=-1)
    kap_hat = kap0 + dk[best]

    return (kap_hat - kap_t[:, 0]).detach().cpu().numpy()


def main():
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=1, N_p=32, P_max=1)
    print(f"\n{'='*76}\nATTAINABILITY OF THE CUBIC DOPPLER BOUND (local ML, P=1, {SNR} dB)")
    print(f"{N_TRIALS} trials/point, coarse-cell error +/-{COARSE_ERR}, fine step {FINE_STEP}")
    print(f"{'='*76}")
    print(f"{'B':>4s}  {'RMSE(kappa)':>13s}  {'1/(B*beta)':>11s}  {'ratio to B=1':>13s}")

    out = {}
    rmses = []
    beta = (128 + 10) / 128
    for B in BS:
        t0 = time.time()
        err = run_B(cfg, B, seed=1234)
        rmse = float(np.sqrt((err ** 2).mean()))
        rmses.append(rmse)
        print(f"{B:>4d}  {rmse:>13.5e}  {1/(B*beta):>11.4f}  "
              f"{rmses[0]/rmse:>13.2f}   ({time.time()-t0:.0f}s)")
        out[str(B)] = {"rmse": rmse}

    # log-log slope
    lb = np.log(np.array(BS, dtype=float)); lr = np.log(np.array(rmses))
    slope = float(np.polyfit(lb, lr, 1)[0])
    # slope over the well-resolved tail only (B>=4)
    m = np.array(BS) >= 4
    slope_tail = float(np.polyfit(lb[m], lr[m], 1)[0])
    print(f"\nlog-log slope (all B) : {slope:+.3f}")
    print(f"log-log slope (B>=4)  : {slope_tail:+.3f}")
    print(f"  Theorem 2 predicts  : -1.500")
    print(f"  classical 1/sqrt(B) : -0.500")
    out["slope_all"] = slope; out["slope_tail"] = slope_tail

    p = Path("runs/localml_bcubed.json"); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"snr_db": SNR, "n_trials": N_TRIALS, "coarse_err": COARSE_ERR,
               "fine_step": FINE_STEP, "results": out}, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()

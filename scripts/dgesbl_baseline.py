"""D-GESBL-style baseline: data-aided grid-evolution MMV SBL across blocks.

This is an ADAPTATION, not a reimplementation. D-GESBL (Luo et al., IEEE TCOM
2026, arXiv:2607.18881) formulates AFDM channel estimation as a multiple
measurement vector (MMV) off-grid sparse-recovery problem under a SUPERIMPOSED
pilot framework, with reliably decoded symbols fed back as pseudo-pilots and a
grid-evolution step that adjusts the virtual DAF-domain grid.

We reproduce the three algorithmic ingredients that matter for our comparison:

  (i)   MMV sparse recovery with support SHARED across the B blocks and
        FREE per-block complex gains h_b (the standard MMV model);
  (ii)  data-aided pseudo-pilot feedback with reliability gating;
  (iii) grid evolution on (ell, kappa).

and adapt the frame: the superimposed-pilot structure is replaced by the same
embedded/hopping pilots every other receiver in this paper uses, so all methods
see IDENTICAL observations, pilots, channels and noise. The GAMP variant of
D-GESBL is a complexity optimization, not an accuracy one, and is not needed for
an SER comparison.

The scientifically important distinction this baseline isolates: MMV shares the
SUPPORT across blocks but lets each block have its own gain, so the deterministic
inter-block Doppler phase D_b(kappa) is ABSORBED into h_b rather than exploited.
Our MB-IDAR instead constrains h_b = D_b(kappa) h and estimates the shared h.
This is exactly the coherent-versus-noncoherent aperture question.

To keep the comparison fair the baseline is tuned over its own hyperparameters
and reported at its BEST configuration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
from afdm.operators import FastAFDMOperator


def dgesbl_receiver(system, batch, const, cfg, T_em=15, T_grid=3, grid_lr=0.05,
                    rho_min=0.5, magnitude_ratio=0.05, K_cg=15):
    """Data-aided grid-evolution MMV SBL. Shared support, free per-block gains."""
    r = batch.r; y = batch.y
    B_batch, B_block, N = r.shape
    dtype = r.dtype; device = r.device
    sigma_w2 = batch.sigma_w2_block
    pp = batch.pilot_positions; pv = batch.pilot_values

    # --- candidate support from the same CFAR front end all receivers use ---
    sr = cfg.support_recovery()
    x_pilot = torch.zeros(B_batch, B_block, N, dtype=dtype, device=device)
    for b in range(B_block):
        x_pilot[:, b, pp[b]] = pv[b].unsqueeze(0)
    s0 = system.idaft(x_pilot[:, 0, :])
    ell, kappa, _ = sr(r[:, 0, :], s0)                     # (Bb, Q) shared support
    Q = ell.shape[1]
    gamma = torch.ones(B_batch, Q, device=device)          # shared ARD precisions

    x_hat = x_pilot.clone()

    def e_step(A, gam):
        """Per-block posterior of h_b under the shared ARD prior."""
        AH = A.conj().transpose(-1, -2)
        G = AH @ A + sigma_w2 * torch.diag_embed(gam.to(A.dtype))
        rhs = (AH @ r[:, blk, :].unsqueeze(-1))
        mu = torch.linalg.solve(G, rhs).squeeze(-1)
        Sig = torch.linalg.inv(G) * sigma_w2
        return mu, torch.diagonal(Sig, dim1=-2, dim2=-1).real.clamp(min=0)

    for it in range(T_em):
        mus, dcs = [], []
        for blk in range(B_block):
            A = build_regression_matrix(system, ell, kappa, x_hat[:, blk, :])
            mu, dc = e_step(A, gamma)
            mus.append(mu); dcs.append(dc)
        MU = torch.stack(mus, 1)                            # (Bb, B, Q)
        DC = torch.stack(dcs, 1)
        # --- MMV M-step: pool the B measurement vectors into one shared gamma ---
        gamma = B_block / ((MU.abs() ** 2 + DC).sum(dim=1) + 1e-12)

        # --- grid evolution (shared support) ---
        if T_grid > 0 and it % 3 == 2:
            eps = 1e-3
            g_e = torch.zeros_like(ell); g_k = torch.zeros_like(kappa)
            for blk in range(B_block):
                base = build_regression_matrix(system, ell, kappa, x_hat[:, blk, :])
                res = r[:, blk, :] - (base @ MU[:, blk, :].unsqueeze(-1)).squeeze(-1)
                for q in range(Q):
                    ep = ell.clone(); ep[:, q] += eps
                    kp = kappa.clone(); kp[:, q] += eps
                    de = (build_regression_matrix(system, ep, kappa, x_hat[:, blk, :])[..., q]
                          - base[..., q]) / eps
                    dk = (build_regression_matrix(system, ell, kp, x_hat[:, blk, :])[..., q]
                          - base[..., q]) / eps
                    g_e[:, q] += (res.conj() * de * MU[:, blk, q:q+1]).real.sum(-1)
                    g_k[:, q] += (res.conj() * dk * MU[:, blk, q:q+1]).real.sum(-1)
            ell = (ell + grid_lr * g_e / (g_e.abs().amax(dim=-1, keepdim=True) + 1e-9)).clamp(0, cfg.ell_max)
            kappa = (kappa + grid_lr * g_k / (g_k.abs().amax(dim=-1, keepdim=True) + 1e-9)).clamp(-cfg.kappa_max, cfg.kappa_max)

        # --- data-aided pseudo-pilots with reliability gating ---
        if it >= 3:
            for blk in range(B_block):
                op = FastAFDMOperator(system=system, ell=ell, kappa=kappa, h=MU[:, blk, :])
                def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
                z = cg_solve(mv, op.rmatvec(y[:, blk, :]), max_iter=K_cg)
                d = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
                p = F.softmax(-d / max(sigma_w2, 1e-9), dim=-1)
                hard = p.argmax(-1); rho = p.max(-1).values
                xb = torch.zeros(B_batch, N, dtype=dtype, device=device)
                keep = rho >= rho_min
                xb[keep] = const[hard[keep]]
                xb[:, pp[blk]] = pv[blk].unsqueeze(0)
                x_hat[:, blk, :] = xb

    # --- final detection per block ---
    hard_out = torch.zeros(B_batch, B_block, N, dtype=torch.long, device=device)
    for blk in range(B_block):
        A = build_regression_matrix(system, ell, kappa, x_hat[:, blk, :])
        mu, _ = e_step(A, gamma)
        op = FastAFDMOperator(system=system, ell=ell, kappa=kappa, h=mu)
        def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
        z = cg_solve(mv, op.rmatvec(y[:, blk, :]), max_iter=30)
        hard_out[:, blk, :] = ((z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2).argmin(-1)
    return hard_out


def eval_dgesbl(cfg, snr, B_block, seed, n_batches, batch_size, **kw):
    system = cfg.system(); ch = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_block,
                                      constellation=const, device=cfg.device, seed=42)
    g = torch.Generator(device=cfg.device); g.manual_seed(seed); acc = 0.0
    for _ in range(n_batches):
        batch = sample_multiblock(system, ch, const, pp, pv, batch_size=batch_size,
                                  snr_db=snr, generator=g)
        with torch.no_grad():
            hard = dgesbl_receiver(system, batch, const, cfg, **kw)
        m = batch.pilot_mask
        acc += float(((hard != batch.labels) * m).float().sum() / m.float().sum())
    return acc / n_batches

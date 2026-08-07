"""Nuisance-aware Cramer-Rao bound for the multi-block AFDM model.

The CRB reported in Table V / Fig 8 is an ORACLE-CONDITIONAL bound: it assumes
(h, x_{1:B}) known and treats each path separately.  Theorem 2 is a statement
about the NUISANCE-ELIMINATED information

    J_{kappa,eff} = J_{kappa kappa} - J_{kappa eta} J_{eta eta}^{-1} J_{eta kappa},
    eta = [ell^T, Re h^T, Im h^T]^T,

so the conditional bound does not verify it.  This script computes the full
joint FIM over the 4P real parameters

    [ ell_1..ell_P , kappa_1..kappa_P , Re h_1..h_P , Im h_1..h_P ]

from the stacked physical multi-block mean

    mu_b = ( A_b(ell,kappa,x_b) .* D_b(kappa) ) h ,   b = 0..B-1,

and forms the Schur complement explicitly.

Three quantities are reported per B:
  CRB_cond : known-gain conditional bound (invert the (ell,kappa) block only)
  CRB_eff  : nuisance-eliminated bound (Schur complement over ell, Re h, Im h)
  ratio    : CRB_eff / CRB_cond

Theorem 2 predicts that eliminating the unknown complex amplitude replaces
sum_b b^2 ~ B^3/3 by sum_b (b - bbar)^2 ~ B^3/12, i.e. a FOURFOLD reduction of
the leading coefficient with the exponent unchanged.  For an isolated path the
ratio CRB_eff / CRB_cond must therefore approach 4 as B grows, while CRB_eff
itself must still fall as 1/B^3.  Both are checked directly.

Derivatives w.r.t. (ell, kappa) are central finite differences (O(eps^2));
derivatives w.r.t. (Re h, Im h) are analytic, since mu is linear in h.  The
step size is validated by recomputing at eps and eps/2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from afdm.classical import build_regression_matrix
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, block_doppler_phase, sample_multiblock

SNR = 15.0
BS = [1, 2, 4, 8, 16]


def stacked_mean(system, ell, kap, h, x_ref, B_block, N, N_cp):
    """mu = [mu_0; ...; mu_{B-1}], mu_b = (A_b .* D_b(kappa)) h.  (Bb, B*N) complex."""
    outs = []
    for b in range(B_block):
        A = build_regression_matrix(system, ell, kap, x_ref[:, b, :])   # (Bb, N, P)
        phase_b = block_doppler_phase(kap, b, N, N_cp)                  # (Bb, P)
        outs.append(((A * phase_b.unsqueeze(1)) @ h.unsqueeze(-1)).squeeze(-1))
    return torch.cat(outs, dim=-1)


def atom_columns(system, ell, kap, x_ref, B_block, N, N_cp):
    """Stacked phase-corrected atom matrix Atil, so that mu = Atil h.  (Bb, B*N, P)."""
    outs = []
    for b in range(B_block):
        A = build_regression_matrix(system, ell, kap, x_ref[:, b, :])
        phase_b = block_doppler_phase(kap, b, N, N_cp)
        outs.append(A * phase_b.unsqueeze(1))
    return torch.cat(outs, dim=1)


def joint_fim(system, ell, kap, h, x_ref, B_block, N, N_cp, sigma_w2, eps=1e-3):
    """Full 4P x 4P real FIM for [ell, kappa, Re h, Im h], per batch element.

    J = (2/sigma^2) Re{ D^H D },  D = d mu / d params  (B*N x 4P complex).
    """
    Bb, P = ell.shape
    M = B_block * N
    D = torch.zeros(Bb, M, 4 * P, dtype=torch.complex128, device=ell.device)

    # --- ell and kappa: central differences ---
    for i in range(P):
        ep = ell.clone(); ep[:, i] += eps
        em = ell.clone(); em[:, i] -= eps
        D[:, :, i] = ((stacked_mean(system, ep, kap, h, x_ref, B_block, N, N_cp)
                       - stacked_mean(system, em, kap, h, x_ref, B_block, N, N_cp))
                      / (2 * eps)).to(torch.complex128)
        kp = kap.clone(); kp[:, i] += eps
        km = kap.clone(); km[:, i] -= eps
        D[:, :, P + i] = ((stacked_mean(system, ell, kp, h, x_ref, B_block, N, N_cp)
                           - stacked_mean(system, ell, km, h, x_ref, B_block, N, N_cp))
                          / (2 * eps)).to(torch.complex128)

    # --- Re h, Im h: analytic (mu is linear in h) ---
    Atil = atom_columns(system, ell, kap, x_ref, B_block, N, N_cp).to(torch.complex128)
    D[:, :, 2 * P:3 * P] = Atil
    D[:, :, 3 * P:4 * P] = 1j * Atil

    J = (2.0 / sigma_w2) * torch.real(D.conj().transpose(-2, -1) @ D)
    return J


def crb_pair(J, P):
    """Return (CRB_cond, CRB_eff) for the kappa coordinates, per batch element.

    CRB_cond : invert the 2P x 2P (ell, kappa) block  -> gain and data known.
    CRB_eff  : Schur-complement out eta = (ell, Re h, Im h), then invert.
    """
    Bb = J.shape[0]
    kap_idx = list(range(P, 2 * P))
    eta_idx = list(range(0, P)) + list(range(2 * P, 4 * P))
    lk_idx = list(range(0, 2 * P))

    ell_idx = list(range(0, P))
    # nuisance sets: for kappa, eliminate (ell, Re h, Im h); for ell, eliminate (kappa, Re h, Im h)
    eta_l_idx = list(range(P, 2 * P)) + list(range(2 * P, 4 * P))

    cond = torch.zeros(Bb, P, dtype=torch.float64)
    eff = torch.zeros(Bb, P, dtype=torch.float64)
    cond_l = torch.zeros(Bb, P, dtype=torch.float64)
    eff_l = torch.zeros(Bb, P, dtype=torch.float64)

    def schur_crb(Jb, tgt, nui):
        Jtt = Jb[np.ix_(tgt, tgt)]
        Jtn = Jb[np.ix_(tgt, nui)]
        Jnn = Jb[np.ix_(nui, nui)]
        Jef = Jtt - Jtn @ torch.linalg.solve(Jnn, Jtn.transpose(-2, -1))
        return torch.diagonal(torch.linalg.inv(Jef)).cpu()

    for b in range(Bb):
        Jb = J[b]
        # known-gain conditional: invert the (ell, kappa) block only
        Clk = torch.linalg.inv(Jb[np.ix_(lk_idx, lk_idx)])
        d = torch.diagonal(Clk).cpu()
        cond_l[b] = d[0:P]
        cond[b] = d[P:2 * P]
        # nuisance-eliminated (gain unknown)
        eff[b] = schur_crb(Jb, kap_idx, eta_idx)
        eff_l[b] = schur_crb(Jb, ell_idx, eta_l_idx)
    return cond, eff, cond_l, eff_l


def run(cfg, label, n_batches=4, batch_size=8, seed=42, eps=1e-3, verbose=True):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    N = cfg.N; N_cp = int(cfg.ell_max)
    out = {}
    if verbose:
        print(f"\n{'='*88}\n{label}   (SNR={SNR} dB, eps={eps})\n{'='*88}")
        print(f"{'B':>3s} {'CRB_cond(kap)':>15s} {'CRB_eff(kap)':>15s} {'ratio':>8s} "
              f"{'RMSE_cond':>11s} {'RMSE_eff':>11s}")
    for B_block in BS:
        pp, pv = PILOT_DESIGNS["hopping"](N=N, N_p=cfg.N_p, B=B_block,
                                          constellation=const, device=cfg.device, seed=42)
        gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
        cs, es, ls, ls2 = [], [], [], []
        for _ in range(n_batches):
            batch = sample_multiblock(system, channel, const, pp, pv,
                                      batch_size=batch_size, snr_db=SNR, generator=gen)
            J = joint_fim(system, batch.theta_true[..., 0], batch.theta_true[..., 1],
                          batch.h_true, batch.x_true, B_block, N, N_cp,
                          batch.sigma_w2_block, eps=eps)
            c, e, cl, el = crb_pair(J, cfg.P)
            cs.append(c.flatten()); es.append(e.flatten())
            ls.append(cl.flatten()); ls2.append(el.flatten())
        def m_(t):
            t = torch.cat(t); t = t[torch.isfinite(t) & (t > 0)]; return float(t.mean())
        cm, em, clm, elm = m_(cs), m_(es), m_(ls), m_(ls2)
        out[str(B_block)] = {"crb_cond": cm, "crb_eff": em, "ratio": em / cm,
                             "rmse_cond": cm ** 0.5, "rmse_eff": em ** 0.5,
                             "crb_ell_cond": clm, "crb_ell_eff": elm,
                             "rmse_ell_eff": elm ** 0.5}
        if verbose:
            print(f"{B_block:>3d} {cm:>15.4e} {em:>15.4e} {em/cm:>8.2f} "
                  f"{cm**0.5:>11.4e} {em**0.5:>11.4e}", flush=True)
    return out


def scaling_report(res, name):
    Bs = sorted(int(k) for k in res)
    print(f"\n  [{name}] successive-doubling reduction of CRB_eff (theory 8x for 1/B^3):")
    for a, b in zip(Bs[:-1], Bs[1:]):
        r = res[str(a)]["crb_eff"] / res[str(b)]["crb_eff"]
        print(f"    B {a:>2d} -> {b:>2d}:  {r:6.2f}x")
    lo, hi = Bs[0], Bs[-1]
    tot = res[str(lo)]["crb_eff"] / res[str(hi)]["crb_eff"]
    print(f"    overall B {lo} -> {hi}: {tot:.1f}x   (B^3 = {(hi/lo)**3:.0f})")
    lb = np.log(np.array([res[str(b)]["crb_eff"] for b in Bs]))
    slope = np.polyfit(np.log(np.array(Bs, float)), lb, 1)[0]
    print(f"    log-log slope of CRB_eff vs B: {slope:.3f}   (theory -3)")
    return float(slope), float(tot)


def main():
    torch.manual_seed(0)

    # --- (1) Isolated-path validation: ratio must approach 4 (Theorem 2) ---
    cfg1 = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=1, N_p=32, P_max=4)
    r1 = run(cfg1, "VALIDATION: isolated path (P=1) -- ratio must -> 4")
    s1, t1 = scaling_report(r1, "P=1")

    # --- (2) Operating point of Table V ---
    cfg5 = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)
    r5 = run(cfg5, "HARD (P=5, N_p=16) -- operating point of Table V")
    s5, t5 = scaling_report(r5, "P=5")

    # --- (3) finite-difference step validation ---
    print(f"\n{'='*88}\nSTEP-SIZE VALIDATION (eps vs eps/2, HARD, B=4)\n{'='*88}")
    chk = {}
    for eps in (1e-3, 5e-4):
        g = run(cfg5, f"eps={eps}", n_batches=1, batch_size=4, eps=eps, verbose=False)
        chk[str(eps)] = g["4"]["crb_eff"]
        print(f"  eps={eps:<8g}  CRB_eff(B=4) = {g['4']['crb_eff']:.6e}")
    rel = abs(chk["0.001"] - chk["0.0005"]) / chk["0.0005"]
    print(f"  relative change: {rel:.2%}  ({'OK' if rel < 0.05 else 'TOO LARGE'})")

    p = Path("runs/crb_nuisance.json"); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"snr_db": SNR, "Bs": BS,
                   "isolated_P1": r1, "hard_P5": r5,
                   "slope_P1": s1, "slope_P5": s5,
                   "total_P1": t1, "total_P5": t5,
                   "eps_check": {"values": chk, "rel_change": rel}}, f, indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()

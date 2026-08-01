"""Data-aided ambiguity function: use x_true or reliable x_hat instead of pilots.

Hypothesis: single-block ambiguity function using pilot-only signal has side
lobes from data self-interference. Using the TRUE x (or reliably decoded x)
should sharpen peaks and improve CFAR support recovery.

Test: for the HARD (P=5, N_p=16) config, compare support recall using
  (a) pilot-only ambiguity function (current baseline)
  (b) pilots + reliable x_hat (from post-DASBL detection)
  (c) x_true oracle (upper bound)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.operators import FastAFDMOperator
from afdm.support import ambiguity_function, cfar_peaks, newton_refine
from afdm.training import sample_batch
from afdm.vem import safeguarded_lm_theta_step


TOL_ELL = 0.75; TOL_KAP = 0.75


def compute_recall_rmse(ell_hat, kap_hat, ell_true, kap_true):
    """Fraction of true paths within tolerance of any detected candidate + RMSE on matched."""
    B = ell_hat.shape[0]
    n_tp = 0; n_true = 0; de_acc = 0.0; dk_acc = 0.0
    for b in range(B):
        for p in range(ell_true.shape[1]):
            n_true += 1
            de = (ell_hat[b] - ell_true[b, p]).abs()
            dk = (kap_hat[b] - kap_true[b, p]).abs()
            mask = (de <= TOL_ELL) & (dk <= TOL_KAP)
            if mask.any():
                n_tp += 1
                i = mask.nonzero()[0].item()
                de_acc += float((ell_hat[b, i] - ell_true[b, p]) ** 2)
                dk_acc += float((kap_hat[b, i] - kap_true[b, p]) ** 2)
    return {
        "recall": n_tp / max(n_true, 1),
        "de_rmse": (de_acc / max(n_tp, 1)) ** 0.5,
        "dk_rmse": (dk_acc / max(n_tp, 1)) ** 0.5,
    }


def ambiguity_with_signal(r, s_signal, system, kappa_max, oversample_doppler=2):
    """Same as ambiguity_function but with an arbitrary signal s_signal (not just pilot)."""
    return ambiguity_function(r, s_signal, N=system.N, N_cp=int(system.ell_max),
                              kappa_max=kappa_max, ell_max=float(system.ell_max),
                              oversample_doppler=oversample_doppler)


def eval_variant(cfg, snr_db, variant: str, n_batches=8, batch_size=16, K_cfar=None):
    """variant in {"pilot", "pilot+reliable_xhat", "xtrue"}"""
    K_cfar = K_cfar or (cfg.P + 3)
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(42)

    metrics_list = []
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        B, N = batch["r"].shape
        dtype = batch["r"].dtype; device = batch["r"].device

        if variant == "pilot":
            x_signal = torch.zeros(B, N, dtype=dtype, device=device)
            x_signal[:, pp] = pv.unsqueeze(0)
            s_signal = system.idaft(x_signal)   # (B, N)
        elif variant == "xtrue":
            s_signal = system.idaft(batch["x_true"])
        elif variant == "pilot+reliable_xhat":
            # Run a quick pilot-only iterative DASBL to get reliable x_hat, then
            # use it in ambiguity function.
            # ---- pilot-only iterative DASBL: 3 outer, 1 LM per outer ----
            def solve_h(x_ref, ell_c, kap_c):
                A = build_regression_matrix(system, ell_c, kap_c, x_ref)
                AH = A.conj().transpose(-1, -2)
                AhA = AH @ A
                Ahr = (AH @ batch["r"].unsqueeze(-1)).squeeze(-1)
                P = ell_c.shape[1]
                ridge = 1e-3 * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
                return torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)
            # Initial CFAR
            x_p = torch.zeros(N, dtype=dtype, device=device); x_p[pp] = pv
            s_p = system.idaft(x_p.unsqueeze(0))[0]
            A_p, e_g, k_g = ambiguity_function(batch["r"], s_p, N=N, N_cp=int(cfg.ell_max),
                                               kappa_max=cfg.kappa_max, ell_max=float(cfg.ell_max))
            peak_idx, _ = cfar_peaks(A_p, K=cfg.P_max, min_separation=2)
            ell_hat0, kap_hat0 = newton_refine(A_p, peak_idx, e_g, k_g, max_iter=2)
            x_hat = torch.zeros(B, N, dtype=dtype, device=device); x_hat[:, pp] = pv.unsqueeze(0)
            h_hat = solve_h(x_hat, ell_hat0, kap_hat0)
            omega = 1.0 / max(batch["sigma_w2_block"], 1e-6)
            for _ in range(3):
                op = FastAFDMOperator(system=system, ell=ell_hat0, kappa=kap_hat0, h=h_hat)
                def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
                z = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=30)
                dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
                p_ms = F.softmax(-omega * dists, dim=-1)
                hard = p_ms.argmax(dim=-1)
                rho = p_ms.max(dim=-1).values
                reliable = rho >= 0.9
                x_hat = torch.zeros(B, N, dtype=dtype, device=device)
                x_hat[reliable] = const[hard[reliable]]
                x_hat[:, pp] = pv.unsqueeze(0)
                h_hat = solve_h(x_hat, ell_hat0, kap_hat0)
                ell_hat0, kap_hat0, _ = safeguarded_lm_theta_step(
                    system, batch["r"], h_hat, x_hat, ell_hat0, kap_hat0,
                    sigma_w2=batch["sigma_w2_block"], v_h=None,
                    gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4,
                )
            s_signal = system.idaft(x_hat)
        else:
            raise ValueError(variant)

        # Compute ambiguity with s_signal
        # Note: ambiguity_function assumes s_signal shape (B, N) per batch
        A_amb, e_g, k_g = ambiguity_with_signal(batch["r"], s_signal, system, cfg.kappa_max)
        peak_idx, _ = cfar_peaks(A_amb, K=K_cfar, min_separation=2)
        ell_hat, kap_hat = newton_refine(A_amb, peak_idx, e_g, k_g, max_iter=2)
        m = compute_recall_rmse(ell_hat, kap_hat,
                                batch["theta_true"][..., 0], batch["theta_true"][..., 1])
        metrics_list.append(m)

    agg = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
    return agg


def main():
    print("=" * 78)
    print("DATA-AIDED AMBIGUITY: how much does using x_hat sharpen peaks?")
    print("=" * 78)
    for name, cfg in (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
        ("HARD (P=5, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)),
    ):
        print(f"\n{name}")
        print(f"{'variant':<25s}  {'recall':>7s}  {'de_rmse':>7s}  {'dk_rmse':>7s}")
        for variant in ("pilot", "pilot+reliable_xhat", "xtrue"):
            m = eval_variant(cfg, snr_db=15.0, variant=variant)
            print(f"{variant:<25s}  {m['recall']:>7.1%}  {m['de_rmse']:>7.3f}  {m['dk_rmse']:>7.3f}")


if __name__ == "__main__":
    main()

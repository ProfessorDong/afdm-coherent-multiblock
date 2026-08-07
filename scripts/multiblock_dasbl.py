"""Multi-block iterative DASBL with SHARED theta across B blocks.

Key idea: same physical channel over a coherence window, different pilot masks
per block. Multi-block observations reduce theta estimator variance by ~1/B.

Pipeline (given B blocks):
  0. Initial theta from multi-block CFAR (sum |A_b|^2 across blocks).
  1. Multi-block pilot-only LS h.
  2. For outer iter t = 1..T:
     a. For each block, detect symbols with shared (theta, h).
     b. reliable_b = (rho_b >= rho_min)
     c. x_hat_b = pilots_b + reliable-hard_b
     d. Multi-block LS h (using x_hat_b per block).
     e. LM refinement of shared theta using stacked objective:
        L = sum_b |r_b - A_b(theta, x_hat_b) h|^2
     f. Detect again.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, block_doppler_phase, sample_multiblock
from afdm.operators import FastAFDMOperator
from afdm.support import ambiguity_function, ambiguity_function_complex, cfar_peaks, newton_refine


def aperture_synthesis_kappa_refine(system, r, x_hats_2d, ell_hat, kap_hat, N_cp,
                                     kappa_window=0.30, kappa_step=0.003, beta=None):
    """Multi-scale aperture-synthesis coherent local search for kappa.

    For each peak (ell_hat_p, kap_hat_p), evaluate the coherent stacked matched-
    filter output on a fine kappa grid within [kap_hat_p - window, kap_hat_p +
    window]. The window is chosen well below the Doppler-Nyquist ambiguity
    Delta_kappa = 1/beta so that the main lobe of the coherent surface is
    unambiguously located.

    Uses the AFDM atom a_b(ell, kappa)[n] = idaft(x_hats_2d[b])[(n - ell) mod N]
    * exp(j 2 pi kappa (n + N_cp)/N). The coherent stacked score at candidate
    kappa_c is:
        S(kappa_c) = | sum_b <r_b, a_b(ell_hat, kappa_c)> exp(-j 2 pi kappa_c b beta) |^2

    Returns refined kap_hat with same shape as kap_hat.

    Complexity: O(B * K * N * N_fine) per batch element.
    """
    B_batch, B_block, N = r.shape
    dtype = r.dtype; device = r.device
    K = ell_hat.shape[-1]
    if beta is None:
        beta = (N + N_cp) / N

    # Precompute per-block time-domain pilot/data signal s_b = idaft(x_hats_2d[b])
    s_list = [system.idaft(x_hats_2d[:, b, :]) for b in range(B_block)]

    # Fine kappa grid (relative offsets)
    N_fine = int(2 * kappa_window / kappa_step) + 1
    offsets = torch.linspace(-kappa_window, kappa_window, N_fine, device=device, dtype=torch.float32)

    # For each block b, precompute z_b(ell) := r_b * conj(s_b_shifted_by_ell) —
    # but shifted per path. To keep it simple, use nearest-integer ell for the
    # shift and let coherent phase absorb residual fractional-delay error.
    n = torch.arange(N, device=device, dtype=torch.float32)
    ell_int = ell_hat.round().long().clamp(0, N - 1)                # (B_batch, K)

    refined_kap = kap_hat.detach().clone()

    for k in range(K):
        # kappa_candidates_k: (B_batch, N_fine)
        kappa_candidates = kap_hat[:, k:k+1] + offsets.unsqueeze(0)   # (B_batch, N_fine)

        # Accumulate coherent stack score across blocks
        S_stack = torch.zeros(B_batch, N_fine, dtype=dtype, device=device)
        for b in range(B_block):
            # Shift s_b by ell_int[:, k]: s_b_shifted[batch, n] = s_b[batch, (n - ell_int_bk) mod N]
            shift_bk = ell_int[:, k]                                  # (B_batch,)
            # Build shifted signal via advanced indexing
            n_shifted = (n.unsqueeze(0) - shift_bk.unsqueeze(1)) % N  # (B_batch, N)
            s_b_shifted = torch.gather(s_list[b], 1, n_shifted.long())
            # z_b = r_b * conj(s_b_shifted)                           # (B_batch, N)
            z_b = r[:, b, :] * s_b_shifted.conj()
            # Correlation at each candidate kappa: sum_n z_b[n] exp(-j 2π κ_c (n + N_cp)/N)
            # phase: (B_batch, N_fine, N)
            phase = torch.exp(
                -1j * 2 * torch.pi * kappa_candidates.unsqueeze(-1)
                * (n + N_cp).unsqueeze(0).unsqueeze(0) / N
            ).to(dtype)
            corr_b = (z_b.unsqueeze(1) * phase).sum(dim=-1)           # (B_batch, N_fine)
            # Align to hypothesis for coherent stacking
            align = torch.exp(
                -1j * 2 * torch.pi * kappa_candidates * b * beta
            ).to(dtype)                                                # (B_batch, N_fine)
            S_stack = S_stack + corr_b * align

        score = S_stack.abs() ** 2                                     # (B_batch, N_fine)
        best_idx = score.argmax(dim=-1)                                # (B_batch,)
        refined_kap[:, k] = kappa_candidates[torch.arange(B_batch, device=device), best_idx]

    return refined_kap


def multiblock_ls_gains_data_aided(system, batch, ell, kap, x_hats, lambda_ridge=1e-3):
    """Solve multi-block LS with SHARED (ell, kap) and per-block x_hats.

    Uses phase-corrected atoms A_b(theta) D_b(kappa) to account for the
    inter-block Doppler phase evolution h_b = h * D_b(kappa).
    """
    B_batch, B_block, N = batch.r.shape
    dtype = batch.r.dtype; device = batch.r.device
    P = ell.shape[1]
    N_cp = system.ell_max
    AhA_sum = torch.zeros(B_batch, P, P, dtype=dtype, device=device)
    Ahr_sum = torch.zeros(B_batch, P, dtype=dtype, device=device)
    for b in range(B_block):
        A = build_regression_matrix(system, ell, kap, x_hats[:, b, :])
        phase_b = block_doppler_phase(kap, b, N, N_cp)             # (B_batch, P)
        A_b = A * phase_b.unsqueeze(1)                             # broadcast over N
        AH = A_b.conj().transpose(-1, -2)
        AhA_sum += AH @ A_b
        Ahr_sum += (AH @ batch.r[:, b, :].unsqueeze(-1)).squeeze(-1)
    ridge = lambda_ridge * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
    return torch.linalg.solve(AhA_sum + ridge, Ahr_sum.unsqueeze(-1)).squeeze(-1)


def multiblock_lm_theta(system, batch, ell, kap, h, x_hats, sigma_w2,
                         gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4):
    """Safeguarded LM step for SHARED theta using stacked B-block objective.

    L(theta) = -(1/N) sum_b |r_b - A_b(theta, x_b) h|^2

    Backtracks per-batch as in single-block version.
    """
    B_batch, B_block, N = batch.r.shape
    device = ell.device
    N_cp = system.ell_max
    ell_v = ell.detach().clone().requires_grad_(True)
    kap_v = kap.detach().clone().requires_grad_(True)

    def obj():
        # Sum residual squared over blocks using phase-corrected atoms.
        loss_per_b = torch.zeros(B_batch, device=device)
        for b in range(B_block):
            A = build_regression_matrix(system, ell_v, kap_v, x_hats[:, b, :])
            phase_b = block_doppler_phase(kap_v, b, N, N_cp)
            A_b = A * phase_b.unsqueeze(1)
            r_hat = (A_b @ h.unsqueeze(-1)).squeeze(-1)
            res = batch.r[:, b, :] - r_hat
            loss_per_b = loss_per_b + (res.abs() ** 2).sum(dim=-1) / N
        return -loss_per_b   # negative so higher = better

    with torch.enable_grad():
        Q_old = obj()
        Q_sum = Q_old.sum()
    ge, gk = torch.autograd.grad(Q_sum, [ell_v, kap_v])
    Q_old = Q_old.detach()
    step_e = torch.clamp(gamma_lr * ge.detach(), min=-max_step, max=max_step)
    step_k = torch.clamp(gamma_lr * gk.detach(), min=-max_step, max=max_step)
    ell_best = ell.detach().clone(); kap_best = kap.detach().clone()
    accepted = torch.zeros(B_batch, dtype=torch.bool, device=device)
    for i in range(max_backtracks + 1):
        scale = 0.5 ** i
        ce = (ell.detach() + scale * step_e).clamp(min=0.0, max=system.ell_max)
        ck = (kap.detach() + scale * step_k).clamp(min=-system.kappa_max, max=system.kappa_max)
        with torch.no_grad():
            # Recompute obj at candidate with phase-corrected atoms
            Q_c = torch.zeros(B_batch, device=device)
            for b in range(B_block):
                A = build_regression_matrix(system, ce, ck, x_hats[:, b, :])
                phase_b = block_doppler_phase(ck, b, N, N_cp)
                A_b = A * phase_b.unsqueeze(1)
                r_hat = (A_b @ h.unsqueeze(-1)).squeeze(-1)
                Q_c = Q_c + (batch.r[:, b, :] - r_hat).abs().pow(2).sum(dim=-1) / N
            Q_c = -Q_c
        improved = (Q_c >= Q_old - slack) & (~accepted)
        if improved.any():
            for b_idx in torch.where(improved)[0].tolist():
                ell_best[b_idx] = ce[b_idx]
                kap_best[b_idx] = ck[b_idx]
            accepted = accepted | improved
        if accepted.all():
            break
    return ell_best.detach(), kap_best.detach()


def multiblock_dasbl_receiver(system, batch, const, cfg,
                              n_outer=6, n_lm_per_outer=3, rho_min=0.5,
                              use_reacq=True, lambda_ridge=1e-3, use_veff=True,
                              soft_symbols=False, calibrate_output=False,
                              use_aperture=True,
                              gamma_lr=0.5, max_step=0.15, slack=1e-4,
                              kappa_window=0.30, kappa_step=0.003, K_cg=30):
    """Multi-block iterative DASBL with data-aided reacquisition and shared theta.

    soft_symbols: feed back the posterior mean E[x_m | y] instead of a thresholded
    hard decision.  The hard rule sets unreliable symbols to zero, which is a crude
    two-level approximation of exactly this shrinkage; for non-constant-modulus
    constellations a single scalar threshold is not meaningful across amplitude
    levels, and the hard rule fails (16-QAM).  The posterior mean shrinks toward the
    constellation centroid in proportion to uncertainty, so no threshold is needed,
    and the residual-based v_eff automatically absorbs the residual symbol variance
    E||A(theta, x - xbar) h||^2 that the feedback leaves behind.
    """
    r = batch.r; y = batch.y
    B_batch, B_block, N = r.shape
    dtype = r.dtype; device = r.device
    pp = batch.pilot_positions; pv = batch.pilot_values
    sigma_w2 = batch.sigma_w2_block

    # 1. Multi-block ambiguity for initial CFAR: noncoherent sum |A_b|² across
    # blocks (unambiguous in κ). The block-dependent Doppler phase is later
    # accounted for in the LS/LM refinement via phase-corrected atoms.
    N_cp_int = int(cfg.ell_max)
    beta = (N + N_cp_int) / N
    def multi_block_ambiguity(x_hats_2d):
        """Noncoherent multi-block ambiguity for initial support estimation."""
        A_sum = None; e_g = None; k_g = None
        for b in range(B_block):
            s_b = system.idaft(x_hats_2d[:, b, :])
            A_b, e_g, k_g = ambiguity_function(
                r[:, b, :], s_b, N=N, N_cp=N_cp_int,
                kappa_max=cfg.kappa_max, ell_max=float(cfg.ell_max),
                oversample_doppler=2,
            )
            A_sum = A_b if A_sum is None else A_sum + A_b
        return A_sum, e_g, k_g

    def coherent_multi_block_ambiguity(x_hats_2d):
        """Coherent phase-aligned ambiguity for local refinement (sharper κ)."""
        C_stack = None; e_g = None; k_g = None
        for b in range(B_block):
            s_b = system.idaft(x_hats_2d[:, b, :])
            C_b, e_g, k_g = ambiguity_function_complex(
                r[:, b, :], s_b, N=N, N_cp=N_cp_int,
                kappa_max=cfg.kappa_max, ell_max=float(cfg.ell_max),
                oversample_doppler=2,
            )
            align_phase = torch.exp(-1j * 2 * torch.pi * k_g * b * beta).to(C_b.dtype)
            C_b_aligned = C_b * align_phase.view(1, -1, 1)
            C_stack = C_b_aligned if C_stack is None else C_stack + C_b_aligned
        return (C_stack.abs() ** 2), e_g, k_g

    # Initial: pilot-only signals.
    x_pilot = torch.zeros(B_batch, B_block, N, dtype=dtype, device=device)
    for b in range(B_block):
        x_pilot[:, b, pp[b]] = pv[b].unsqueeze(0)

    A_sum, e_g, k_g = multi_block_ambiguity(x_pilot)
    peak_idx, _ = cfar_peaks(A_sum, K=cfg.P_max, min_separation=2)
    ell_hat, kap_hat = newton_refine(A_sum, peak_idx, e_g, k_g, max_iter=2)
    # Aperture-synthesis coherent local κ refinement (uses fine grid within a
    # window ±0.30 well below Doppler-Nyquist Δκ = 1/β ≈ 0.93; exploits the O(B³)
    # Fisher information for κ under the physical multi-block model).
    if B_block > 1 and use_aperture:
        kap_hat = aperture_synthesis_kappa_refine(
            system, r, x_pilot, ell_hat, kap_hat, N_cp=N_cp_int,
            kappa_window=kappa_window, kappa_step=kappa_step, beta=beta,
        )

    # 2. Multi-block LS h.
    h_hat = multiblock_ls_gains_data_aided(system, batch, ell_hat, kap_hat, x_pilot,
                                           lambda_ridge=lambda_ridge)

    # 3. Detect symbols per block with shared (theta, h_ref); effective per-block
    # gain h_b = h_ref * D_b(kappa) is used in the block-b equalizer.
    #
    # Reliability weighting uses an EFFECTIVE post-equalization variance rather than
    # the nominal sigma_w^2. The nominal value omits residual channel-estimation
    # error, unresolved path interference, and support mismatch, so as sigma_w^2 -> 0
    # the softmax becomes arbitrarily sharp and admits wrong hard decisions with
    # near-unity confidence (the high-SNR upturn). We augment it with the measured
    # per-block residual power, which is already computed by the outer loop:
    #     v_eff = sigma_w^2 + ||r_b - A_b(theta,x_hat) D_b h||^2 / N
    # At high SNR the residual term dominates and the weighting stays calibrated.
    N_cp_det = int(cfg.ell_max)

    def residual_power(ell_t, kap_t, h_t, x_ref):
        """Mean per-sample stacked residual power (scalar per batch element)."""
        res = torch.zeros(B_batch, device=device)
        for b in range(B_block):
            A = build_regression_matrix(system, ell_t, kap_t, x_ref[:, b, :])
            phase_b = block_doppler_phase(kap_t, b, N, N_cp_det)
            r_hat = ((A * phase_b.unsqueeze(1)) @ h_t.unsqueeze(-1)).squeeze(-1)
            res = res + (batch.r[:, b, :] - r_hat).abs().pow(2).sum(dim=-1)
        return res / (B_block * N)

    def detect_all_blocks(v_eff=None):
        # v_eff: (B_batch,) effective variance; falls back to nominal sigma_w2
        if v_eff is None:
            om = torch.full((B_batch,), 1.0 / max(sigma_w2, 1e-6), device=device)
        else:
            om = 1.0 / v_eff.clamp(min=max(sigma_w2, 1e-6))
        p_ms = torch.zeros(B_batch, B_block, N, const.numel(), dtype=torch.float32, device=device)
        hard = torch.zeros(B_batch, B_block, N, dtype=torch.long, device=device)
        # Regularize the equalizer with the SAME effective error used for the
        # confidence weighting. The nominal sigma_w^2 vanishes at high SNR while
        # the channel-mismatch error does not, so a sigma_w^2-regularized CG-MMSE
        # increasingly amplifies channel-estimation error -- this, not softmax
        # overconfidence, is the dominant cause of the high-SNR upturn.
        reg = sigma_w2 if v_eff is None else float(v_eff.mean().clamp(min=sigma_w2))
        for b in range(B_block):
            phase_b = block_doppler_phase(kap_hat, b, N, N_cp_det)     # (B_batch, P)
            h_b = h_hat * phase_b
            op = FastAFDMOperator(system=system, ell=ell_hat, kappa=kap_hat, h=h_b)
            def mv(v): return op.rmatvec(op.matvec(v)) + reg * v
            z = cg_solve(mv, op.rmatvec(y[:, b, :]), max_iter=K_cg)
            if calibrate_output:
                # The MMSE equalizer is biased: E[z] = G x with G = (H^H H + vI)^-1 H^H H,
                # so z is shrunk toward the origin.  For a constant-modulus alphabet the
                # shrinkage is common to all points and does not move the argmax, but for
                # a non-constant-modulus alphabet it systematically favours inner points.
                # Estimate the complex scale and the residual variance on the pilots,
                # where x is known exactly, then de-bias.
                zp = z[:, pp[b]]                              # (Bb, N_p)
                xp = pv[b].unsqueeze(0)                       # (1,  N_p)
                num = (zp * xp.conj()).sum(dim=-1)
                den = (xp.abs() ** 2).sum(dim=-1).clamp(min=1e-12)
                g = num / den                                 # (Bb,) complex
                g = torch.where(g.abs() < 1e-3, torch.ones_like(g), g)
                v_post = ((zp - g.unsqueeze(-1) * xp).abs() ** 2).mean(dim=-1)
                z_cal = z / g.unsqueeze(-1)
                om_b = (g.abs() ** 2 / v_post.clamp(min=1e-9)).clamp(max=1e6)
            else:
                z_cal, om_b = z, om
            dists = (z_cal.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
            p_ms[:, b, :] = F.softmax(-om_b.view(-1, 1, 1) * dists, dim=-1)
            hard[:, b, :] = p_ms[:, b, :].argmax(dim=-1)
        return p_ms, hard

    v0 = residual_power(ell_hat, kap_hat, h_hat, x_pilot) if use_veff else None
    p_ms, hard = detect_all_blocks(v0)

    # 4. Outer loop.
    for it in range(n_outer):
        # Build data-aided x_hats per block.
        x_hats = torch.zeros(B_batch, B_block, N, dtype=dtype, device=device)
        for b in range(B_block):
            if soft_symbols:
                # Posterior mean E[x_m | y] = sum_s s p_m(s).
                x_hats[:, b, :] = (p_ms[:, b, :].to(dtype)
                                   * const.reshape(1, 1, -1)).sum(dim=-1)
            else:
                rho_b = p_ms[:, b, :].max(dim=-1).values
                reliable_b = rho_b >= rho_min
                x_hats[:, b, :][reliable_b] = const[hard[:, b, :][reliable_b]]
            x_hats[:, b, pp[b]] = pv[b].unsqueeze(0)

        # Reacquire theta: noncoherent CFAR (unambiguous peaks) then aperture-
        # synthesis coherent local refinement (fine κ).
        if use_reacq:
            A_sum, e_g, k_g = multi_block_ambiguity(x_hats)
            peak_idx, _ = cfar_peaks(A_sum, K=cfg.P_max, min_separation=2)
            ell_new, kap_new = newton_refine(A_sum, peak_idx, e_g, k_g, max_iter=2)
            if B_block > 1 and use_aperture:
                kap_new = aperture_synthesis_kappa_refine(
                    system, r, x_hats, ell_new, kap_new, N_cp=N_cp_int,
                    kappa_window=kappa_window, kappa_step=kappa_step, beta=beta,
                )
            # Accept if lower residual.
            def stacked_residual(ell_t, kap_t):
                h_t = multiblock_ls_gains_data_aided(system, batch, ell_t, kap_t, x_hats,
                                                     lambda_ridge=lambda_ridge)
                res = torch.zeros(B_batch, device=device)
                for b in range(B_block):
                    A = build_regression_matrix(system, ell_t, kap_t, x_hats[:, b, :])
                    phase_b = block_doppler_phase(kap_t, b, N, N_cp_det)
                    A_b = A * phase_b.unsqueeze(1)
                    r_hat = (A_b @ h_t.unsqueeze(-1)).squeeze(-1)
                    res = res + (batch.r[:, b, :] - r_hat).abs().pow(2).sum(dim=-1)
                return res, h_t
            res_new, h_new = stacked_residual(ell_new, kap_new)
            res_old, _ = stacked_residual(ell_hat, kap_hat)
            accept = res_new < res_old
            for b_idx in range(B_batch):
                if accept[b_idx]:
                    ell_hat[b_idx] = ell_new[b_idx]
                    kap_hat[b_idx] = kap_new[b_idx]

        # LS h with current theta and x_hats.
        h_hat = multiblock_ls_gains_data_aided(system, batch, ell_hat, kap_hat, x_hats,
                                               lambda_ridge=lambda_ridge)

        # LM refinement of shared theta.
        for _ in range(n_lm_per_outer):
            ell_hat, kap_hat = multiblock_lm_theta(
                system, batch, ell_hat, kap_hat, h_hat, x_hats, sigma_w2,
                gamma_lr=gamma_lr, max_step=max_step, slack=slack, max_backtracks=4,
            )

        # Refit h.
        h_hat = multiblock_ls_gains_data_aided(system, batch, ell_hat, kap_hat, x_hats,
                                               lambda_ridge=lambda_ridge)

        # Detect all blocks, with reliability weighting driven by the measured
        # residual rather than the nominal noise variance.
        v_eff = residual_power(ell_hat, kap_hat, h_hat, x_hats) if use_veff else None
        p_ms, hard = detect_all_blocks(v_eff)

    return hard, ell_hat, kap_hat, h_hat


def eval_multiblock(cfg, snr_db, B_block, design="hopping", n_batches=6, batch_size=16,
                    seed=42, soft_symbols=False, n_outer=6):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS[design](N=cfg.N, N_p=cfg.N_p, B=B_block,
                                   constellation=const, device=cfg.device, seed=42)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_multiblock(system, channel, const, pp, pv,
                                  batch_size=batch_size, snr_db=snr_db, generator=gen)
        hard, _, _, _ = multiblock_dasbl_receiver(system, batch, const, cfg,
                                                   n_outer=n_outer, n_lm_per_outer=3,
                                                   rho_min=0.5, use_reacq=True,
                                                   soft_symbols=soft_symbols)
        mask = batch.pilot_mask
        ser = float(((hard != batch.labels) * mask).float().sum() / mask.float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="hopping",
                    choices=("repeated", "hopping", "complementary"))
    ap.add_argument("--Bs", type=str, default="1,2,4")
    ap.add_argument("--configs", type=str, default="all")
    args = ap.parse_args()

    Bs = [int(b) for b in args.Bs.split(",")]

    all_cfgs = (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
        ("HARD (P=5, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)),
    )
    cfgs = all_cfgs if args.configs == "all" else [c for c in all_cfgs if args.configs in c[0]]

    print("=" * 78)
    print(f"MULTI-BLOCK ITERATIVE DASBL (shared theta) — design={args.design}, Bs={Bs}")
    print("=" * 78)
    for name, cfg in cfgs:
        print(f"\n{name}")
        print(f"  {'SNR':<6s}  " + "  ".join(f"B={b}" for b in Bs))
        for snr in (5.0, 15.0, 25.0):
            row = []
            for B in Bs:
                ser = eval_multiblock(cfg, snr, B_block=B, design=args.design)
                row.append(ser)
            print(f"  {snr:>4.1f}dB  " + "  ".join(f"{v:.3e}" for v in row))


if __name__ == "__main__":
    main()

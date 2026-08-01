"""Data-aided SBL/EM gain refinement.

After the initial channel estimate + soft symbol detection, use only reliably-
decoded symbols as pseudo-pilots and re-estimate path gains. This closes the
pilot-power gap identified in Diagnostic A: pilot-only LS at (P=5, N_p=16)
yielded 19% SER even with true positions; augmenting with reliable data
symbols provides many more effective observations for gain estimation.

Reference: recent AFDM data-aided SBL work (Xu 2026 arXiv:2607.18881) uses
reliably decoded symbols as pseudo-pilots and jointly evolves the off-grid
support.

We keep this deterministic/model-based — no learned parameters. The neural
estimator's job is to provide accurate positions and initial existence
selection; the model-based SBL closes the remaining gain-estimation gap.
"""

from __future__ import annotations

import torch

from .classical import build_regression_matrix, cg_solve
from .operators import FastAFDMOperator
from .system import AFDMSystem


def _select_active_paths(pred: dict, min_exist_prob: float = 0.5) -> torch.Tensor:
    """Return (B, K) bool mask of paths passing the existence threshold."""
    p = torch.sigmoid(pred["exist_logit"])
    return (p >= min_exist_prob) & pred["valid"]


def _reliability_from_p_ms(p_ms: torch.Tensor) -> torch.Tensor:
    """rho_m = max_s p_m(s), shape (B, N)."""
    return p_ms.max(dim=-1).values


def data_aided_sbl_step(
    system: AFDMSystem,
    r: torch.Tensor,               # (B, N)
    y: torch.Tensor,               # (B, N) DAFT-domain
    ell: torch.Tensor,             # (B, P_active)
    kappa: torch.Tensor,           # (B, P_active)
    h_init: torch.Tensor,          # (B, P_active) complex
    x_hat: torch.Tensor,           # (B, N) current soft/hard symbol estimate
    p_ms: torch.Tensor,            # (B, N, |S|) current soft symbol posterior
    sigma_w2: float,
    pilot_positions: torch.Tensor,
    pilot_values: torch.Tensor,
    rho_min: float = 0.9,
    lambda_ridge: float = 1e-3,
    weight_scheme: str = "soft",   # "hard" or "soft"
) -> tuple[torch.Tensor, torch.Tensor]:
    """One data-aided SBL/EM iteration.

    weight_scheme:
      "hard": only symbols with rho_m >= rho_min contribute (binary weight).
      "soft": each row weighted by rho_m directly (continuous 0..1).

    Returns (h_new, weights) — h_new is the refined gain estimate.
    """
    B, N = r.shape
    device = r.device; dtype = r.dtype

    # 1. Build effective pseudo-pilot symbol tensor: x_pp = x_hat with pilots enforced.
    x_pp = x_hat.clone()
    x_pp[:, pilot_positions] = pilot_values.unsqueeze(0)

    # 2. Compute reliability weights per DAFT-domain bin.
    rho = _reliability_from_p_ms(p_ms)                     # (B, N)
    if weight_scheme == "hard":
        w = (rho >= rho_min).to(torch.float32)
    else:
        w = torch.clamp((rho - rho_min) / (1.0 - rho_min), min=0.0)
    # Pilots always contribute with weight 1.
    w = w.clone()
    w[:, pilot_positions] = 1.0

    # 3. Weighted LS: min |D_w (r - A(theta) h)|^2 where D_w is diag(sqrt(w))
    #    over the TIME-DOMAIN samples. To convert w from DAFT-bin weights to
    #    time-domain sample weights, we need the row weights consistent with
    #    the regression matrix, whose rows are indexed by time-domain n.
    #    Since x_pp gates data via its DAFT bins, the effective observation at
    #    time n contains contributions from ALL bins weighted by DAFT basis.
    #    A simple heuristic: use uniform time-domain weights but modulate x_pp
    #    by w in the DAFT domain BEFORE building the regression. That way, low-
    #    reliability bins contribute proportionally less to A.
    #
    #    Concretely: x_pp_weighted = x_pp * w  (as DAFT-domain vector).
    x_pp_w = x_pp * w.to(dtype)
    A = build_regression_matrix(system, ell, kappa, x_pp_w)   # (B, N, P)
    AH = A.conj().transpose(-1, -2)
    AhA = AH @ A
    Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)
    P_active = ell.shape[1]
    ridge = lambda_ridge * torch.eye(P_active, dtype=dtype, device=device).unsqueeze(0)
    h_new = torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)
    return h_new, w


def run_data_aided_sbl(
    system: AFDMSystem,
    r: torch.Tensor, y: torch.Tensor,
    ell: torch.Tensor, kappa: torch.Tensor, h_init: torch.Tensor,
    constellation: torch.Tensor,
    pilot_positions: torch.Tensor, pilot_values: torch.Tensor,
    sigma_w2: float,
    n_iters: int = 3,
    rho_min: float = 0.9,
    K_cg: int = 30,
    weight_scheme: str = "soft",
) -> dict:
    """Full data-aided SBL loop: iterate (detect symbols -> refine h)."""
    import torch.nn.functional as F
    B, N = r.shape
    h_cur = h_init
    p_ms = None; x_hat = None
    for it in range(n_iters):
        # 1. Detect symbols with current h.
        op = FastAFDMOperator(system=system, ell=ell, kappa=kappa, h=h_cur)
        def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
        z = cg_solve(mv, op.rmatvec(y), max_iter=K_cg)
        dists = (z.unsqueeze(-1) - constellation.reshape(1, 1, -1)).abs() ** 2
        omega = 1.0 / max(sigma_w2, 1e-6)
        logits = -omega * dists
        p_ms = F.softmax(logits, dim=-1)
        hard_idx = p_ms.argmax(dim=-1)
        x_hat = constellation[hard_idx]
        x_hat[:, pilot_positions] = pilot_values.unsqueeze(0)

        # 2. Refine h with data-aided SBL step.
        h_new, w = data_aided_sbl_step(
            system, r, y, ell, kappa, h_cur, x_hat, p_ms, sigma_w2,
            pilot_positions, pilot_values,
            rho_min=rho_min, weight_scheme=weight_scheme,
        )
        h_cur = h_new

    # Final detection with refined h.
    op = FastAFDMOperator(system=system, ell=ell, kappa=kappa, h=h_cur)
    def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
    z = cg_solve(mv, op.rmatvec(y), max_iter=K_cg)
    dists = (z.unsqueeze(-1) - constellation.reshape(1, 1, -1)).abs() ** 2
    omega = 1.0 / max(sigma_w2, 1e-6)
    p_ms = F.softmax(-omega * dists, dim=-1)
    hard_idx = p_ms.argmax(dim=-1)
    return {"h": h_cur, "x_soft": z, "hard_x": hard_idx, "p_ms": p_ms,
            "ell": ell, "kappa": kappa}

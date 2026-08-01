"""Assignment-aware losses for the PathSetEstimator.

Composite training objective:
    L = L_set + lambda_rec * L_rec

L_set : Hungarian match between K predicted paths and P_true true paths, then
        per-match losses on existence, offsets, and complex gain, plus false-
        alarm (predicted-exists-but-unmatched) and missed-detection (true-but-
        unmatched-in-top-K) penalties.

L_rec : Pilot reconstruction loss, |r_P - A(theta_hat) h_hat|^2 / |r_P|^2.
        Enforces physical consistency independently of set matching.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .classical import build_regression_matrix
from .system import AFDMSystem


def _pairwise_match_cost(
    ell_hat: torch.Tensor,   # (K,)
    kap_hat: torch.Tensor,
    h_hat: torch.Tensor,     # complex (K,)
    exist_hat: torch.Tensor, # (K,) probabilities (sigmoid applied outside)
    ell_true: torch.Tensor,  # (P,)
    kap_true: torch.Tensor,
    h_true: torch.Tensor,    # complex (P,)
    w_ell: float, w_kap: float, w_h: float, w_e: float,
) -> torch.Tensor:
    """Per-instance (K, P) cost matrix for Hungarian matching (one batch element)."""
    K = ell_hat.shape[0]; P = ell_true.shape[0]
    d_ell = ell_hat.unsqueeze(1) - ell_true.unsqueeze(0)         # (K, P)
    d_kap = kap_hat.unsqueeze(1) - kap_true.unsqueeze(0)
    d_h = h_hat.unsqueeze(1) - h_true.unsqueeze(0)               # complex (K, P)
    # Cost per potential match:
    #   position error + gain error + BCE(existence=1) at predicted probability.
    cost = (w_ell * d_ell.pow(2) + w_kap * d_kap.pow(2)
            + w_h * d_h.abs().pow(2)
            - w_e * torch.log(exist_hat.unsqueeze(1).clamp(min=1e-8)))
    return cost


def hungarian_pathset_loss(
    pred: dict,               # PathSetEstimator output, all shape (B, K)
    ell_cfar: torch.Tensor,   # (B, K) initial CFAR positions (delta added to these)
    kap_cfar: torch.Tensor,
    theta_true: torch.Tensor, # (B, P, 2)
    h_true: torch.Tensor,     # (B, P) complex
    w_ell: float = 1.0,
    w_kap: float = 1.0,
    w_h: float = 1.0,
    w_e: float = 0.5,
    mu_fa: float = 0.2,
    mu_md: float = 1.0,
    pos_weight: float = 7.0,  # roughly K/P ratio: counteract negative-class dominance
) -> tuple[torch.Tensor, dict]:
    """Hungarian set loss with false-alarm and missed-detection penalties.

    Returns (loss, breakdown). Breakdown contains per-term scalars for logging.
    """
    B, K = pred["exist_logit"].shape
    device = pred["exist_logit"].device
    dtype_r = pred["exist_logit"].dtype

    ell_hat = ell_cfar + pred["delta_ell"]     # (B, K)
    kap_hat = kap_cfar + pred["delta_kappa"]
    exist_prob = torch.sigmoid(pred["exist_logit"])

    ell_t = theta_true[..., 0]; kap_t = theta_true[..., 1]

    l_ell = torch.zeros(B, device=device); l_kap = torch.zeros(B, device=device)
    l_h = torch.zeros(B, device=device); l_e = torch.zeros(B, device=device)
    l_fa = torch.zeros(B, device=device); l_md = torch.zeros(B, device=device)
    l_nll = torch.zeros(B, device=device)   # Gaussian NLL on complex gain
    n_matched_total = 0

    valid_mask = pred["valid"]  # (B, K)

    for b in range(B):
        vmask = valid_mask[b]
        K_valid = int(vmask.sum().item())
        if K_valid == 0:
            l_md[b] = mu_md * ell_t.shape[1]
            continue
        # Compute cost only over valid predictions.
        cost = _pairwise_match_cost(
            ell_hat[b, vmask], kap_hat[b, vmask],
            pred["h"][b, vmask], exist_prob[b, vmask],
            ell_t[b], kap_t[b], h_true[b],
            w_ell, w_kap, w_h, w_e,
        )  # (K_valid, P)
        with torch.no_grad():
            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

        # Map row indices in K_valid back to full-K indices.
        valid_idx_full = torch.nonzero(vmask, as_tuple=True)[0]  # (K_valid,)
        matched_pred_idx = valid_idx_full[torch.tensor(row_ind, device=device)]
        matched_true_idx = torch.tensor(col_ind, device=device, dtype=torch.long)

        # Matched pairs contribute to l_ell, l_kap, l_h, and existence BCE=1.
        n = matched_pred_idx.shape[0]
        n_matched_total += n
        de = ell_hat[b, matched_pred_idx] - ell_t[b, matched_true_idx]
        dk = kap_hat[b, matched_pred_idx] - kap_t[b, matched_true_idx]
        dh = pred["h"][b, matched_pred_idx] - h_true[b, matched_true_idx]
        l_ell[b] = w_ell * de.pow(2).sum()
        l_kap[b] = w_kap * dk.pow(2).sum()
        l_h[b] = w_h * dh.abs().pow(2).sum()
        # Gaussian NLL on gain: teaches the log-variance head to calibrate.
        # For complex Gaussian with variance v (over each real component):
        #   -log p(h_true | h_hat, v) = log(2 pi v) + |h_true - h_hat|^2 / v
        log_v = pred["log_var"][b, matched_pred_idx]        # (n,)
        v = log_v.exp().clamp(min=1e-9)
        l_nll[b] = (log_v + dh.abs().pow(2) / v).sum()
        # Existence BCE targets: 1 for matched, 0 for unmatched (valid) preds.
        target = torch.zeros(K, device=device, dtype=dtype_r)
        target[matched_pred_idx] = 1.0
        target = target * vmask.to(dtype_r)  # unmatched invalid stays 0
        pw = torch.tensor(pos_weight, device=device, dtype=dtype_r)
        bce = F.binary_cross_entropy_with_logits(
            pred["exist_logit"][b] * vmask.to(dtype_r),
            target,
            reduction="none",
            pos_weight=pw,
        ) * vmask.to(dtype_r)
        l_e[b] = w_e * bce.sum()

        # Missed detections: true paths with no matched predicted path.
        P_true = ell_t.shape[1]
        matched_true_set = set(matched_true_idx.tolist())
        n_md = P_true - len(matched_true_set)
        l_md[b] = mu_md * n_md

        # False alarms: preds with exist_prob > 0.5 that are NOT matched.
        matched_pred_set = set(matched_pred_idx.tolist())
        unmatched_pred_mask = torch.tensor(
            [i not in matched_pred_set for i in range(K)],
            device=device,
        ) & vmask
        l_fa[b] = mu_fa * (exist_prob[b] * unmatched_pred_mask.to(dtype_r)).sum()

    w_nll_scale = 0.1   # keep NLL modest relative to hard errors
    total = (l_ell + l_kap + l_h + l_e + l_fa + l_md + w_nll_scale * l_nll).mean()
    return total, {
        "ell": float(l_ell.mean().detach()),
        "kap": float(l_kap.mean().detach()),
        "h": float(l_h.mean().detach()),
        "exist": float(l_e.mean().detach()),
        "fa": float(l_fa.mean().detach()),
        "md": float(l_md.mean().detach()),
        "nll": float(l_nll.mean().detach()),
        "matched_per_batch": n_matched_total / max(B, 1),
    }


def reconstruction_loss(
    system: AFDMSystem,
    r: torch.Tensor,                 # (B, N) received time-domain
    ell_hat: torch.Tensor,           # (B, K)
    kap_hat: torch.Tensor,
    h_hat: torch.Tensor,             # (B, K) complex
    valid: torch.Tensor,             # (B, K) bool
    x_ref: torch.Tensor,             # (B, N) reference DAFT-domain symbols
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reconstruction loss: |r - A(theta_hat, x_ref) h_hat|^2 / |r|^2.

    During training, x_ref = x_true (we know the ground truth), so this
    measures how well the (theta_hat, h_hat) channel model explains r under the
    known-symbol assumption. During inference, x_ref would be pilot-only or
    the current soft-symbol estimate.

    Zero-gain invalid paths contribute nothing.
    """
    B, N = r.shape
    h_hat_masked = h_hat * valid.to(h_hat.dtype)
    A = build_regression_matrix(system, ell_hat, kap_hat, x_ref)       # (B, N, K)
    r_hat = (A @ h_hat_masked.unsqueeze(-1)).squeeze(-1)                # (B, N)
    num = (r - r_hat).abs().pow(2).sum(dim=-1)
    den = r.abs().pow(2).sum(dim=-1).clamp(min=eps)
    return (num / den).mean()


def compose_pathset_loss(
    pred: dict,
    ell_cfar: torch.Tensor,
    kap_cfar: torch.Tensor,
    system: AFDMSystem,
    r: torch.Tensor,
    theta_true: torch.Tensor,
    h_true: torch.Tensor,
    x_true: torch.Tensor,   # (B, N) known DAFT-domain symbols (train-time supervision)
    lambda_rec: float = 0.5,
    hungarian_kwargs: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Total loss = Hungarian set loss + lambda_rec * reconstruction loss."""
    hk = hungarian_kwargs or {}
    l_set, breakdown = hungarian_pathset_loss(
        pred, ell_cfar, kap_cfar, theta_true, h_true, **hk,
    )
    ell_hat = ell_cfar + pred["delta_ell"]
    kap_hat = kap_cfar + pred["delta_kappa"]
    l_rec = reconstruction_loss(system, r, ell_hat, kap_hat, pred["h"],
                                pred["valid"], x_true)
    total = l_set + lambda_rec * l_rec
    breakdown["set"] = float(l_set.detach())
    breakdown["rec"] = float(l_rec.detach())
    breakdown["total"] = float(total.detach())
    return total, breakdown

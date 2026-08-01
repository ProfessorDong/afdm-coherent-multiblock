"""End-to-end PathSet receiver.

Pipeline:
    r
    | build_frontend_inputs  ->  scalar_feats, patch_feats, valid, ell_cfar, kap_cfar
    | PathSetEstimator       ->  {exist_logit, delta_ell, delta_kappa, h, log_var}
    | existence threshold    ->  active set (variable cardinality per batch element)
    | (optional) safeguarded_lm_theta_step  x 1-2         [Stage 3: LM polish]
    | run_data_aided_sbl x n_iters                       [Stage 4: DA-SBL]
    | final CG-MMSE                                       [Stage 5: fixed detector]
    | hard decisions
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classical import cg_solve
from .data_aided_sbl import run_data_aided_sbl
from .operators import FastAFDMOperator
from .pathset_estimator import PathSetEstimator, PathSetEstimatorConfig
from .pathset_frontend import build_frontend_inputs
from .support import SupportRecovery
from .system import AFDMSystem
from .vem import safeguarded_lm_theta_step


@dataclass
class PathSetReceiverConfig:
    K: int = 24
    est_cfg: PathSetEstimatorConfig = None
    kappa_max: float = 5.0
    min_exist_prob: float = 0.5
    max_active_paths: int = 8         # cap on active paths, top by existence probability
    lm_polish_iters: int = 5          # 0 to disable Stage 3 (5 gives ~0.8pp over 1 iter)
    sbl_iters: int = 3                # 0 to disable Stage 4
    sbl_rho_min: float = 0.9
    sbl_weight_scheme: str = "soft"
    K_cg_final: int = 30

    def __post_init__(self):
        if self.est_cfg is None:
            self.est_cfg = PathSetEstimatorConfig(K=self.K)


class PathSetReceiver(nn.Module):
    """End-to-end receiver combining the learned estimator with model-based polish."""

    def __init__(self, cfg: PathSetReceiverConfig,
                 system: AFDMSystem,
                 constellation: torch.Tensor,
                 pilot_positions: torch.Tensor, pilot_values: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.system = system
        self.register_buffer("constellation", constellation)
        self.register_buffer("pilot_positions", pilot_positions)
        self.register_buffer("pilot_values", pilot_values)
        self.estimator = PathSetEstimator(cfg.est_cfg)

    def forward(self, r: torch.Tensor, sigma_w2_block: float,
                return_intermediates: bool = False) -> dict:
        """Run the full receiver on a batch of received signals."""
        # Stage 1-2: front-end + amortized path-set estimator.
        fe = build_frontend_inputs(
            r, self.system, self.pilot_positions, self.pilot_values,
            kappa_max=self.cfg.kappa_max, K=self.cfg.K,
            sigma_w2_block=sigma_w2_block,
        )
        pred = self.estimator(fe["scalar_feats"], fe["patch_feats"], fe["valid"])

        # Absolute refined positions.
        ell_pred = fe["ell_cfar"] + pred["delta_ell"]
        kap_pred = fe["kap_cfar"] + pred["delta_kappa"]

        # Existence-based path selection with top-N_max cap.
        exist_p = torch.sigmoid(pred["exist_logit"]) * pred["valid"].to(pred["exist_logit"].dtype)
        exist_p = torch.where(exist_p >= self.cfg.min_exist_prob, exist_p,
                              torch.zeros_like(exist_p))
        top_vals, top_idx = exist_p.topk(k=self.cfg.max_active_paths, dim=-1)
        active_mask = top_vals > 0  # (B, N_max)

        ell_a = torch.gather(ell_pred, 1, top_idx)
        kap_a = torch.gather(kap_pred, 1, top_idx)
        h_a = torch.gather(pred["h"], 1, top_idx)
        # Clamp positions to valid range.
        ell_a = ell_a.clamp(min=0.0, max=float(self.system.ell_max))
        kap_a = kap_a.clamp(min=-self.cfg.kappa_max, max=self.cfg.kappa_max)
        # Zero out inactive slots.
        h_a = h_a * active_mask.to(h_a.dtype)

        # DAFT-domain observation.
        y = self.system.daft(r)

        # Stage 3: safeguarded LM polish (deterministic).
        if self.cfg.lm_polish_iters > 0:
            # For LM we need an x_hat estimate. Use a first-pass hard decode.
            op = FastAFDMOperator(system=self.system, ell=ell_a, kappa=kap_a, h=h_a)
            def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2_block * v
            z0 = cg_solve(mv, op.rmatvec(y), max_iter=self.cfg.K_cg_final)
            hard_idx0 = (z0.unsqueeze(-1) - self.constellation.reshape(1, 1, -1)).abs().argmin(dim=-1)
            x_hat0 = self.constellation[hard_idx0]
            x_hat0[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)
            for _ in range(self.cfg.lm_polish_iters):
                ell_a, kap_a, _acc = safeguarded_lm_theta_step(
                    self.system, r, h_a, x_hat0, ell_a, kap_a,
                    sigma_w2=sigma_w2_block, v_h=None,
                    gamma_lr=0.5, max_step=0.2, slack=1e-4, max_backtracks=4,
                )

        # Stage 4: data-aided SBL gain refinement.
        if self.cfg.sbl_iters > 0:
            sbl_out = run_data_aided_sbl(
                self.system, r, y, ell_a, kap_a, h_a,
                self.constellation, self.pilot_positions, self.pilot_values,
                sigma_w2=sigma_w2_block,
                n_iters=self.cfg.sbl_iters,
                rho_min=self.cfg.sbl_rho_min,
                K_cg=self.cfg.K_cg_final,
                weight_scheme=self.cfg.sbl_weight_scheme,
            )
            h_a = sbl_out["h"]
            hard_idx = sbl_out["hard_x"]
            p_ms = sbl_out["p_ms"]
            z_final = sbl_out["x_soft"]
        else:
            # Stage 5: final CG-MMSE detection.
            op = FastAFDMOperator(system=self.system, ell=ell_a, kappa=kap_a, h=h_a)
            def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2_block * v
            z_final = cg_solve(mv, op.rmatvec(y), max_iter=self.cfg.K_cg_final)
            omega = 1.0 / max(sigma_w2_block if isinstance(sigma_w2_block, float)
                              else float(sigma_w2_block), 1e-6)
            dists = (z_final.unsqueeze(-1) - self.constellation.reshape(1, 1, -1)).abs() ** 2
            p_ms = F.softmax(-omega * dists, dim=-1)
            hard_idx = p_ms.argmax(dim=-1)

        out = {
            "hard_x": hard_idx, "p_ms": p_ms, "x_soft": z_final,
            "h_hat": h_a, "ell_hat": ell_a, "kappa_hat": kap_a,
            "active_mask": active_mask,
        }
        if return_intermediates:
            out.update({"pred": pred, "frontend": fe})
        return out

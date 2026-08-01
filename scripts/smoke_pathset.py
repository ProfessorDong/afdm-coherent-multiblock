"""Smoke test for the PathSetEstimator pipeline.

Verifies:
  * Front-end produces correct shapes on both easy and hard configs.
  * Estimator forward runs and outputs valid shapes.
  * Composite loss is finite and backward gives non-zero gradients on every
    trainable parameter.
  * Reconstruction loss with true (theta, h) is close to zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import ExperimentConfig
from afdm.pathset_estimator import (
    PathSetEstimator, PathSetEstimatorConfig, PATCH_FLAT_DIM,
)
from afdm.pathset_frontend import build_frontend_inputs
from afdm.pathset_loss import compose_pathset_loss, reconstruction_loss
from afdm.training import sample_batch


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK   {msg}")


def smoke_config(cfg: ExperimentConfig, snr_db: float = 15.0, K: int = 24):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(0)
    batch = sample_batch(system, channel, const, pp, pv,
                         batch_size=8, snr_db=snr_db, generator=gen)

    print(f"\n[cfg P={cfg.P}, N_p={cfg.N_p}, K={K}, snr={snr_db}]")

    fe = build_frontend_inputs(
        batch["r"], system, pp, pv, kappa_max=cfg.kappa_max,
        K=K, sigma_w2_block=batch["sigma_w2_block"],
    )
    _assert(fe["scalar_feats"].shape == (8, K, 5), "scalar_feats shape")
    _assert(fe["patch_feats"].shape == (8, K, PATCH_FLAT_DIM),
            f"patch_feats shape (expect (8, {K}, {PATCH_FLAT_DIM}))")
    _assert(fe["valid"].dtype == torch.bool, "valid is bool")
    n_valid = int(fe["valid"].sum())
    _assert(n_valid > 0, f"at least some valid peaks (got {n_valid})")

    est_cfg = PathSetEstimatorConfig(K=K)
    est = PathSetEstimator(est_cfg).to(cfg.device)
    pred = est(fe["scalar_feats"], fe["patch_feats"], fe["valid"])
    _assert(pred["exist_logit"].shape == (8, K), "exist_logit shape")
    _assert(pred["h"].dtype == torch.complex64, "h is complex64")

    loss, breakdown = compose_pathset_loss(
        pred, fe["ell_cfar"], fe["kap_cfar"], system, batch["r"],
        batch["theta_true"], batch["h_true"], batch["x_true"],
        lambda_rec=0.5,
        hungarian_kwargs=dict(w_ell=1.0, w_kap=1.0, w_h=1.0, w_e=0.5,
                              mu_fa=0.2, mu_md=1.0),
    )
    _assert(torch.isfinite(loss), f"total loss finite (got {loss.item():.3e})")
    print(f"  breakdown: {breakdown}")

    loss.backward()
    n_zero_grad = 0; n_total = 0
    for name, p in est.named_parameters():
        n_total += 1
        if p.grad is None or p.grad.abs().max() < 1e-12:
            print(f"  ZERO grad on: {name}")
            n_zero_grad += 1
    _assert(n_zero_grad == 0, f"all {n_total} parameters have nonzero gradient")

    # Reconstruction loss with true (theta, h) should be small.
    B, P_true = batch["h_true"].shape
    ell_true = batch["theta_true"][..., 0]
    kap_true = batch["theta_true"][..., 1]
    valid_true = torch.ones(B, P_true, dtype=torch.bool, device=cfg.device)
    rec_oracle = reconstruction_loss(
        system, batch["r"], ell_true, kap_true, batch["h_true"],
        valid_true, batch["x_true"],
    )
    print(f"  oracle rec loss (x_true): {rec_oracle.item():.3e}   "
          f"random-net rec loss: {breakdown['rec']:.3e}")


def main():
    print("=" * 78)
    print("PathSetEstimator smoke test")
    print("=" * 78)
    for cfg in (
        ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6),
        ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8),
    ):
        smoke_config(cfg, snr_db=15.0)
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()

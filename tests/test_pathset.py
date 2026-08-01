"""Tests for PathSetEstimator, front-end feature builder, Hungarian loss."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from afdm.experiments import ExperimentConfig
from afdm.pathset_estimator import (
    PathSetEstimator, PathSetEstimatorConfig, extract_ambiguity_patches,
    PATCH_HALF, PATCH_FLAT_DIM,
)
from afdm.pathset_frontend import build_frontend_inputs, complex_ambiguity
from afdm.pathset_loss import (
    hungarian_pathset_loss, reconstruction_loss, compose_pathset_loss,
)
from afdm.training import sample_batch


@pytest.fixture(scope="module")
def cfg():
    return ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)


@pytest.fixture(scope="module")
def batch(cfg):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(0)
    return sample_batch(system, channel, const, pp, pv,
                        batch_size=4, snr_db=15.0, generator=gen), pp, pv


class TestPatchExtractor:
    def test_shape(self):
        B, L_k, L_e, K = 3, 23, 11, 8
        A = torch.randn(B, L_k, L_e, dtype=torch.complex64, device="cuda:0")
        peak_idx = torch.zeros(B, K, 2, dtype=torch.long, device="cuda:0")
        peak_idx[0, 0] = torch.tensor([5, 3])
        peak_idx[1, 0] = torch.tensor([-1, -1])  # invalid
        patches, valid = extract_ambiguity_patches(A, peak_idx)
        assert patches.shape == (B, K, PATCH_FLAT_DIM)
        assert valid[0, 0].item() is True
        assert valid[1, 0].item() is False
        # Invalid should be zeroed.
        assert torch.allclose(patches[1, 0], torch.zeros_like(patches[1, 0]))

    def test_circular_boundary_ell(self):
        """Delay is circular: patches at l_idx = L_e - 1 wrap around."""
        B, L_k, L_e, K = 1, 5, 11, 1
        A = torch.arange(B * L_k * L_e, dtype=torch.float32, device="cuda:0").reshape(B, L_k, L_e).to(torch.complex64)
        peak_idx = torch.tensor([[[2, L_e - 1]]], device="cuda:0")  # boundary in delay
        patches, valid = extract_ambiguity_patches(A, peak_idx, half=1)
        # 3x3x3 = 27 features. Should not crash and should contain wrapped index 0.
        assert patches.shape == (1, 1, 27)
        assert valid[0, 0].item() is True


class TestFrontend:
    def test_output_shapes(self, cfg, batch):
        b, pp, pv = batch
        system = cfg.system()
        fe = build_frontend_inputs(b["r"], system, pp, pv, kappa_max=cfg.kappa_max,
                                   K=24, sigma_w2_block=b["sigma_w2_block"])
        assert fe["scalar_feats"].shape == (4, 24, 5)
        assert fe["patch_feats"].shape == (4, 24, PATCH_FLAT_DIM)
        assert fe["valid"].shape == (4, 24)
        assert fe["valid"].dtype == torch.bool
        assert fe["A_complex"].dtype == torch.complex64

    def test_at_least_p_peaks_recovered(self, cfg, batch):
        """Front-end should return at least P valid candidates in every batch element."""
        b, pp, pv = batch
        system = cfg.system()
        fe = build_frontend_inputs(b["r"], system, pp, pv, kappa_max=cfg.kappa_max,
                                   K=24, sigma_w2_block=b["sigma_w2_block"])
        n_valid_per_b = fe["valid"].sum(dim=-1)
        assert (n_valid_per_b >= cfg.P).all(), \
            f"Some batch element has < P valid peaks: {n_valid_per_b.tolist()}"


class TestEstimator:
    def test_forward_shapes(self, cfg, batch):
        b, pp, pv = batch
        fe = build_frontend_inputs(b["r"], cfg.system(), pp, pv,
                                   kappa_max=cfg.kappa_max, K=24,
                                   sigma_w2_block=b["sigma_w2_block"])
        est = PathSetEstimator(PathSetEstimatorConfig(K=24)).to(cfg.device)
        pred = est(fe["scalar_feats"], fe["patch_feats"], fe["valid"])
        for k in ("exist_logit", "delta_ell", "delta_kappa", "log_var"):
            assert pred[k].shape == (4, 24)
        assert pred["h"].dtype == torch.complex64
        assert pred["h"].shape == (4, 24)

    def test_offset_clipping(self, cfg, batch):
        """delta_ell and delta_kappa should be clipped to [-1.5, 1.5]."""
        b, pp, pv = batch
        fe = build_frontend_inputs(b["r"], cfg.system(), pp, pv,
                                   kappa_max=cfg.kappa_max, K=24,
                                   sigma_w2_block=b["sigma_w2_block"])
        est = PathSetEstimator(PathSetEstimatorConfig(K=24, max_delta_ell=1.5,
                                                     max_delta_kap=1.5)).to(cfg.device)
        pred = est(fe["scalar_feats"], fe["patch_feats"], fe["valid"])
        assert pred["delta_ell"].abs().max() <= 1.5 + 1e-5
        assert pred["delta_kappa"].abs().max() <= 1.5 + 1e-5

    def test_permutation_equivariance(self, cfg, batch):
        """Permuting input candidates should permute outputs the same way."""
        b, pp, pv = batch
        fe = build_frontend_inputs(b["r"], cfg.system(), pp, pv,
                                   kappa_max=cfg.kappa_max, K=24,
                                   sigma_w2_block=b["sigma_w2_block"])
        est = PathSetEstimator(PathSetEstimatorConfig(K=24)).to(cfg.device)
        est.eval()
        # Use a single batch element and permute its 24 tokens.
        perm = torch.randperm(24, device=cfg.device)
        sf1 = fe["scalar_feats"][:1]; pf1 = fe["patch_feats"][:1]; v1 = fe["valid"][:1]
        sf2 = sf1[:, perm]; pf2 = pf1[:, perm]; v2 = v1[:, perm]
        with torch.no_grad():
            pred1 = est(sf1, pf1, v1)
            pred2 = est(sf2, pf2, v2)
        # After un-permuting pred2, it should equal pred1 (up to fp noise).
        inv_perm = torch.argsort(perm)
        assert torch.allclose(pred1["delta_ell"], pred2["delta_ell"][:, inv_perm], atol=1e-4)
        assert torch.allclose(pred1["exist_logit"], pred2["exist_logit"][:, inv_perm], atol=1e-4)


class TestLoss:
    def test_oracle_reconstruction_near_zero(self, cfg, batch):
        """With true (theta, h, x_true), reconstruction loss should be ~ sigma_w^2 (small)."""
        b, pp, pv = batch
        system = cfg.system()
        ell = b["theta_true"][..., 0]; kap = b["theta_true"][..., 1]
        valid = torch.ones_like(ell, dtype=torch.bool)
        rec = reconstruction_loss(system, b["r"], ell, kap, b["h_true"], valid, b["x_true"])
        # At 15 dB the residual should be about 3-5% of the signal power.
        assert rec.item() < 0.1, f"oracle rec loss too large: {rec.item():.3f}"

    def test_hungarian_matches_all_true_paths(self, cfg, batch):
        """When K >= P and all preds valid, Hungarian should match all P true paths."""
        b, pp, pv = batch
        fe = build_frontend_inputs(b["r"], cfg.system(), pp, pv,
                                   kappa_max=cfg.kappa_max, K=24,
                                   sigma_w2_block=b["sigma_w2_block"])
        est = PathSetEstimator(PathSetEstimatorConfig(K=24)).to(cfg.device)
        pred = est(fe["scalar_feats"], fe["patch_feats"], fe["valid"])
        _, breakdown = hungarian_pathset_loss(
            pred, fe["ell_cfar"], fe["kap_cfar"],
            b["theta_true"], b["h_true"],
        )
        # matched_per_batch should equal P_true (=3) since all preds valid.
        assert breakdown["matched_per_batch"] == 3.0

    def test_composite_loss_backward(self, cfg, batch):
        """All parameters should receive nonzero gradient."""
        b, pp, pv = batch
        fe = build_frontend_inputs(b["r"], cfg.system(), pp, pv,
                                   kappa_max=cfg.kappa_max, K=24,
                                   sigma_w2_block=b["sigma_w2_block"])
        est = PathSetEstimator(PathSetEstimatorConfig(K=24)).to(cfg.device)
        pred = est(fe["scalar_feats"], fe["patch_feats"], fe["valid"])
        loss, _ = compose_pathset_loss(
            pred, fe["ell_cfar"], fe["kap_cfar"], cfg.system(), b["r"],
            b["theta_true"], b["h_true"], b["x_true"],
        )
        loss.backward()
        for name, p in est.named_parameters():
            assert p.grad is not None and p.grad.abs().max() > 0, f"{name} has zero grad"

"""Tests for Hungarian set loss and training composition."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm.loss import hungarian_set_loss, hungarian_match_batch, compose_training_loss


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_hungarian_matching_finds_optimal():
    """For a small hand-crafted cost matrix, matching should be exact."""
    cost = torch.tensor([
        [[1.0, 5.0, 3.0],
         [4.0, 2.0, 6.0],
         [7.0, 8.0, 0.5]],
    ], device=DEVICE)
    row_ind, col_ind = hungarian_match_batch(cost)
    # Optimal is (0,0)+(1,1)+(2,2) with total cost 1+2+0.5 = 3.5
    r, c = row_ind[0].tolist(), col_ind[0].tolist()
    matched = [(r[i], c[i]) for i in range(3)]
    assert set(matched) == {(0, 0), (1, 1), (2, 2)}


def test_hungarian_loss_zero_at_truth():
    """With h_hat = h_true, theta_hat = theta_true (perfect match), loss = 0."""
    torch.manual_seed(0)
    B, P = 4, 3
    h_true = torch.randn(B, P, dtype=torch.complex64, device=DEVICE)
    theta_true = torch.randn(B, P, 2, device=DEVICE)
    # Permute predicted paths — Hungarian should still match perfectly.
    perm = torch.tensor([1, 2, 0], device=DEVICE)
    h_hat = h_true[:, perm]
    theta_hat = theta_true[:, perm]
    loss = hungarian_set_loss(h_hat, theta_hat, h_true, theta_true,
                              w_h=1.0, w_ell=1.0, w_kap=1.0, mu_fa=0.0, mu_md=0.0)
    assert loss.item() < 1e-4, f"loss should be ~0 at perfect match: {loss.item()}"


def test_hungarian_loss_increases_with_perturbation():
    """Perturbing h_hat should increase the loss."""
    torch.manual_seed(1)
    B, P = 4, 3
    h_true = torch.randn(B, P, dtype=torch.complex64, device=DEVICE)
    theta_true = torch.randn(B, P, 2, device=DEVICE)
    loss_0 = hungarian_set_loss(h_true, theta_true, h_true, theta_true).item()
    h_perturbed = h_true + 0.5 * torch.randn_like(h_true)
    loss_p = hungarian_set_loss(h_perturbed, theta_true, h_true, theta_true).item()
    assert loss_p > loss_0


def test_hungarian_loss_gradient_flows():
    """Gradient w.r.t. h_hat and theta_hat should be non-zero."""
    torch.manual_seed(2)
    B, P = 2, 3
    h_true = torch.randn(B, P, dtype=torch.complex64, device=DEVICE)
    theta_true = torch.randn(B, P, 2, device=DEVICE)
    h_hat = h_true + 0.5 * torch.randn_like(h_true)
    h_hat = h_hat.requires_grad_(True)
    theta_hat = (theta_true + 0.5 * torch.randn_like(theta_true)).requires_grad_(True)
    loss = hungarian_set_loss(h_hat, theta_hat, h_true, theta_true)
    (g_h, g_th) = torch.autograd.grad(loss, [h_hat, theta_hat])
    assert g_h.abs().max().item() > 1e-6
    assert g_th.abs().max().item() > 1e-6


def test_fa_md_penalty_scales_correctly():
    """FA + MD penalties should scale with cardinality mismatch."""
    B = 1
    P_true, P_hat = 3, 5  # 2 false alarms expected
    torch.manual_seed(0)
    h_true = torch.zeros(B, P_true, dtype=torch.complex64, device=DEVICE)
    theta_true = torch.zeros(B, P_true, 2, device=DEVICE)
    h_hat = torch.zeros(B, P_hat, dtype=torch.complex64, device=DEVICE)
    theta_hat = torch.zeros(B, P_hat, 2, device=DEVICE)
    loss = hungarian_set_loss(h_hat, theta_hat, h_true, theta_true,
                              mu_fa=1.0, mu_md=1.0)
    # Loss = 0 (matching) + (5-3)*1.0 = 2.0
    assert abs(loss.item() - 2.0) < 1e-4, f"expected 2.0, got {loss.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

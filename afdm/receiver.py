"""Full unfolded uncertainty-gated V-EM AFDM receiver (paper Section IV.B).

Composes the modules built in P1-P3:
  * FastAFDMOperator (paper eq. 8)
  * Off-grid support recovery (SupportRecovery)
  * V-EM primitives (h_step_damped_ridge, posterior_covariance,
    safeguarded_lm_theta_step, symbol_step_soft_posterior)
  * Permutation-equivariant Set-Transformer (SetTransformer)
  * Uncertainty gate (UncertaintyGate)

Each layer contains learned scalars (alpha_t, lambda_t, sigma_w_t^2 calibration,
beta_t, omega_t) plus the set-attention weights. Learned parameters are shared
per-layer index (not tied across depths), matching the paper's T=8 untied layers.

The forward pass returns per-layer states so that the training loss can supervise
h, theta, x at every depth (deep supervision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classical import build_regression_matrix, cg_solve
from .operators import FastAFDMOperator
from .set_transformer import SetTransformer, UncertaintyGate
from .support import SupportRecovery
from .system import AFDMSystem
from .vem import (
    h_step_damped_ridge,
    posterior_covariance,
    safeguarded_lm_theta_step,
    symbol_step_soft_posterior,
)


class UGVEMReceiverLayer(nn.Module):
    """A single V-EM layer with learned scalars and the Set-Transformer correction.

    Learned parameters:
      * alpha_raw: h-step relaxation, mapped to (0, 2) via 2 * sigmoid.
      * lambda_raw: h-step ridge, mapped to positive via softplus.
      * sigma_calib_raw: multiplicative calibration on the block sigma_w^2.
      * beta_raw: symbol damping in (0, 1), mapped via sigmoid.
      * omega_raw: symbol inverse temperature, mapped to positive via softplus.
      * gamma_raw: LM support step size, mapped to (0, 1) via sigmoid.
      * set_transformer: correction network.
      * gate: uncertainty gate on the learned correction.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 3,
        max_delta_norm: float = 5.0,
        gate_u_ref: float = 1e-2,
        gate_init_a: float = 1.0,
        gate_init_b: float = 0.0,
    ) -> None:
        super().__init__()
        # Init learned scalars near sensible defaults.
        # alpha: 2 * sigmoid(alpha_raw). alpha_raw = 0 -> alpha=1 (exact ridge).
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        # lambda: softplus(lambda_raw). Init small (~2.5e-3) — a large ridge biases
        # h_hat toward zero when signal_pow ~ 1 (empirically better convergence).
        self.lambda_raw = nn.Parameter(torch.tensor(-6.0))  # softplus(-6) ~ 2.5e-3
        # Multiplicative calibration on sigma_w^2: sigma_calib = softplus(sigma_calib_raw).
        # softplus(0.5413) ≈ 1.0
        self.sigma_calib_raw = nn.Parameter(torch.tensor(0.5413))
        # beta: sigmoid(beta_raw). beta_raw = 5.0 -> beta ≈ 0.9933 (mostly new estimate).
        self.beta_raw = nn.Parameter(torch.tensor(5.0))
        # omega: softplus(omega_raw). Init omega ≈ 20 (2 * softplus(3.0) * 10 rescaling ≈ 30 — but
        # we use 10 * softplus so softplus(2.0) * 10 ≈ 21.3 is a good default).
        self.omega_raw = nn.Parameter(torch.tensor(2.0))
        # gamma (LM step size): sigmoid. gamma_raw = 0.0 -> gamma = 0.5.
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))
        # Set-Transformer
        self.set_transformer = SetTransformer(
            input_dim=5, d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
            output_dim=2, max_delta_norm=max_delta_norm,
        )
        self.gate = UncertaintyGate(u_ref=gate_u_ref, init_a=gate_init_a, init_b=gate_init_b)
        # LM safeguarding hyperparameters (not learned — architectural constants).
        self.max_step = 0.15
        self.slack = 1e-4
        self.max_backtracks = 3

    # ------------------------------------------------------------------
    # Learned-scalar accessors
    # ------------------------------------------------------------------
    def alpha(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.alpha_raw)  # (0, 2)

    def lam(self) -> torch.Tensor:
        return F.softplus(self.lambda_raw)  # (0, inf)

    def sigma_calib(self) -> torch.Tensor:
        return F.softplus(self.sigma_calib_raw)

    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_raw)  # (0, 1)

    def omega(self) -> torch.Tensor:
        return 10.0 * F.softplus(self.omega_raw)  # positive, scaled

    def gamma_lm(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_raw)  # (0, 1)

    # ------------------------------------------------------------------
    # Layer forward
    # ------------------------------------------------------------------
    def forward(
        self,
        system: AFDMSystem,
        r: torch.Tensor,
        y: torch.Tensor,
        eta_h_old: torch.Tensor,
        ell_old: torch.Tensor,
        kappa_old: torch.Tensor,
        x_hat_old: torch.Tensor,
        z_prev: torch.Tensor | None,
        sigma_w2_block: torch.Tensor | float,
        constellation: torch.Tensor,
        pilot_positions: torch.Tensor,
        pilot_values: torch.Tensor,
        K_cg: int = 10,
        refine_theta: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Execute one V-EM layer.

        Returns a dict with keys eta_h, ell, kappa, x_mean, p_ms, z, v, g, delta.
        All returned tensors preserve autograd (except the theta step which uses
        detach + explicit acceptance).
        """
        alpha_t = self.alpha()
        lam_t = self.lam()
        sigma_calib_t = self.sigma_calib()
        beta_t = self.beta()
        omega_t = self.omega()
        gamma_t = self.gamma_lm()

        # Effective noise variance used by the layer (paper eq. 30). Keep as tensor.
        sigma_block_t = sigma_w2_block if torch.is_tensor(sigma_w2_block) \
            else torch.tensor(float(sigma_w2_block), device=r.device)
        sigma_w2_eff = sigma_block_t * sigma_calib_t

        # 1. Build regression matrix at (ell_old, kappa_old, x_hat_old)
        A = build_regression_matrix(system, ell_old, kappa_old, x_hat_old)

        # 2. h-mean update (damped exact ridge)
        eta_tilde = h_step_damped_ridge(A, r, lam=lam_t, alpha=alpha_t, eta_old=eta_h_old)

        # 3. Posterior covariance
        V_h, v = posterior_covariance(A, lam=lam_t, sigma_w2=sigma_w2_eff)

        # 4. Set-Transformer features (real)
        features = torch.stack([
            eta_tilde.real,
            eta_tilde.imag,
            ell_old.to(eta_tilde.real.dtype),
            kappa_old.to(eta_tilde.real.dtype),
            v.sqrt(),
        ], dim=-1)  # (B, P, 5)
        delta = self.set_transformer(features)  # (B, P) complex

        # 5. Uncertainty gate
        g = self.gate(v)  # (B,)

        # 6. Refined h
        eta_h = eta_tilde + g.unsqueeze(-1) * delta

        # 7. Support LM step (safeguarded); skip if refine_theta=False.
        # LM step uses .item() for its own scalars (it's not differentiable through
        # the accept/reject rule anyway; gradients into theta parameters would need a
        # differentiable relaxation).
        if refine_theta:
            ell, kappa, accepted = safeguarded_lm_theta_step(
                system, r, eta_h, x_hat_old, ell_old, kappa_old,
                sigma_w2=float(sigma_w2_eff.item()), v_h=v,
                gamma_lr=float(gamma_t.item()), max_step=self.max_step,
                slack=self.slack, max_backtracks=self.max_backtracks,
            )
        else:
            ell, kappa = ell_old, kappa_old
            accepted = torch.zeros(ell.shape[0], dtype=torch.bool, device=ell.device)

        # 8. Symbol step (CG-MMSE + soft posterior). Pass tensors so omega/beta
        # can receive gradients.
        x_mean, p_ms, z = symbol_step_soft_posterior(
            system, y, ell, kappa, eta_h, sigma_w2=float(sigma_w2_eff.item()),
            omega=omega_t, constellation=constellation,
            pilot_positions=pilot_positions, pilot_values=pilot_values,
            K_cg=K_cg, z_prev=z_prev, beta=beta_t,
        )
        return {
            "eta_h": eta_h,
            "ell": ell,
            "kappa": kappa,
            "x_mean": x_mean,
            "p_ms": p_ms,
            "z": z,
            "v": v,
            "g": g,
            "delta": delta,
            "accepted": accepted,
        }


class UGVEMReceiver(nn.Module):
    """Uncertainty-gated V-EM AFDM receiver, T layers deep."""

    def __init__(
        self,
        system: AFDMSystem,
        support_recovery: SupportRecovery,
        constellation: torch.Tensor,
        pilot_positions: torch.Tensor,
        pilot_values: torch.Tensor,
        T: int = 8,
        K_cg: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 3,
        max_delta_norm: float = 5.0,
        gate_u_ref: float = 1e-2,
    ) -> None:
        super().__init__()
        self.system = system
        self.support_recovery = support_recovery
        # Register buffers so they move with .to(device)
        self.register_buffer("constellation", constellation)
        self.register_buffer("pilot_positions", pilot_positions)
        self.register_buffer("pilot_values", pilot_values)
        self.T = T
        self.K_cg = K_cg
        self.layers = nn.ModuleList([
            UGVEMReceiverLayer(
                d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
                max_delta_norm=max_delta_norm, gate_u_ref=gate_u_ref,
            )
            for _ in range(T)
        ])

    def forward(
        self,
        r: torch.Tensor,
        sigma_w2_block: torch.Tensor | float,
        refine_theta: bool = True,
        return_layer_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Parameters
        ----------
        r                : (B, N) time-domain received signal.
        sigma_w2_block   : absolute noise variance for the block.
        refine_theta     : whether to run the safeguarded LM support step (default True).
        return_layer_states : if True, returns a list of per-layer state dicts as well.

        Returns
        -------
        dict with keys x_mean, p_ms, eta_h, ell, kappa, and optionally layer_states.
        """
        B, N = r.shape
        device = r.device
        dtype = r.dtype

        # 1. Off-grid support recovery on r using pilot-only reference (non-differentiable).
        x_pilot = torch.zeros(N, dtype=dtype, device=device)
        x_pilot[self.pilot_positions] = self.pilot_values
        s_pilot = self.system.idaft(x_pilot.unsqueeze(0))[0]
        with torch.no_grad():
            ell, kappa, p_hat = self.support_recovery(r, s_pilot)

        # 2. Initialize x_hat with pilots + zeros for data, y = DAFT(r), eta_h = 0.
        x_hat = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)
        y = self.system.daft(r)
        eta_h = torch.zeros(B, ell.shape[1], dtype=dtype, device=device)
        z_prev = None

        # 3. Run T layers.
        layer_states = []
        for t in range(self.T):
            state = self.layers[t](
                system=self.system, r=r, y=y, eta_h_old=eta_h,
                ell_old=ell, kappa_old=kappa, x_hat_old=x_hat,
                z_prev=z_prev, sigma_w2_block=sigma_w2_block,
                constellation=self.constellation,
                pilot_positions=self.pilot_positions,
                pilot_values=self.pilot_values,
                K_cg=self.K_cg, refine_theta=refine_theta,
            )
            eta_h, ell, kappa = state["eta_h"], state["ell"], state["kappa"]
            x_hat = state["x_mean"]
            z_prev = state["z"]
            if return_layer_states:
                layer_states.append(state)

        result = {
            "x_mean": x_hat,
            "p_ms": layer_states[-1]["p_ms"] if return_layer_states else state["p_ms"],
            "eta_h": eta_h,
            "ell": ell,
            "kappa": kappa,
            "p_hat": p_hat,
        }
        if return_layer_states:
            result["layer_states"] = layer_states
        return result

    def hard_decision(self, p_ms: torch.Tensor) -> torch.Tensor:
        """Return argmax categorical labels; shape (B, N)."""
        return p_ms.argmax(dim=-1)

"""Redesign smoke A: replace gate*delta composition with sigma_w * scale * delta.

Rationale: current design η = η_LS + g(v)·Δ has bad optimization landscape
because g and Δ have multiplicative-vanishing gradients at cold-start states.

Redesign A: η = η_LS + c · sqrt(sigma_w2) · Δ_norm
  where c is a learned per-layer scalar (softplus for positivity),
  Δ_norm = Δ / max(||Δ||, epsilon) is unit-normalized,
  sqrt(sigma_w2) provides high-SNR consistency (as noise → 0, correction → 0).

Advantages:
  * Single multiplicative factor c, no gate composition.
  * Gradient of loss w.r.t. c is well-scaled (no vanishing).
  * Δ_norm has bounded output magnitude regardless of network init.
  * Preserves high-SNR consistency by construction.
"""

from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from afdm.experiments import (
    ExperimentConfig, evaluate_receiver_sweep, evaluate_classical_sweep,
    genie_mmse_sweep,
)
from afdm.classical import ClassicalCGDetector
from afdm.receiver import UGVEMReceiver, UGVEMReceiverLayer
from afdm.training import TrainingConfig, train


class RedesignAReceiverLayer(UGVEMReceiverLayer):
    """Layer with η = η_LS + c · sqrt(σ_w²) · Δ_norm instead of η + g·Δ."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Learned per-layer scale factor (softplus-positive).
        self.c_raw = nn.Parameter(torch.tensor(-2.0))  # softplus(-2) ≈ 0.127

    def c_scale(self) -> torch.Tensor:
        return F.softplus(self.c_raw)

    def forward(self, system, r, y, eta_h_old, ell_old, kappa_old, x_hat_old,
                z_prev, sigma_w2_block, constellation, pilot_positions, pilot_values,
                K_cg=10, refine_theta=True):
        alpha_t = self.alpha()
        lam_t = self.lam()
        sigma_calib_t = self.sigma_calib()
        beta_t = self.beta()
        omega_t = self.omega()
        gamma_t = self.gamma_lm()

        sigma_block_t = sigma_w2_block if torch.is_tensor(sigma_w2_block) \
            else torch.tensor(float(sigma_w2_block), device=r.device)
        sigma_w2_eff = sigma_block_t * sigma_calib_t

        # Build regression matrix and h-step (same as original)
        from afdm.classical import build_regression_matrix
        from afdm.vem import (h_step_damped_ridge, posterior_covariance,
                              safeguarded_lm_theta_step, symbol_step_soft_posterior)
        A = build_regression_matrix(system, ell_old, kappa_old, x_hat_old)
        eta_tilde = h_step_damped_ridge(A, r, lam=lam_t, alpha=alpha_t, eta_old=eta_h_old)
        V_h, v = posterior_covariance(A, lam=lam_t, sigma_w2=sigma_w2_eff)

        # Set-Transformer features (same as original)
        features = torch.stack([
            eta_tilde.real, eta_tilde.imag,
            ell_old.to(eta_tilde.real.dtype), kappa_old.to(eta_tilde.real.dtype),
            v.sqrt(),
        ], dim=-1)
        delta = self.set_transformer(features)  # (B, P) complex

        # REDESIGN: unit-normalize delta per-batch, then scale by c * sqrt(sigma_w2_eff)
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        delta_unit = delta / delta_norm  # ||δ||=1 per batch element
        # scale = c * sqrt(sigma_w2_eff)   → vanishes as sigma_w → 0
        scale = self.c_scale() * torch.sqrt(sigma_w2_eff.clamp(min=1e-18))
        # scalar scale factor is real; delta_unit is complex
        # Approximate "typical h magnitude" is sqrt(P) for unit-power channel.
        # So the correction magnitude scale ~ c * sigma_w * sqrt(P)
        eta_h = eta_tilde + (scale * delta_unit.norm(dim=-1, keepdim=True)) * (delta_unit)

        # Simpler: just eta = eta_tilde + scale * delta_unit
        eta_h = eta_tilde + scale.to(delta_unit.dtype) * delta_unit

        # (Fake gate for legacy caller compatibility)
        g_placeholder = self.c_scale() * torch.sqrt(sigma_w2_eff.clamp(min=1e-18))
        g_placeholder = g_placeholder.expand(r.shape[0])

        # Support LM step (same as original)
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

        # Symbol step (same as original)
        x_mean, p_ms, z = symbol_step_soft_posterior(
            system, y, ell, kappa, eta_h, sigma_w2=float(sigma_w2_eff.item()),
            omega=omega_t, constellation=constellation,
            pilot_positions=pilot_positions, pilot_values=pilot_values,
            K_cg=K_cg, z_prev=z_prev, beta=beta_t,
        )
        return {
            "eta_h": eta_h, "ell": ell, "kappa": kappa,
            "x_mean": x_mean, "p_ms": p_ms, "z": z,
            "v": v, "g": g_placeholder, "delta": delta, "accepted": accepted,
        }


class RedesignAReceiver(UGVEMReceiver):
    """UGVEMReceiver with RedesignAReceiverLayer instead of default."""
    def __init__(self, *args, **kwargs):
        # Skip the parent's layer construction; we'll build our own.
        nn.Module.__init__(self)
        self.system = kwargs["system"]
        self.support_recovery = kwargs["support_recovery"]
        self.register_buffer("constellation", kwargs["constellation"])
        self.register_buffer("pilot_positions", kwargs["pilot_positions"])
        self.register_buffer("pilot_values", kwargs["pilot_values"])
        self.T = kwargs.get("T", 8)
        self.K_cg = kwargs.get("K_cg", 10)
        self.layers = nn.ModuleList([
            RedesignAReceiverLayer(
                d_model=kwargs.get("d_model", 64),
                n_heads=kwargs.get("n_heads", 4),
                n_blocks=kwargs.get("n_blocks", 3),
                max_delta_norm=kwargs.get("max_delta_norm", 5.0),
                gate_u_ref=kwargs.get("gate_u_ref", 1e-2),
            )
            for _ in range(self.T)
        ])


def build_redesign_A(cfg):
    pp, pv = cfg.pilots()
    return RedesignAReceiver(
        system=cfg.system(), support_recovery=cfg.support_recovery(),
        constellation=cfg.constellation(), pilot_positions=pp, pilot_values=pv,
        T=cfg.T, K_cg=cfg.K_cg, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_blocks=cfg.n_blocks,
    ).to(cfg.device)


def main():
    cfg = ExperimentConfig(
        N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32,
        T=8, K_cg=10, d_model=64, n_heads=4, n_blocks=3, P_max=3, seed=0,
    )
    snrs = [5.0, 15.0, 25.0]
    n_batches = 3; batch_size = 32

    print("=" * 80)
    print("REDESIGN A SMOKE: sigma_w-scaled residual (no gate)")
    print(f"Config: N={cfg.N}, N_p={cfg.N_p}, P={cfg.P}")
    print("=" * 80)

    pp, pv = cfg.pilots()
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(),
        constellation=cfg.constellation(), pilot_positions=pp, pilot_values=pv,
        T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    r_class = evaluate_classical_sweep(classical, cfg, snrs, n_batches, batch_size, seed=42)
    r_genie = genie_mmse_sweep(cfg, snrs, n_batches, batch_size, seed=42)

    torch.manual_seed(0)
    receiver = build_redesign_A(cfg)

    print("Pre-training:")
    r_pre = evaluate_receiver_sweep(receiver, cfg, snrs, n_batches, batch_size, seed=42)
    for snr in snrs:
        print(f"  {snr:>4.1f}dB: pre {r_pre[snr]['ser']:.3e}  vs classical {r_class[snr]['ser']:.3e}  vs genie {r_genie[snr]['ser']:.3e}")

    tc = TrainingConfig(
        lr=5e-4, n_epochs=30, steps_per_epoch=50, batch_size=32,
        snr_db_min=5.0, snr_db_max=25.0, grad_clip=1.0,
        val_every=10, val_batches=2, val_snr_dbs=(15.0,),
        layer_gamma=0.7, mu_ce=5.0, eta_anchor=0.0,
        hungarian_kwargs=dict(w_h=1.0, w_ell=0.2, w_kap=0.2, mu_fa=0.1, mu_md=0.1),
        log_every=50,
    )
    print(f"\nTraining {tc.n_epochs} epochs...")
    t0 = time.time()
    history = train(receiver, cfg.system(), cfg.channel(), cfg.constellation(),
                    pp, pv, tc, seed=0, verbose=False)
    print(f"  {time.time()-t0:.1f}s; loss: init {history['train_loss'][0]:.3f} -> end {history['train_loss'][-1]:.3f}")

    r_post = evaluate_receiver_sweep(receiver, cfg, snrs, n_batches, batch_size, seed=42)
    print("\nResults:")
    print(f"{'SNR':<6s}  {'Genie':>10s}  {'Classical':>10s}  {'Pre':>10s}  {'Post':>10s}  {'Delta':>10s}")
    for snr in snrs:
        pre = r_pre[snr]["ser"]; post = r_post[snr]["ser"]
        cls = r_class[snr]["ser"]; gen = r_genie[snr]["ser"]
        print(f"{snr:>4.1f}dB  {gen:>10.3e}  {cls:>10.3e}  {pre:>10.3e}  {post:>10.3e}  {pre-post:+.3e}")

    post_15 = r_post[15.0]["ser"]; cls_15 = r_class[15.0]["ser"]
    ratio = post_15 / cls_15
    verdict = "PASS" if ratio < 0.5 else ("MARGINAL" if ratio < 0.9 else "FAIL")
    print(f"\n{verdict}: post 15dB = {post_15:.3e} ({ratio:.0%} of classical)")


if __name__ == "__main__":
    main()

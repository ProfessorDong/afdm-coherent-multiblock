"""Training loop for the UGVEMReceiver.

Online-generation training: at each iteration we sample a fresh batch of
(channel, symbols, noise) and take an Adam step on the composite loss defined in
`afdm.loss.compose_training_loss`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .channels import UniformFractionalChannel
from .loss import compose_training_loss
from .operators import FastAFDMOperator
from .receiver import UGVEMReceiver
from .system import AFDMSystem


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    n_epochs: int = 100
    steps_per_epoch: int = 100
    batch_size: int = 32
    snr_db_min: float = 0.0
    snr_db_max: float = 30.0
    grad_clip: float = 5.0
    val_every: int = 5
    val_batches: int = 4
    val_snr_dbs: tuple = (5.0, 15.0, 25.0)
    layer_gamma: float = 0.7
    mu_ce: float = 1.0
    mu_theta: float = 0.5
    eta_anchor: float = 0.0  # anchor term disabled for the first training pass
    hungarian_kwargs: dict = field(default_factory=lambda: dict(w_h=1.0, w_ell=0.5, w_kap=0.5, mu_fa=0.3, mu_md=0.3))
    use_amp: bool = False   # mixed precision (autocast) — disabled for CG stability
    log_every: int = 20


def sample_batch(
    system: AFDMSystem,
    channel: UniformFractionalChannel,
    constellation: torch.Tensor,
    pilot_positions: torch.Tensor,
    pilot_values: torch.Tensor,
    batch_size: int,
    snr_db: float | None,
    snr_db_range: tuple[float, float] | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample one training batch: channel, symbols, noise, absolute noise variance.

    Returns dict with keys r, y, x_true, labels, h_true, theta_true, sigma_w2_block, pilot_mask.
    """
    device = system.device
    N = system.N
    S = constellation.numel()
    d = channel.sample(batch_size, generator=generator)  # ell, kappa, h
    if generator is None:
        idx = torch.randint(0, S, (batch_size, N), device=device)
    else:
        idx = torch.randint(0, S, (batch_size, N), device=device, generator=generator)
    x = constellation[idx]  # (B, N)
    x[:, pilot_positions] = pilot_values.unsqueeze(0)
    labels = (x.unsqueeze(-1) - constellation.reshape(1, 1, -1)).abs().argmin(dim=-1)

    op = FastAFDMOperator(system=system, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    y_clean = op.matvec(x)
    signal_pow = (y_clean.abs() ** 2).mean()
    if snr_db is None:
        assert snr_db_range is not None
        lo, hi = snr_db_range
        snr_db_actual = lo + (hi - lo) * torch.rand(1, generator=generator, device=device).item()
    else:
        snr_db_actual = snr_db
    sigma_w2 = 10 ** (-snr_db_actual / 10)
    noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
    if generator is None:
        w = torch.randn_like(y_clean) * noise_std
    else:
        w = torch.randn(y_clean.shape, dtype=y_clean.dtype, device=device, generator=generator) * noise_std
    y = y_clean + w
    r = system.idaft(y)
    abs_noise = (signal_pow * sigma_w2).item()
    pilot_mask = torch.ones(N, dtype=torch.bool, device=device)
    pilot_mask[pilot_positions] = False
    pilot_mask = pilot_mask.unsqueeze(0).expand(batch_size, -1)
    theta_true = torch.stack([d["ell"], d["kappa"]], dim=-1)
    return {
        "r": r, "y": y, "x_true": x, "labels": labels,
        "h_true": d["h"], "theta_true": theta_true,
        "sigma_w2_block": abs_noise, "pilot_mask": pilot_mask,
        "snr_db": snr_db_actual,
    }


@torch.no_grad()
def evaluate_snr(
    receiver: UGVEMReceiver,
    system: AFDMSystem,
    channel: UniformFractionalChannel,
    constellation: torch.Tensor,
    pilot_positions: torch.Tensor,
    pilot_values: torch.Tensor,
    snr_db: float,
    n_batches: int,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, float]:
    """Evaluate receiver at a fixed SNR over multiple batches.

    Returns dict with keys ser, nmse_h, delay_rmse, doppler_rmse.
    """
    receiver.eval()
    ser_sum = 0.0
    nmse_sum = 0.0
    d_rmse_sum = 0.0
    k_rmse_sum = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, constellation, pilot_positions, pilot_values,
                             batch_size=batch_size, snr_db=snr_db, generator=generator)
        out = receiver(batch["r"], sigma_w2_block=batch["sigma_w2_block"])
        hard = out["p_ms"].argmax(dim=-1)
        ser = ((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum()
        # Simple gain NMSE using first-P matches (batch may have different P_hat)
        P_true = batch["h_true"].shape[1]
        h_hat = out["eta_h"][:, :P_true]
        nmse = ((h_hat - batch["h_true"]).abs() ** 2).sum() / (batch["h_true"].abs() ** 2).sum()
        ell_hat = out["ell"][:, :P_true]
        kap_hat = out["kappa"][:, :P_true]
        d_rmse = ((ell_hat - batch["theta_true"][..., 0]) ** 2).mean().sqrt()
        k_rmse = ((kap_hat - batch["theta_true"][..., 1]) ** 2).mean().sqrt()
        ser_sum += ser.item()
        nmse_sum += nmse.item()
        d_rmse_sum += d_rmse.item()
        k_rmse_sum += k_rmse.item()
    receiver.train()
    return {
        "ser": ser_sum / n_batches,
        "nmse_h": nmse_sum / n_batches,
        "delay_rmse": d_rmse_sum / n_batches,
        "doppler_rmse": k_rmse_sum / n_batches,
    }


def train(
    receiver: UGVEMReceiver,
    system: AFDMSystem,
    channel: UniformFractionalChannel,
    constellation: torch.Tensor,
    pilot_positions: torch.Tensor,
    pilot_values: torch.Tensor,
    config: TrainingConfig = TrainingConfig(),
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Train the receiver and return history dict.

    Uses online sampling: channels and symbols are freshly sampled each step,
    so there is no fixed training set.
    """
    device = system.device
    train_gen = torch.Generator(device=device); train_gen.manual_seed(seed)
    val_gen = torch.Generator(device=device); val_gen.manual_seed(seed + 10)

    receiver.train()
    optimizer = optim.Adam(receiver.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)

    history = {"train_loss": [], "val": []}
    step_count = 0
    for epoch in range(config.n_epochs):
        epoch_losses = []
        for step in range(config.steps_per_epoch):
            batch = sample_batch(
                system, channel, constellation, pilot_positions, pilot_values,
                batch_size=config.batch_size, snr_db=None,
                snr_db_range=(config.snr_db_min, config.snr_db_max),
                generator=train_gen,
            )
            out = receiver(
                batch["r"], sigma_w2_block=batch["sigma_w2_block"],
                refine_theta=True, return_layer_states=True,
            )
            loss, breakdown = compose_training_loss(
                out, batch["x_true"], batch["labels"],
                batch["h_true"], batch["theta_true"],
                batch["pilot_mask"],
                layer_gamma=config.layer_gamma,
                mu_theta=config.mu_theta,
                mu_ce=config.mu_ce,
                eta_anchor=config.eta_anchor,
                hungarian_kwargs=config.hungarian_kwargs,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(receiver.parameters(), config.grad_clip)
            optimizer.step()
            step_count += 1
            epoch_losses.append(loss.item())
            if verbose and step % config.log_every == 0:
                print(f"  ep {epoch:03d} step {step:03d} loss {loss.item():.4f} "
                      f"[set={breakdown['set_loss']:.3f}, ce={breakdown['ce']:.3f}]")
        history["train_loss"].append(sum(epoch_losses) / len(epoch_losses))
        scheduler.step()
        if (epoch + 1) % config.val_every == 0 or epoch == config.n_epochs - 1:
            val_results = {}
            for snr in config.val_snr_dbs:
                val_results[snr] = evaluate_snr(
                    receiver, system, channel, constellation,
                    pilot_positions, pilot_values,
                    snr_db=snr, n_batches=config.val_batches,
                    batch_size=config.batch_size, generator=val_gen,
                )
            history["val"].append({"epoch": epoch, "results": val_results})
            if verbose:
                for snr, m in val_results.items():
                    print(f"  [val] ep {epoch:03d} SNR {snr:5.1f}dB: SER={m['ser']:.4e} "
                          f"NMSE={m['nmse_h']:.3e} delay_RMSE={m['delay_rmse']:.3f}")
    return history

"""Experiment orchestration: config, training, and evaluation utilities.

Central hub for P4 experiments. Every figure/table script uses these functions
so that all experiments share a common setup.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch

from .channels import UniformFractionalChannel
from .classical import ClassicalCGDetector, cg_solve
from .jpnce_sbl import JPNCESBLDetector
from .operators import FastAFDMOperator
from .pbigabp import PBiGaBPDetector
from .pilots import uniform_daft_pilots
from .receiver import UGVEMReceiver
from .support import SupportRecovery
from .system import AFDMSystem
from .training import TrainingConfig, sample_batch, train, evaluate_snr


DEFAULT_QPSK = None  # populated lazily


def qpsk_constellation(device: str = "cuda:0") -> torch.Tensor:
    return torch.tensor(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], device=device, dtype=torch.complex64
    ) / (2 ** 0.5)


def qam16_constellation(device: str = "cuda:0") -> torch.Tensor:
    pts = []
    for r in [-3, -1, 1, 3]:
        for i in [-3, -1, 1, 3]:
            pts.append(complex(r, i))
    c = torch.tensor(pts, device=device, dtype=torch.complex64)
    return c / (c.abs() ** 2).mean().sqrt()


@dataclass
class ExperimentConfig:
    """One-stop config for building a system + channel + receiver."""

    # System
    N: int = 128
    kappa_max: float = 5.0
    ell_max: float = 10.0
    device: str = "cuda:0"

    # Channel
    P: int = 5
    decay_db_per_path: float = 2.0

    # Pilots
    N_p: int = 16

    # Receiver
    T: int = 8
    K_cg: int = 10
    d_model: int = 64
    n_heads: int = 4
    n_blocks: int = 3
    P_max: int = 6      # candidate count for support recovery
    max_delta_norm: float = 5.0
    gate_u_ref: float = 1e-2

    # Constellation
    constellation_kind: str = "qpsk"  # "qpsk" or "qam16"

    # Reproducibility
    seed: int = 0

    def system(self) -> AFDMSystem:
        return AFDMSystem(N=self.N, kappa_max=int(self.kappa_max), ell_max=int(self.ell_max), device=self.device)

    def channel(self, P: int | None = None, kappa_max: float | None = None, ell_max: float | None = None) -> UniformFractionalChannel:
        return UniformFractionalChannel(
            P=P if P is not None else self.P,
            ell_max=ell_max if ell_max is not None else self.ell_max,
            kappa_max=kappa_max if kappa_max is not None else self.kappa_max,
            decay_db_per_path=self.decay_db_per_path,
            device=self.device,
        )

    def constellation(self) -> torch.Tensor:
        if self.constellation_kind == "qpsk":
            return qpsk_constellation(self.device)
        elif self.constellation_kind == "qam16":
            return qam16_constellation(self.device)
        raise ValueError(f"unknown constellation {self.constellation_kind}")

    def pilots(self) -> tuple[torch.Tensor, torch.Tensor]:
        pp = uniform_daft_pilots(N=self.N, N_p=self.N_p, device=self.device)
        gen = torch.Generator(device=self.device); gen.manual_seed(self.seed + 999)
        constellation = self.constellation()
        idx = torch.randint(0, constellation.numel(), (self.N_p,), device=self.device, generator=gen)
        return pp, constellation[idx]

    def support_recovery(self) -> SupportRecovery:
        return SupportRecovery(
            N=self.N, N_cp=int(self.ell_max), kappa_max=self.kappa_max, ell_max=self.ell_max,
            P_max=self.P_max,
        )

    def receiver(self, refine_theta_enabled: bool = True) -> UGVEMReceiver:
        pp, pv = self.pilots()
        return UGVEMReceiver(
            system=self.system(), support_recovery=self.support_recovery(),
            constellation=self.constellation(), pilot_positions=pp, pilot_values=pv,
            T=self.T, K_cg=self.K_cg, d_model=self.d_model, n_heads=self.n_heads,
            n_blocks=self.n_blocks, max_delta_norm=self.max_delta_norm,
            gate_u_ref=self.gate_u_ref,
        ).to(self.device)


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------
def build_ablation(name: str, config: ExperimentConfig) -> UGVEMReceiver:
    """Build a variant of the receiver for the ablation study.

    Variants:
      * "proposed": full receiver (gate + set-attention + LM support step).
      * "gate":     +gate, no Set-Transformer (delta is zero).
      * "attention": +Set-Transformer, no gate (g==1 always).
      * "scalars":  no Set-Transformer, no gate (delta*g = 0 always).
    """
    rx = config.receiver()
    if name == "proposed":
        return rx
    if name == "scalars":
        # Zero out Set-Transformer output at inference time by re-initializing
        # its output projection to zero (and freezing).
        for layer in rx.layers:
            layer.set_transformer.output_proj.weight.data.zero_()
            layer.set_transformer.output_proj.bias.data.zero_()
            for p in layer.set_transformer.parameters():
                p.requires_grad = False
        return rx
    if name == "attention":
        # Force gate value to 1: set gate.b to a huge positive value.
        for layer in rx.layers:
            layer.gate.b.data.fill_(1e6)
            for p in layer.gate.parameters():
                p.requires_grad = False
        return rx
    if name == "gate":
        # Zero out Set-Transformer (as in "scalars") but keep gate trainable.
        for layer in rx.layers:
            layer.set_transformer.output_proj.weight.data.zero_()
            layer.set_transformer.output_proj.bias.data.zero_()
            for p in layer.set_transformer.parameters():
                p.requires_grad = False
        return rx
    raise ValueError(f"unknown ablation variant {name}")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_receiver(
    receiver: UGVEMReceiver,
    config: ExperimentConfig,
    training_config: TrainingConfig,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    system = config.system()
    channel = config.channel()
    constellation = config.constellation()
    pp, pv = config.pilots()
    history = train(
        receiver, system, channel, constellation, pp, pv,
        config=training_config, seed=config.seed, verbose=verbose,
    )
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": receiver.state_dict(),
            "config": asdict(config),
            "training_config": asdict(training_config),
            "history": history,
        }, checkpoint_path)
        if verbose:
            print(f"Saved checkpoint: {checkpoint_path}")
    return history


def load_receiver(checkpoint_path: str, refine_theta_enabled: bool = True) -> tuple[UGVEMReceiver, ExperimentConfig]:
    ckpt = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
    cfg = ExperimentConfig(**{k: v for k, v in ckpt["config"].items()})
    rx = cfg.receiver(refine_theta_enabled=refine_theta_enabled)
    rx.load_state_dict(ckpt["state_dict"])
    rx.eval()
    return rx, cfg


# ---------------------------------------------------------------------------
# Evaluation helpers (multi-SNR sweep, multiple metrics)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_receiver_sweep(
    receiver: UGVEMReceiver,
    config: ExperimentConfig,
    snr_dbs: list[float],
    n_batches_per_snr: int = 4,
    batch_size: int = 32,
    channel: Optional[UniformFractionalChannel] = None,
    seed: int = 42,
    refine_theta: bool = True,
) -> dict[float, dict[str, float]]:
    system = config.system()
    channel = channel if channel is not None else config.channel()
    constellation = config.constellation()
    pp, pv = config.pilots()
    gen = torch.Generator(device=config.device); gen.manual_seed(seed)
    results = {}
    for snr in snr_dbs:
        ser = 0.0; nmse = 0.0; d_rmse = 0.0; k_rmse = 0.0
        for _ in range(n_batches_per_snr):
            batch = sample_batch(system, channel, constellation, pp, pv,
                                 batch_size=batch_size, snr_db=snr, generator=gen)
            out = receiver(batch["r"], sigma_w2_block=batch["sigma_w2_block"], refine_theta=refine_theta)
            hard = out["p_ms"].argmax(dim=-1)
            ser_b = ((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum()
            P_true = batch["h_true"].shape[1]
            h_hat = out["eta_h"][:, :P_true]
            nmse_b = ((h_hat - batch["h_true"]).abs() ** 2).sum() / (batch["h_true"].abs() ** 2).sum()
            ell_hat = out["ell"][:, :P_true]
            kap_hat = out["kappa"][:, :P_true]
            d_rmse_b = ((ell_hat - batch["theta_true"][..., 0]) ** 2).mean().sqrt()
            k_rmse_b = ((kap_hat - batch["theta_true"][..., 1]) ** 2).mean().sqrt()
            ser += ser_b.item(); nmse += nmse_b.item()
            d_rmse += d_rmse_b.item(); k_rmse += k_rmse_b.item()
        results[snr] = {
            "ser": ser / n_batches_per_snr,
            "nmse_h": nmse / n_batches_per_snr,
            "delay_rmse": d_rmse / n_batches_per_snr,
            "doppler_rmse": k_rmse / n_batches_per_snr,
        }
    return results


@torch.no_grad()
def evaluate_classical_sweep(
    detector,   # ClassicalCGDetector | PBiGaBPDetector | JPNCESBLDetector
    config: ExperimentConfig,
    snr_dbs: list[float],
    n_batches_per_snr: int = 4,
    batch_size: int = 32,
    channel: Optional[UniformFractionalChannel] = None,
    seed: int = 42,
) -> dict[float, dict[str, float]]:
    """Evaluate a classical baseline at a range of SNRs."""
    system = config.system()
    channel = channel if channel is not None else config.channel()
    constellation = config.constellation()
    pp, pv = config.pilots()
    gen = torch.Generator(device=config.device); gen.manual_seed(seed)
    results = {}
    for snr in snr_dbs:
        ser = 0.0; nmse = 0.0; d_rmse = 0.0; k_rmse = 0.0
        for _ in range(n_batches_per_snr):
            batch = sample_batch(system, channel, constellation, pp, pv,
                                 batch_size=batch_size, snr_db=snr, generator=gen)
            out = detector.detect(batch["r"], sigma_w2=batch["sigma_w2_block"])
            hard = out["hard_x"]
            ser_b = ((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum()
            P_true = batch["h_true"].shape[1]
            P_hat = out["h_hat"].shape[1]
            k = min(P_true, P_hat)
            h_hat = out["h_hat"][:, :k]
            h_true = batch["h_true"][:, :k]
            nmse_b = ((h_hat - h_true).abs() ** 2).sum() / (h_true.abs() ** 2).sum().clamp(min=1e-12)
            ell_hat = out["ell_hat"][:, :k]
            kap_hat = out["kappa_hat"][:, :k]
            d_rmse_b = ((ell_hat - batch["theta_true"][..., 0][:, :k]) ** 2).mean().sqrt()
            k_rmse_b = ((kap_hat - batch["theta_true"][..., 1][:, :k]) ** 2).mean().sqrt()
            ser += ser_b.item(); nmse += nmse_b.item()
            d_rmse += d_rmse_b.item(); k_rmse += k_rmse_b.item()
        results[snr] = {
            "ser": ser / n_batches_per_snr,
            "nmse_h": nmse / n_batches_per_snr,
            "delay_rmse": d_rmse / n_batches_per_snr,
            "doppler_rmse": k_rmse / n_batches_per_snr,
        }
    return results


@torch.no_grad()
def genie_mmse_sweep(
    config: ExperimentConfig,
    snr_dbs: list[float],
    n_batches_per_snr: int = 4,
    batch_size: int = 32,
    channel: Optional[UniformFractionalChannel] = None,
    seed: int = 42,
    K_cg: int = 30,
) -> dict[float, dict[str, float]]:
    """Genie-CSI CG-MMSE lower bound."""
    system = config.system()
    channel = channel if channel is not None else config.channel()
    constellation = config.constellation()
    pp, pv = config.pilots()
    gen = torch.Generator(device=config.device); gen.manual_seed(seed)
    results = {}
    for snr in snr_dbs:
        ser = 0.0
        for _ in range(n_batches_per_snr):
            batch = sample_batch(system, channel, constellation, pp, pv,
                                 batch_size=batch_size, snr_db=snr, generator=gen)
            op = FastAFDMOperator(system=system, ell=batch["theta_true"][..., 0],
                                  kappa=batch["theta_true"][..., 1], h=batch["h_true"])
            def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
            x_soft = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=K_cg)
            hard = (x_soft.unsqueeze(-1) - constellation.reshape(1, 1, -1)).abs().argmin(dim=-1)
            ser_b = ((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum()
            ser += ser_b.item()
        results[snr] = {"ser": ser / n_batches_per_snr}
    return results


def save_results_json(results: dict, path: str) -> None:
    """Save nested-dict results to JSON, converting Tensors as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def convert(o):
        if isinstance(o, dict):
            return {str(k): convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(v) for v in o]
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)

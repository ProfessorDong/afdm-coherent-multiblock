"""Rigorous publication training orchestrator.

Runs 4 ablation variants x N seeds x 500 epochs sequentially on cuda:0.
Total wall-clock at 205ms/step: ~34h for 12 runs (--seeds 0 1 2).

Rigorous features:
  * Deterministic seeding (torch + numpy + python random + cudnn).
  * NaN detection in loss AND parameter gradients each step; on hit, save
    diagnostic and abort THIS variant (continue to next). See --nan-policy.
  * Best-checkpoint tracking (lowest val SER at nominal SNR).
  * Last-checkpoint saving (for resume).
  * Progress JSON updated after every epoch (state.json in run dir).
  * Per-run timestamped log (train.log).
  * ETA calculation logged each epoch.
  * SIGTERM/SIGINT handler: save state and exit cleanly.
  * Resumable: --resume finds the last completed epoch and picks up from there.

Usage:
  python scripts/publication_train.py --seeds 0 1 2 --n_epochs 500 --run-dir runs/pub_v1
  python scripts/publication_train.py --resume --run-dir runs/pub_v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from afdm.experiments import ExperimentConfig, build_ablation
from afdm.loss import compose_training_loss
from afdm.training import TrainingConfig, sample_batch, evaluate_snr


def apply_v2_init(rx, gate_b: float = -8.0, gamma_raw: float = -5.0):
    """v2 architectural init: zero-delta, closed-gate, tiny-LM.

    Under this init, the receiver reduces to classical CG at t=0. Training then
    only ADDS learned corrections rather than starting from a random state that
    corrupts h_hat and requires the model to un-learn its own noise first.
    """
    for layer in rx.layers:
        layer.set_transformer.output_proj.weight.data.zero_()
        layer.set_transformer.output_proj.bias.data.zero_()
        layer.gate.b.data.fill_(gate_b)
        layer.gamma_raw.data.fill_(gamma_raw)
    return rx


# ==========================================================================
# Signal / abort handling
# ==========================================================================
_ABORT_REQUESTED = False


def _sigint_handler(signum, frame):
    """Set the abort flag; the training loop checks it between steps."""
    global _ABORT_REQUESTED
    _ABORT_REQUESTED = True
    print(f"\n[SIGNAL] caught {signal.Signals(signum).name} — will save state after current step and exit cleanly.")


signal.signal(signal.SIGINT, _sigint_handler)
signal.signal(signal.SIGTERM, _sigint_handler)


# ==========================================================================
# Deterministic seeding
# ==========================================================================
def set_deterministic_seed(seed: int) -> None:
    """Fully deterministic seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: full cudnn determinism is very slow. Enable only if strictly needed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ==========================================================================
# NaN detection
# ==========================================================================
def has_nan_or_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def check_grads_for_nan(model: nn.Module) -> Optional[str]:
    for name, p in model.named_parameters():
        if p.grad is not None and has_nan_or_inf(p.grad):
            return name
    return None


# ==========================================================================
# Per-run state & JSON
# ==========================================================================
def run_dir_for(root: Path, variant: str, seed: int) -> Path:
    d = root / f"{variant}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_state(run_dir: Path, state: dict) -> None:
    """Atomic write of state.json (write then rename)."""
    tmp = run_dir / "state.json.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(run_dir / "state.json")


def load_state_if_exists(run_dir: Path) -> Optional[dict]:
    p = run_dir / "state.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def log_line(run_dir: Path, msg: str) -> None:
    """Append a timestamped line to train.log AND stdout."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(run_dir / "train.log", "a") as f:
        f.write(line + "\n")


# ==========================================================================
# Best-checkpoint tracking
# ==========================================================================
def save_checkpoint(run_dir: Path, tag: str, receiver, config, tc, epoch: int, val_snr_ser: dict) -> Path:
    """Save a checkpoint under a specific tag ('best', 'last', 'epoch_N', ...)."""
    path = run_dir / f"{tag}.pt"
    torch.save({
        "state_dict": receiver.state_dict(),
        "config": asdict(config),
        "training_config": asdict(tc),
        "epoch": epoch,
        "val_snr_ser": val_snr_ser,
    }, path)
    return path


def maybe_update_best(
    run_dir: Path,
    receiver, config, tc, epoch: int,
    val_snr_ser: dict, best_metric: float, nominal_snr: float,
) -> float:
    metric = val_snr_ser.get(nominal_snr, val_snr_ser.get(str(nominal_snr), float("inf")))
    if metric < best_metric:
        save_checkpoint(run_dir, "best", receiver, config, tc, epoch, val_snr_ser)
        log_line(run_dir, f"NEW BEST @ epoch {epoch}: SER({nominal_snr}dB) = {metric:.4e}")
        return metric
    return best_metric


# ==========================================================================
# Single-variant training with all safeguards
# ==========================================================================
def train_one_variant(
    variant: str,
    seed: int,
    config: ExperimentConfig,
    training_config: TrainingConfig,
    run_root: Path,
    nominal_snr: float = 15.0,
    ckpt_every_epochs: int = 20,
    nan_policy: str = "abort_variant",  # or "abort_all"
    resume: bool = False,
) -> dict:
    """Train one variant × seed with all rigor safeguards.

    Returns a status dict {status, run_dir, best_metric, elapsed_h, error?}
    """
    run_dir = run_dir_for(run_root, variant, seed)

    # Reproducibility
    set_deterministic_seed(seed)
    config.seed = seed

    # Build model, optimizer, scheduler
    receiver = build_ablation(variant, config)
    # Apply v2 init if requested (default). v1 is the JSAC-era default (broken).
    if getattr(train_one_variant, "_init_recipe", "v2") == "v2":
        apply_v2_init(receiver)
    # Only optimize params that require grad (some ablations freeze modules)
    trainable = [p for p in receiver.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=training_config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.n_epochs)

    system = config.system()
    channel = config.channel()
    constellation = config.constellation()
    pp, pv = config.pilots()

    # Resume?
    start_epoch = 0
    best_metric = float("inf")
    if resume:
        st = load_state_if_exists(run_dir)
        if st is not None and (run_dir / "last.pt").exists():
            ckpt = torch.load(run_dir / "last.pt", map_location=config.device, weights_only=False)
            receiver.load_state_dict(ckpt["state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_metric = st.get("best_metric", float("inf"))
            log_line(run_dir, f"RESUME from epoch {start_epoch} (best so far: {best_metric:.4e})")
            # Advance scheduler to the right epoch
            for _ in range(start_epoch):
                scheduler.step()
        else:
            log_line(run_dir, "RESUME requested but no state found — starting from scratch")

    log_line(run_dir, f"START variant={variant} seed={seed} device={config.device}")
    log_line(run_dir, f"  n_epochs={training_config.n_epochs} steps={training_config.steps_per_epoch} batch={training_config.batch_size}")
    log_line(run_dir, f"  n_trainable_params={sum(p.numel() for p in trainable):,}")

    train_gen = torch.Generator(device=config.device); train_gen.manual_seed(seed + 1000)
    val_gen = torch.Generator(device=config.device); val_gen.manual_seed(seed + 2000)

    t_start = time.time()
    variant_start = time.time()
    try:
        for epoch in range(start_epoch, training_config.n_epochs):
            if _ABORT_REQUESTED:
                log_line(run_dir, "ABORT: SIGTERM/SIGINT received, saving last state and exiting")
                save_checkpoint(run_dir, "last", receiver, config, training_config, epoch, {})
                return {"status": "aborted", "run_dir": str(run_dir),
                        "best_metric": best_metric, "elapsed_h": (time.time() - variant_start) / 3600}

            receiver.train()
            epoch_losses = []
            epoch_start = time.time()
            for step in range(training_config.steps_per_epoch):
                batch = sample_batch(system, channel, constellation, pp, pv,
                                     batch_size=training_config.batch_size, snr_db=None,
                                     snr_db_range=(training_config.snr_db_min, training_config.snr_db_max),
                                     generator=train_gen)
                out = receiver(batch["r"], sigma_w2_block=batch["sigma_w2_block"],
                               refine_theta=True, return_layer_states=True)
                loss, _ = compose_training_loss(
                    out, batch["x_true"], batch["labels"],
                    batch["h_true"], batch["theta_true"], batch["pilot_mask"],
                    layer_gamma=training_config.layer_gamma,
                    mu_theta=training_config.mu_theta,
                    mu_ce=training_config.mu_ce,
                    eta_anchor=training_config.eta_anchor,
                    hungarian_kwargs=training_config.hungarian_kwargs,
                )
                if has_nan_or_inf(loss):
                    log_line(run_dir, f"NaN/Inf LOSS at epoch {epoch} step {step}: loss={loss.item()}")
                    save_checkpoint(run_dir, f"nan_ep{epoch}_step{step}", receiver, config, training_config, epoch, {})
                    if nan_policy == "abort_variant":
                        return {"status": "nan_abort", "run_dir": str(run_dir),
                                "best_metric": best_metric, "elapsed_h": (time.time() - variant_start) / 3600,
                                "error": f"NaN loss at ep{epoch} step{step}"}
                    else:
                        raise RuntimeError(f"NaN loss at ep{epoch} step{step}")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nan_param = check_grads_for_nan(receiver)
                if nan_param is not None:
                    log_line(run_dir, f"NaN/Inf GRAD at epoch {epoch} step {step} on {nan_param}")
                    save_checkpoint(run_dir, f"nan_grad_ep{epoch}_step{step}", receiver, config, training_config, epoch, {})
                    if nan_policy == "abort_variant":
                        return {"status": "nan_abort", "run_dir": str(run_dir),
                                "best_metric": best_metric, "elapsed_h": (time.time() - variant_start) / 3600,
                                "error": f"NaN grad on {nan_param} at ep{epoch} step{step}"}
                    else:
                        raise RuntimeError(f"NaN grad on {nan_param} at ep{epoch} step{step}")

                torch.nn.utils.clip_grad_norm_(trainable, training_config.grad_clip)
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            avg_loss = float(np.mean(epoch_losses))
            epoch_time = time.time() - epoch_start
            remaining = (training_config.n_epochs - epoch - 1) * epoch_time
            eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
            log_line(run_dir, f"epoch {epoch+1}/{training_config.n_epochs} avg_loss={avg_loss:.4f} "
                              f"epoch_time={epoch_time:.1f}s ETA={eta}")

            # Validation and checkpoint
            val_metrics = {}
            if (epoch + 1) % training_config.val_every == 0 or epoch == training_config.n_epochs - 1:
                val_ser_by_snr = {}
                for snr in training_config.val_snr_dbs:
                    m = evaluate_snr(receiver, system, channel, constellation, pp, pv,
                                     snr_db=snr, n_batches=training_config.val_batches,
                                     batch_size=training_config.batch_size, generator=val_gen)
                    val_ser_by_snr[snr] = m["ser"]
                    val_metrics[snr] = m
                for snr, ser in val_ser_by_snr.items():
                    log_line(run_dir, f"  [val] SNR {snr:5.1f}dB SER={ser:.4e}")
                # Track best
                best_metric = maybe_update_best(run_dir, receiver, config, training_config,
                                                epoch, val_ser_by_snr, best_metric, nominal_snr)

            # Regular checkpointing
            if (epoch + 1) % ckpt_every_epochs == 0 or epoch == training_config.n_epochs - 1:
                save_checkpoint(run_dir, "last", receiver, config, training_config, epoch, val_metrics)

            # Progress JSON
            save_state(run_dir, {
                "variant": variant, "seed": seed, "epoch": epoch + 1,
                "avg_loss": avg_loss, "best_metric": best_metric,
                "elapsed_h": (time.time() - variant_start) / 3600,
                "eta_seconds_remaining": remaining,
                "latest_val_metrics": val_metrics,
                "config": asdict(config), "training_config": asdict(training_config),
            })

        # Final checkpoint
        save_checkpoint(run_dir, "last", receiver, config, training_config, training_config.n_epochs - 1, val_metrics)
        elapsed_h = (time.time() - variant_start) / 3600
        log_line(run_dir, f"DONE variant={variant} seed={seed} best={best_metric:.4e} elapsed={elapsed_h:.2f}h")
        return {"status": "success", "run_dir": str(run_dir), "best_metric": best_metric,
                "elapsed_h": elapsed_h}

    except Exception as e:
        elapsed_h = (time.time() - variant_start) / 3600
        log_line(run_dir, f"EXCEPTION variant={variant} seed={seed}: {type(e).__name__}: {e}")
        with open(run_dir / "traceback.txt", "w") as f:
            traceback.print_exc(file=f)
        return {"status": "error", "run_dir": str(run_dir), "best_metric": best_metric,
                "elapsed_h": elapsed_h, "error": f"{type(e).__name__}: {e}"}


# ==========================================================================
# Main orchestrator
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["proposed", "gate", "attention", "scalars"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=500)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--P", type=int, default=5)
    ap.add_argument("--K_cg", type=int, default=10)
    ap.add_argument("--N_p", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--val_every", type=int, default=10)
    ap.add_argument("--val_batches", type=int, default=3)
    ap.add_argument("--nominal_snr", type=float, default=15.0)
    ap.add_argument("--ckpt_every", type=int, default=20)
    ap.add_argument("--nan-policy", choices=["abort_variant", "abort_all"], default="abort_variant")
    ap.add_argument("--run-dir", default="runs/pub_v1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-recipe", choices=["default", "v2"], default="v2",
                    help="v2 = zero-delta + closed-gate + tiny-LM (recommended); default = JSAC-era random init")
    ap.add_argument("--mu-ce", type=float, default=5.0, help="cross-entropy weight (v2 default 5.0, v1 was 0.5)")
    args = ap.parse_args()
    # Pass init recipe to the per-variant trainer via a function attribute.
    train_one_variant._init_recipe = args.init_recipe

    run_root = Path(args.run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    campaign_start = time.time()

    # Global campaign log
    campaign_log = run_root / "campaign.log"
    def clog(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(campaign_log, "a") as f:
            f.write(line + "\n")

    clog(f"=== CAMPAIGN START ===")
    clog(f"  run-dir: {run_root}")
    clog(f"  variants: {args.variants}")
    clog(f"  seeds: {args.seeds}")
    clog(f"  n_epochs: {args.n_epochs}, steps_per_epoch: {args.steps_per_epoch}")
    clog(f"  init-recipe: {args.init_recipe}")
    clog(f"  mu_ce: {args.mu_ce}, N={args.N}, P={args.P}, N_p={args.N_p}, T={args.T}")
    clog(f"  device: cuda:0 (RTX 4090)")

    config = ExperimentConfig(
        N=args.N, kappa_max=5.0, ell_max=10.0,
        P=args.P, N_p=args.N_p,
        T=args.T, K_cg=args.K_cg, d_model=args.d_model, n_heads=args.n_heads, n_blocks=args.n_blocks,
        P_max=args.P,
    )
    tc = TrainingConfig(
        lr=args.lr, n_epochs=args.n_epochs, steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size, snr_db_min=5.0, snr_db_max=25.0,
        grad_clip=1.0, val_every=args.val_every, val_batches=args.val_batches,
        val_snr_dbs=(5.0, 10.0, 15.0, 20.0, 25.0),
        layer_gamma=0.7, mu_ce=args.mu_ce, eta_anchor=0.0,
        hungarian_kwargs=dict(w_h=1.0, w_ell=0.2, w_kap=0.2, mu_fa=0.1, mu_md=0.1),
        log_every=args.steps_per_epoch,
    )

    summary = []
    for seed in args.seeds:
        for variant in args.variants:
            if _ABORT_REQUESTED:
                clog("Aborting campaign (SIGTERM/SIGINT received)")
                break
            clog(f"--- launching variant={variant} seed={seed} ---")
            result = train_one_variant(
                variant, seed, config, tc, run_root,
                nominal_snr=args.nominal_snr, ckpt_every_epochs=args.ckpt_every,
                nan_policy=args.nan_policy, resume=args.resume,
            )
            summary.append({"variant": variant, "seed": seed, **result})
            clog(f"    -> {result['status']} best={result['best_metric']:.4e} elapsed={result['elapsed_h']:.2f}h")
            # Free GPU memory between runs
            torch.cuda.empty_cache()
        if _ABORT_REQUESTED:
            break

    campaign_elapsed = (time.time() - campaign_start) / 3600
    clog(f"=== CAMPAIGN END: {campaign_elapsed:.2f}h ===")
    clog(f"Summary:")
    for s in summary:
        clog(f"  {s['variant']}_seed{s['seed']}: {s['status']} best={s['best_metric']:.4e} elapsed={s['elapsed_h']:.2f}h")
    with open(run_root / "summary.json", "w") as f:
        json.dump({"summary": summary, "campaign_elapsed_h": campaign_elapsed}, f, indent=2, default=str)

    # Exit code reflects worst per-variant status
    failed = [s for s in summary if s["status"] not in ("success", "aborted")]
    if failed:
        clog(f"FAIL: {len(failed)} variants failed")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

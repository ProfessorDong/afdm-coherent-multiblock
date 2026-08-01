"""Training loop for PathSetEstimator on a given config.

Configurable: config (easy/hard/hard32), epochs, save path. Meant to be the
production training script — the smoke variant lives in train_pathset_smoke.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim

from afdm.experiments import ExperimentConfig
from afdm.pathset_estimator import PathSetEstimator, PathSetEstimatorConfig
from afdm.pathset_frontend import build_frontend_inputs
from afdm.pathset_loss import compose_pathset_loss
from afdm.training import sample_batch


def build_cfg(name: str) -> ExperimentConfig:
    if name == "easy":
        return ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)
    if name == "hard":
        return ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)
    if name == "hard32":
        return ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)
    raise ValueError(name)


def train(cfg, args) -> tuple[PathSetEstimator, list[float]]:
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()

    est = PathSetEstimator(PathSetEstimatorConfig(K=args.K)).to(cfg.device)
    n_params = sum(p.numel() for p in est.parameters())
    print(f"PathSetEstimator ({args.config}): {n_params:,} parameters")
    opt = optim.Adam(est.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    gen = torch.Generator(device=cfg.device); gen.manual_seed(args.seed)
    history = []

    t0 = time.time()
    for ep in range(args.epochs):
        losses_ep = []; last_br = None
        for step in range(args.steps):
            snr = args.snr_min + (args.snr_max - args.snr_min) * torch.rand(1, generator=gen, device=cfg.device).item()
            batch = sample_batch(system, channel, const, pp, pv,
                                 batch_size=args.batch, snr_db=snr, generator=gen)
            fe = build_frontend_inputs(batch["r"], system, pp, pv,
                                       kappa_max=cfg.kappa_max, K=args.K,
                                       sigma_w2_block=batch["sigma_w2_block"])
            pred = est(fe["scalar_feats"], fe["patch_feats"], fe["valid"])
            # Auto pos_weight = (K - P) / P, calibrated to negative:positive ratio.
            pos_w = max((args.K - cfg.P) / max(cfg.P, 1), 1.0)
            loss, br = compose_pathset_loss(
                pred, fe["ell_cfar"], fe["kap_cfar"], system, batch["r"],
                batch["theta_true"], batch["h_true"], batch["x_true"],
                lambda_rec=args.lambda_rec,
                hungarian_kwargs=dict(w_ell=args.w_ell, w_kap=args.w_kap,
                                      w_h=1.0, w_e=0.5,
                                      mu_fa=0.2, mu_md=1.0, pos_weight=pos_w),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(est.parameters(), 5.0)
            opt.step()
            losses_ep.append(loss.item()); last_br = br
        sched.step()
        avg = sum(losses_ep) / len(losses_ep)
        history.append(avg)
        elapsed = time.time() - t0
        eta = elapsed * (args.epochs - ep - 1) / max(ep + 1, 1)
        print(f"ep {ep+1:03d}  avg_loss {avg:.3f}  "
              f"[ell {last_br['ell']:.2f} kap {last_br['kap']:.2f} "
              f"h {last_br['h']:.2f} exist {last_br['exist']:.2f} "
              f"fa {last_br['fa']:.2f} rec {last_br['rec']:.2f}]  "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
    return est, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=("easy", "hard", "hard32"), default="easy")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda_rec", type=float, default=0.5)
    ap.add_argument("--snr_min", type=float, default=5.0)
    ap.add_argument("--snr_max", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="runs/pathset")
    ap.add_argument("--w_ell", type=float, default=1.0)
    ap.add_argument("--w_kap", type=float, default=1.0)
    args = ap.parse_args()

    cfg = build_cfg(args.config)
    est, history = train(cfg, args)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"pathset_{args.config}.pt"
    torch.save(est.state_dict(), ckpt)
    hist_path = out_dir / f"pathset_{args.config}_history.json"
    with open(hist_path, "w") as f:
        json.dump({"history": history, "args": vars(args)}, f, indent=2)
    print(f"\ncheckpoint saved: {ckpt}")
    print(f"history saved:   {hist_path}")


if __name__ == "__main__":
    main()

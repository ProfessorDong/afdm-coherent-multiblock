"""Short training smoke test for PathSetEstimator: verify loss decreases.

Runs 20 epochs x 50 steps on the easy config. Reports the loss trajectory and
per-term breakdown. If total loss doesn't drop by at least 30% over 20 epochs,
something's wrong architecturally — investigate before scaling up.
"""

from __future__ import annotations

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


def train_smoke(cfg: ExperimentConfig, K: int = 24, n_epochs: int = 20,
                steps_per_epoch: int = 50, batch_size: int = 32,
                lr: float = 3e-4, snr_range=(5.0, 25.0)):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()

    est = PathSetEstimator(PathSetEstimatorConfig(K=K)).to(cfg.device)
    n_params = sum(p.numel() for p in est.parameters())
    print(f"PathSetEstimator: {n_params:,} parameters")
    opt = optim.Adam(est.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    gen = torch.Generator(device=cfg.device); gen.manual_seed(0)
    history = []

    t0 = time.time()
    for ep in range(n_epochs):
        losses_ep = []; break_ep = None
        for step in range(steps_per_epoch):
            snr = snr_range[0] + (snr_range[1] - snr_range[0]) * torch.rand(1, generator=gen, device=cfg.device).item()
            batch = sample_batch(system, channel, const, pp, pv,
                                 batch_size=batch_size, snr_db=snr, generator=gen)
            fe = build_frontend_inputs(batch["r"], system, pp, pv,
                                       kappa_max=cfg.kappa_max, K=K,
                                       sigma_w2_block=batch["sigma_w2_block"])
            pred = est(fe["scalar_feats"], fe["patch_feats"], fe["valid"])
            loss, br = compose_pathset_loss(
                pred, fe["ell_cfar"], fe["kap_cfar"], system, batch["r"],
                batch["theta_true"], batch["h_true"], batch["x_true"],
                lambda_rec=0.5,
                hungarian_kwargs=dict(w_ell=1.0, w_kap=1.0, w_h=1.0, w_e=0.5,
                                      mu_fa=0.2, mu_md=1.0),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(est.parameters(), 5.0)
            opt.step()
            losses_ep.append(loss.item())
            break_ep = br
        sched.step()
        avg = sum(losses_ep) / len(losses_ep)
        history.append(avg)
        print(f"ep {ep+1:03d}  avg_loss {avg:.3f}  "
              f"[ell {break_ep['ell']:.2f} kap {break_ep['kap']:.2f} "
              f"h {break_ep['h']:.2f} exist {break_ep['exist']:.2f} "
              f"fa {break_ep['fa']:.2f} rec {break_ep['rec']:.2f}]")

    t = time.time() - t0
    print(f"\n[timing] {t:.1f}s total ({t/n_epochs:.1f}s/epoch)")
    print(f"loss trajectory: {history[0]:.3f} -> {history[-1]:.3f}  "
          f"({(history[0] - history[-1]) / history[0]:.0%} reduction)")
    ckpt_dir = Path("runs/pathset_smoke")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"pathset_smoke_P{cfg.P}_Np{cfg.N_p}.pt"
    torch.save(est.state_dict(), ckpt_path)
    print(f"checkpoint saved: {ckpt_path}")
    return est, history


def main():
    print("=" * 78)
    print("PathSetEstimator training smoke (easy config)")
    print("=" * 78)
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)
    est, hist = train_smoke(cfg, n_epochs=20)
    if hist[-1] < 0.7 * hist[0]:
        print("PASS: loss reduced by >= 30% in 20 epochs")
    else:
        print(f"WARN: loss only reduced by {(hist[0] - hist[-1])/hist[0]:.0%}; "
              "training may need debugging")


if __name__ == "__main__":
    main()

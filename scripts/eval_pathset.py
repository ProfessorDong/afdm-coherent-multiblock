"""Evaluate a trained PathSetEstimator end-to-end.

Reports:
  * SER at multiple SNRs, with and without Stage 3 (LM polish) and Stage 4 (DA-SBL)
  * Path recall/precision (matched vs true, tolerance 0.75 x 0.75)
  * Delay/Doppler RMSE on matched paths
  * Gain NMSE on matched paths
  * Comparison against R1 (genie), R2 (true pos + LS h), R4 (classical CG)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from scipy.optimize import linear_sum_assignment

from afdm.classical import ClassicalCGDetector
from afdm.experiments import ExperimentConfig
from afdm.pathset_estimator import PathSetEstimator, PathSetEstimatorConfig
from afdm.pathset_receiver import PathSetReceiver, PathSetReceiverConfig
from afdm.training import sample_batch

from oracle_ladder import rung1_genie, rung2_trueth_lsh


TOL_ELL = 0.75  # match tolerance for recall/precision (samples)
TOL_KAP = 0.75


@torch.no_grad()
def path_metrics(ell_hat, kap_hat, h_hat, active_mask,
                 ell_true, kap_true, h_true):
    """Per-batch: recall, precision, delay RMSE, Doppler RMSE, gain NMSE."""
    B = ell_hat.shape[0]
    tot_tp = 0; tot_fp = 0; tot_fn = 0
    de_sq = 0.0; dk_sq = 0.0; nmse_num = 0.0; nmse_den = 0.0
    n_matched = 0
    for b in range(B):
        active = active_mask[b].bool()
        n_act = int(active.sum())
        n_true = ell_true.shape[1]
        if n_act == 0:
            tot_fn += n_true
            continue
        e_a = ell_hat[b, active]; k_a = kap_hat[b, active]; h_a = h_hat[b, active]
        cost = ((e_a.unsqueeze(-1) - ell_true[b].unsqueeze(0)) ** 2 +
                (k_a.unsqueeze(-1) - kap_true[b].unsqueeze(0)) ** 2)
        row, col = linear_sum_assignment(cost.cpu().numpy())
        matched_pairs = []
        for r, c in zip(row, col):
            if (float(abs(e_a[r] - ell_true[b, c])) <= TOL_ELL and
                float(abs(k_a[r] - kap_true[b, c])) <= TOL_KAP):
                matched_pairs.append((r, c))
        tp = len(matched_pairs)
        tot_tp += tp
        tot_fp += n_act - tp
        tot_fn += n_true - tp
        for r, c in matched_pairs:
            de_sq += float((e_a[r] - ell_true[b, c]) ** 2)
            dk_sq += float((k_a[r] - kap_true[b, c]) ** 2)
            nmse_num += float((h_a[r] - h_true[b, c]).abs() ** 2)
            nmse_den += float(h_true[b, c].abs() ** 2)
        n_matched += tp
    recall = tot_tp / max(tot_tp + tot_fn, 1)
    precision = tot_tp / max(tot_tp + tot_fp, 1)
    d_rmse = (de_sq / max(n_matched, 1)) ** 0.5 if n_matched else float("nan")
    k_rmse = (dk_sq / max(n_matched, 1)) ** 0.5 if n_matched else float("nan")
    nmse = nmse_num / max(nmse_den, 1e-12) if nmse_den > 0 else float("nan")
    return recall, precision, d_rmse, k_rmse, nmse


@torch.no_grad()
def evaluate_config(name: str, cfg: ExperimentConfig, receiver_variants: dict,
                    snrs=(5.0, 15.0, 25.0), n_batches=4, batch_size=32, seed=42):
    print()
    print("=" * 78)
    print(f"CONFIG: {name}")
    print("=" * 78)

    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    classical = ClassicalCGDetector(
        system=system, support_recovery=cfg.support_recovery(),
        constellation=const, pilot_positions=pp, pilot_values=pv,
        T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )

    print(f"{'SNR':<6s}  {'R1':>8s}  {'R2':>8s}  {'R4':>8s}  ", end="")
    for vname in receiver_variants: print(f"{vname:>10s}", end="  ")
    print(f"{'recall':>7s}  {'prec':>6s}  {'d_rmse':>7s}  {'k_rmse':>7s}  {'nmse_h':>8s}")

    for snr in snrs:
        gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
        ser_r1 = ser_r2 = ser_r4 = 0.0
        ser_rx = {k: 0.0 for k in receiver_variants}
        metrics = {k: [0.0]*5 for k in receiver_variants}
        for _ in range(n_batches):
            batch = sample_batch(system, channel, const, pp, pv,
                                 batch_size=batch_size, snr_db=snr, generator=gen)
            mask = batch["pilot_mask"]
            def ser_of(hard): return float(((hard != batch["labels"]) * mask).float().sum() / mask.float().sum())
            hard1 = rung1_genie(batch, system, const)
            hard2, _ = rung2_trueth_lsh(batch, system, const, pp, pv)
            out4 = classical.detect(batch["r"], sigma_w2=batch["sigma_w2_block"])
            ser_r1 += ser_of(hard1); ser_r2 += ser_of(hard2); ser_r4 += ser_of(out4["hard_x"])

            for vname, rx in receiver_variants.items():
                out = rx(batch["r"], sigma_w2_block=batch["sigma_w2_block"])
                ser_rx[vname] += ser_of(out["hard_x"])
                # Path metrics only tracked for the default (full) variant, if present.
                if vname == list(receiver_variants.keys())[-1]:  # last = strongest variant
                    m = path_metrics(out["ell_hat"], out["kappa_hat"], out["h_hat"],
                                     out["active_mask"],
                                     batch["theta_true"][..., 0], batch["theta_true"][..., 1],
                                     batch["h_true"])
                    for i in range(5):
                        metrics[vname][i] += m[i]

        for k in ser_rx: ser_rx[k] /= n_batches
        ser_r1 /= n_batches; ser_r2 /= n_batches; ser_r4 /= n_batches
        vlast = list(receiver_variants.keys())[-1]
        m = [x / n_batches for x in metrics[vlast]]
        line = f"{snr:>4.1f}dB  {ser_r1:>8.2e}  {ser_r2:>8.2e}  {ser_r4:>8.2e}  "
        for vname in receiver_variants:
            line += f"{ser_rx[vname]:>10.2e}  "
        line += f"{m[0]:>7.1%}  {m[1]:>6.1%}  {m[2]:>7.3f}  {m[3]:>7.3f}  {m[4]:>8.2e}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=False,
                    help="Path to trained PathSetEstimator checkpoint")
    ap.add_argument("--config", choices=("easy", "hard", "hard32"), default="easy")
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    if args.config == "easy":
        cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)
    elif args.config == "hard":
        cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)
    else:
        cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)

    pp, pv = cfg.pilots()
    const = cfg.constellation()

    # Build the estimator, load checkpoint if provided, else use random init.
    est_cfg = PathSetEstimatorConfig(K=24)

    def _new_rx(lm=0, sbl=0):
        rcfg = PathSetReceiverConfig(K=24, est_cfg=est_cfg,
                                     lm_polish_iters=lm, sbl_iters=sbl)
        rx = PathSetReceiver(rcfg, cfg.system(), const, pp, pv).to(cfg.device)
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location=cfg.device, weights_only=True)
            rx.estimator.load_state_dict(state)
        rx.eval()
        return rx

    variants = {
        "NN-only":      _new_rx(lm=0, sbl=0),
        "NN+LM":        _new_rx(lm=1, sbl=0),
        "NN+LM+SBL":    _new_rx(lm=1, sbl=3),
    }

    evaluate_config(f"{args.config.upper()}", cfg, variants,
                    n_batches=args.n_batches, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

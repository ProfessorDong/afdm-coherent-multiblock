"""Diagnose existence-classifier calibration on a trained PathSetEstimator.

Reports for each candidate:
  * distance to nearest true path (tolerance 0.75 x 0.75 defines "matched")
  * sigmoid(exist_logit) probability

Then: histogram of probs for matched vs unmatched candidates. If they overlap
heavily, the classifier can't separate; if the true-path distribution has a
mode far below 0.5, we're just too conservative.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from afdm.experiments import ExperimentConfig
from afdm.pathset_estimator import PathSetEstimator, PathSetEstimatorConfig
from afdm.pathset_frontend import build_frontend_inputs
from afdm.training import sample_batch


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--config", choices=("easy", "hard"), default="easy")
    ap.add_argument("--snr", type=float, default=15.0)
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    if args.config == "easy":
        cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32)
    else:
        cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16)

    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    est = PathSetEstimator(PathSetEstimatorConfig(K=24)).to(cfg.device)
    est.load_state_dict(torch.load(args.checkpoint, map_location=cfg.device, weights_only=True))
    est.eval()

    tol_ell, tol_kap = 0.75, 0.75
    matched_probs = []
    unmatched_probs = []
    matched_ell_err = []
    matched_kap_err = []
    matched_h_err = []
    matched_h_true_mag = []

    gen = torch.Generator(device=cfg.device); gen.manual_seed(42)
    for _ in range(args.n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=args.batch_size, snr_db=args.snr, generator=gen)
        fe = build_frontend_inputs(batch["r"], system, pp, pv, kappa_max=5.0,
                                   K=24, sigma_w2_block=batch["sigma_w2_block"])
        with torch.no_grad():
            pred = est(fe["scalar_feats"], fe["patch_feats"], fe["valid"])
        exist_p = torch.sigmoid(pred["exist_logit"])
        ell_hat = fe["ell_cfar"] + pred["delta_ell"]
        kap_hat = fe["kap_cfar"] + pred["delta_kappa"]

        B, K = pred["exist_logit"].shape
        for b in range(B):
            valid_k = fe["valid"][b]
            for k in range(K):
                if not valid_k[k]:
                    continue
                # Distance to nearest true path
                d_ell = (ell_hat[b, k] - batch["theta_true"][b, :, 0]).abs()
                d_kap = (kap_hat[b, k] - batch["theta_true"][b, :, 1]).abs()
                is_match = (d_ell <= tol_ell) & (d_kap <= tol_kap)
                if is_match.any():
                    matched_probs.append(float(exist_p[b, k]))
                    # Take the closest true path
                    dist = d_ell**2 + d_kap**2
                    i = int(dist.argmin())
                    matched_ell_err.append(float(d_ell[i]))
                    matched_kap_err.append(float(d_kap[i]))
                    matched_h_err.append(float((pred["h"][b, k] - batch["h_true"][b, i]).abs()))
                    matched_h_true_mag.append(float(batch["h_true"][b, i].abs()))
                else:
                    unmatched_probs.append(float(exist_p[b, k]))

    print(f"\n=== Existence probability distribution ===")
    print(f"Matched candidates ({len(matched_probs)} total):")
    if matched_probs:
        arr = np.array(matched_probs)
        print(f"  mean {arr.mean():.3f}  median {np.median(arr):.3f}  "
              f"25% {np.percentile(arr, 25):.3f}  75% {np.percentile(arr, 75):.3f}")
        print(f"  fraction > 0.5: {(arr > 0.5).mean():.1%}")
        print(f"  fraction > 0.3: {(arr > 0.3).mean():.1%}")
        print(f"  fraction > 0.1: {(arr > 0.1).mean():.1%}")
    print(f"Unmatched candidates ({len(unmatched_probs)} total):")
    if unmatched_probs:
        arr = np.array(unmatched_probs)
        print(f"  mean {arr.mean():.3f}  median {np.median(arr):.3f}  "
              f"25% {np.percentile(arr, 25):.3f}  75% {np.percentile(arr, 75):.3f}")
        print(f"  fraction > 0.5: {(arr > 0.5).mean():.1%}")
        print(f"  fraction > 0.3: {(arr > 0.3).mean():.1%}")

    print(f"\n=== Matched-pair errors ===")
    if matched_ell_err:
        arr = np.array(matched_ell_err)
        print(f"delay err   : mean {arr.mean():.3f}  RMSE {(arr**2).mean()**0.5:.3f}  max {arr.max():.3f}")
    if matched_kap_err:
        arr = np.array(matched_kap_err)
        print(f"doppler err : mean {arr.mean():.3f}  RMSE {(arr**2).mean()**0.5:.3f}  max {arr.max():.3f}")
    if matched_h_err:
        arr_e = np.array(matched_h_err); arr_t = np.array(matched_h_true_mag)
        print(f"gain err    : mean {arr_e.mean():.3f}  NMSE {(arr_e**2).sum() / (arr_t**2).sum():.3e}")

    # Ideal threshold via ROC-like split
    if matched_probs and unmatched_probs:
        m = np.array(matched_probs); u = np.array(unmatched_probs)
        best_thr = 0.5; best_f1 = 0
        for thr in np.linspace(0.05, 0.95, 19):
            tp = (m > thr).sum(); fp = (u > thr).sum(); fn = (m <= thr).sum()
            f1 = 2*tp / max(2*tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1 = f1; best_thr = thr
        print(f"\nBest F1 threshold: {best_thr:.2f} (F1 = {best_f1:.3f})")


if __name__ == "__main__":
    main()

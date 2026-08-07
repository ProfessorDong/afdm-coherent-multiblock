"""Per-path RMSE of the coarse acquisition front end (peak selection + 2 Newton
steps) from pilot-only initialization, measured separately at Easy and Hard.

Backs the Section III claim locating the coarse-support RMSE relative to the
basin of attraction of Fig. 3. Estimated paths are matched to true paths by
optimal assignment on Euclidean (ell, kappa) distance over the P true paths;
RMSE is reported per coordinate and jointly over matched pairs.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock

SNR, SEEDS, NB, BS = 15.0, 5, 8, 32
out = {"snr_db": SNR, "N_seeds": SEEDS, "N_batches": NB, "batch_size": BS, "results": {}}

for name, cfg in [("Easy (P=3,N_p=32)", ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=3, N_p=32, P_max=6)),
                  ("Hard (P=5,N_p=16)", ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=5, N_p=16, P_max=8))]:
    system, ch, const = cfg.system(), cfg.channel(), cfg.constellation()
    sr = cfg.support_recovery()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=1,
                                      constellation=const, device=cfg.device, seed=42)
    per_seed_e, per_seed_k, per_seed_j, per_seed_m, per_seed_det = [], [], [], [], []
    for k in range(SEEDS):
        g = torch.Generator(device=cfg.device); g.manual_seed(k * 137 + 42)
        de, dk, dem, dkm, nmatch = [], [], [], [], []
        for _ in range(NB):
            batch = sample_multiblock(system, ch, const, pp, pv, batch_size=BS,
                                      snr_db=SNR, generator=g)
            xp = torch.zeros(BS, cfg.N, dtype=batch.r.dtype, device=cfg.device)
            xp[:, pp[0]] = pv[0].unsqueeze(0)
            s0 = system.idaft(xp)
            with torch.no_grad():
                eh, kh, _ = sr(batch.r[:, 0, :], s0)
            et = batch.theta_true[..., 0]; kt = batch.theta_true[..., 1]   # (BS, P)
            eh_n, kh_n = eh.cpu().numpy(), kh.cpu().numpy()
            et_n, kt_n = et.cpu().numpy(), kt.cpu().numpy()
            for i in range(BS):
                C = np.hypot(et_n[i][:, None] - eh_n[i][None, :],
                             kt_n[i][:, None] - kh_n[i][None, :])
                r_, c_ = linear_sum_assignment(C)
                ee = et_n[i][r_] - eh_n[i][c_]; kk = kt_n[i][r_] - kh_n[i][c_]
                de.extend(ee); dk.extend(kk)
                m = (np.abs(ee) <= 0.75) & (np.abs(kk) <= 0.75)   # "detected" tolerance
                dem.extend(ee[m]); dkm.extend(kk[m]); nmatch.append(m.mean())
        de, dk = np.array(de), np.array(dk)
        dem, dkm = np.array(dem), np.array(dkm)
        per_seed_m.append(float(np.sqrt(((dem**2+dkm**2)/2).mean())) if len(dem) else float('nan'))
        per_seed_det.append(float(np.mean(nmatch)))
        per_seed_e.append(float(np.sqrt((de ** 2).mean())))
        per_seed_k.append(float(np.sqrt((dk ** 2).mean())))
        per_seed_j.append(float(np.sqrt((de ** 2 + dk ** 2).mean())))
    f = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v))}
    out["results"][name] = {"rmse_ell": f(per_seed_e), "rmse_kappa": f(per_seed_k),
                            "rmse_joint": f(per_seed_j),
                            "rmse_percoord_detected_only": f(per_seed_m),
                            "detection_rate": f(per_seed_det)}
    print(f"{name}: ell {np.mean(per_seed_e):.3f}  kappa {np.mean(per_seed_k):.3f}  "
          f"joint {np.mean(per_seed_j):.3f} | detected-only per-coord "
          f"{np.mean(per_seed_m):.3f} (det rate {np.mean(per_seed_det)*100:.0f}%)", flush=True)
    Path("runs/coarse_rmse.json").write_text(json.dumps(out, indent=1))

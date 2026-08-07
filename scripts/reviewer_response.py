"""Three reviewer-driven experiments, run sequentially.

(1) P_max fixed at 8 for every operating point, removing all dependence on the
    true path count P (previously P_max = P+3). HARD already used 8, so only
    EASY changes; this re-runs EASY.
(2) Correct-cell probability P(|kappa_hat_coarse - kappa| < 1/(2 beta)) and the
    RMSE conditioned on that event, which is the missing diagnostic for the
    practical-to-CRB gap.
(3) D-GESBL re-tuning on seeds DISJOINT from the evaluation seeds.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
from multiblock_dasbl import multiblock_dasbl_receiver
from dgesbl_baseline import eval_dgesbl

SEEDS = [k * 137 + 42 for k in range(5)]
SNR, NB, BS = 15.0, 8, 32
out = json.loads(Path("runs/reviewer_response.json").read_text()) if Path("runs/reviewer_response.json").exists() else {}

# ---------- (1) fixed P_max = 8, EASY ----------
print("[1] EASY P_max=8 (cached)", flush=True)
if "easy_Pmax8" in out: SKIP1=True
else: SKIP1=False
res1 = {}
for B in ([] if SKIP1 else [1, 2, 4, 8]):
    cfg = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=3, N_p=32, P_max=8)
    system, ch, const = cfg.system(), cfg.channel(), cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B,
                                      constellation=const, device=cfg.device, seed=42)
    t0 = time.time(); vals = []
    for sd in SEEDS:
        g = torch.Generator(device=cfg.device); g.manual_seed(sd); acc = 0.
        for _ in range(NB):
            b = sample_multiblock(system, ch, const, pp, pv, batch_size=BS, snr_db=SNR, generator=g)
            with torch.no_grad():
                hard, _, _, _ = multiblock_dasbl_receiver(system, b, const, cfg)
            m = b.pilot_mask
            acc += float(((hard != b.labels) * m).float().sum() / m.float().sum())
        vals.append(acc / NB)
    a = np.array(vals); res1[str(B)] = {"mean": float(a.mean()), "std": float(a.std())}
    print(f"    B={B}: {a.mean()*100:6.2f}% +/- {a.std()*100:4.2f}  ({time.time()-t0:.0f}s)", flush=True)
    out["easy_Pmax8"] = res1; Path("runs/reviewer_response.json").write_text(json.dumps(out, indent=1))

# ---------- (2) correct-cell probability ----------
print("[2] correct-cell probability and conditional RMSE", flush=True)
res2 = {}
for name, cfg in [("Easy", ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=3, N_p=32, P_max=8)),
                  ("Hard", ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=5, N_p=16, P_max=8))]:
    system, ch, const = cfg.system(), cfg.channel(), cfg.constellation()
    sr = cfg.support_recovery(); beta = (cfg.N + int(cfg.ell_max)) / cfg.N; half = 1.0 / (2 * beta)
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=1,
                                      constellation=const, device=cfg.device, seed=42)
    pc, rc, ra = [], [], []
    for sd in SEEDS:
        g = torch.Generator(device=cfg.device); g.manual_seed(sd)
        inc, allk, condk = [], [], []
        for _ in range(NB):
            b = sample_multiblock(system, ch, const, pp, pv, batch_size=BS, snr_db=SNR, generator=g)
            xp = torch.zeros(BS, cfg.N, dtype=b.r.dtype, device=cfg.device); xp[:, pp[0]] = pv[0].unsqueeze(0)
            with torch.no_grad():
                eh, kh, _ = sr(b.r[:, 0, :], system.idaft(xp))
            et = b.theta_true[..., 0].cpu().numpy(); kt = b.theta_true[..., 1].cpu().numpy()
            ehn, khn = eh.cpu().numpy(), kh.cpu().numpy()
            for i in range(BS):
                C = np.hypot(et[i][:, None] - ehn[i][None, :], kt[i][:, None] - khn[i][None, :])
                r_, c_ = linear_sum_assignment(C)
                dk = kt[i][r_] - khn[i][c_]
                inc.extend(np.abs(dk) < half); allk.extend(dk); condk.extend(dk[np.abs(dk) < half])
        pc.append(float(np.mean(inc))); ra.append(float(np.sqrt(np.mean(np.square(allk)))))
        rc.append(float(np.sqrt(np.mean(np.square(condk)))) if condk else float('nan'))
    f = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v))}
    res2[name] = {"half_cell": half, "P_cell": f(pc), "rmse_kappa_all": f(ra), "rmse_kappa_given_cell": f(rc)}
    print(f"    {name}: P_cell={np.mean(pc)*100:5.1f}%  RMSE|cell={np.mean(rc):.4f}  RMSE_all={np.mean(ra):.3f}", flush=True)
    out["correct_cell"] = res2; Path("runs/reviewer_response.json").write_text(json.dumps(out, indent=1))

# ---------- (3) D-GESBL retuned on DISJOINT seeds ----------
print("[3] D-GESBL tuning on held-out seeds (k=5,6,7; eval uses k=0..4)", flush=True)
cfgh = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=5, N_p=16, P_max=8)
tune_seeds = [k * 137 + 42 for k in (5, 6, 7)]
res3 = {}
for T in [20, 40, 80, 160]:
    for glr in [0.05, 0.1, 0.2]:
        v = [eval_dgesbl(cfgh, SNR, B_block=1, seed=s, n_batches=4, batch_size=32,
                         T_em=T, grid_lr=glr) for s in tune_seeds]
        a = np.array(v); res3[f"T_em={T},grid_lr={glr}"] = {"mean": float(a.mean()), "std": float(a.std())}
        print(f"    T_em={T:3d} lr={glr:<5} {a.mean()*100:6.2f}%", flush=True)
        out["dgesbl_heldout_tuning"] = res3
        out["dgesbl_heldout_best"] = min(res3.items(), key=lambda kv: kv[1]["mean"])[0]
        Path("runs/reviewer_response.json").write_text(json.dumps(out, indent=1))
print("BEST on held-out seeds:", out["dgesbl_heldout_best"])
print("done")

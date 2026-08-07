"""Extension of the D-GESBL-style tuning grid past its T_em boundary.

The original 20-point grid's optimum sat at T_em=40, its largest value, so the
baseline was potentially under-tuned. This extends T_em to 160 for the three
non-degenerate grid_lr values, giving 32 configurations in total, and confirms
whether grid_lr=0.1 remains optimal once T_em is no longer binding.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from afdm.experiments import ExperimentConfig
from dgesbl_baseline import eval_dgesbl

cfg = ExperimentConfig(N=128, kappa_max=5., ell_max=10., P=5, N_p=16, P_max=8)
base = json.loads(Path("runs/dgesbl_tuning.json").read_text())
res = base["results"]
for T in [60, 80, 120, 160]:
    for glr in [0.05, 0.1, 0.2]:
        t0 = time.time()
        v = [eval_dgesbl(cfg, 15.0, B_block=1, seed=k * 137 + 42, n_batches=4,
                         batch_size=32, T_em=T, grid_lr=glr) for k in range(3)]
        a = np.array(v)
        res[f"T_em={T},grid_lr={glr}"] = {"T_em": T, "grid_lr": glr,
                                          "mean": float(a.mean()), "std": float(a.std())}
        print(f"  T_em={T:3d},grid_lr={glr:<5} {a.mean()*100:6.2f}% +/- {a.std()*100:4.2f}  ({time.time()-t0:.0f}s)", flush=True)
        base["grid"] = {"T_em": [10, 15, 20, 30, 40, 60, 80, 120, 160],
                        "grid_lr": [0.0, 0.05, 0.1, 0.2],
                        "note": "grid_lr=0.0 only run for T_em<=40; it is dominated everywhere"}
        base["n_configs"] = len(res)
        b = min(res.items(), key=lambda kv: kv[1]["mean"])
        base["best"] = {"key": b[0], "mean": b[1]["mean"]}
        Path("runs/dgesbl_tuning.json").write_text(json.dumps(base, indent=1))
b = min(res.items(), key=lambda kv: kv[1]["mean"])
print(f"\n{len(res)} configurations; BEST {b[0]} at {b[1]['mean']*100:.2f}%")

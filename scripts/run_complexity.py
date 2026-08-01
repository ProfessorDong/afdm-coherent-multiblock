"""Figure 10: Per-block FLOPs vs N (analytical + measured wall-clock).

FLOPs breakdown from paper Table I; wall-clock measured on cuda:0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from afdm.experiments import load_receiver, save_results_json
from afdm.system import AFDMSystem
from afdm.operators import FastAFDMOperator
from afdm.classical import cg_solve
from _figure_utils import set_paper_style, save_figure, results_dir


def flops_analytical(N: int, P: int = 5, T: int = 8, K: int = 10, d: int = 64) -> dict:
    """Estimate per-block FLOPs per Table I of the paper."""
    ambiguity = 2 * (N ** 2) * np.log2(N + 1) if N > 1 else 0    # 2D FFT-based ambiguity
    cfar = N ** 2
    daft = T * N * np.log2(N + 1)
    h_step = T * (N * P * P + P ** 3)
    cg = T * K * (N * P + N * np.log2(N + 1))
    st = T * (P ** 2 * d + P * d * d)
    lm = T * P * (N + P)
    softmax = T * N * 4
    total = ambiguity + cfar + daft + h_step + cg + st + lm + softmax
    return {
        "ambiguity": ambiguity, "cfar": cfar, "daft": daft, "h_step": h_step,
        "cg": cg, "set_transformer": st, "lm": lm, "softmax": softmax,
        "total": total,
    }


def measure_wall_clock(N: int, P: int = 5, K: int = 10, n_reps: int = 20) -> float:
    """Measure the time for one classical CG-MMSE MVP at (N, P) on cuda:0."""
    device = "cuda:0"
    system = AFDMSystem(N=N, kappa_max=5, ell_max=10, device=device)
    ell = torch.rand(1, P, device=device) * 8
    kappa = (torch.rand(1, P, device=device) * 2 - 1) * 5
    h = torch.randn(1, P, dtype=torch.complex64, device=device)
    op = FastAFDMOperator(system=system, ell=ell, kappa=kappa, h=h)
    y = torch.randn(1, N, dtype=torch.complex64, device=device)
    def mv(v): return op.rmatvec(op.matvec(v)) + 0.01 * v
    # Warmup
    for _ in range(5): _ = cg_solve(mv, op.rmatvec(y), max_iter=K)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_reps): _ = cg_solve(mv, op.rmatvec(y), max_iter=K)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_reps


def main():
    set_paper_style()
    Ns = [64, 128, 256, 512]
    results = {"analytical": {}, "wall_clock_ms": {}}
    for N in Ns:
        a = flops_analytical(N)
        results["analytical"][N] = a
        wc = measure_wall_clock(N, n_reps=20)
        results["wall_clock_ms"][N] = wc * 1000
        print(f"  N={N}: total FLOPs {a['total']:.2e}, CG wall-clock {wc*1000:.2f} ms")
    save_results_json(results, str(results_dir() / "complexity.json"))

    fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(6.5, 2.7))
    totals = [results["analytical"][N]["total"] for N in Ns]
    Ns_dense = np.array(Ns, dtype=float)
    dense = 3 * Ns_dense ** 3  # ~O(N^3) dense-MMSE ballpark
    ax_f.loglog(Ns, totals, "o-", label="Proposed (total)", color="tab:red")
    ax_f.loglog(Ns_dense, dense, "s--", label=r"Dense MMSE $\mathcal{O}(N^3)$", color="tab:gray")
    ax_f.set_xlabel("N (subcarriers)"); ax_f.set_ylabel("FLOPs per block")
    ax_f.grid(True, which="both", alpha=0.3); ax_f.legend()
    ax_f.set_title("(a) FLOPs")

    wc_vals = [results["wall_clock_ms"][N] for N in Ns]
    ax_t.loglog(Ns, wc_vals, "o-", color="tab:red")
    ax_t.set_xlabel("N (subcarriers)"); ax_t.set_ylabel("CG-MMSE wall-clock (ms)")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.set_title("(b) CG-MMSE latency (RTX 4090)")

    fig.tight_layout()
    save_figure(fig, "complexity")
    print("Saved figures/complexity.pdf")


if __name__ == "__main__":
    main()

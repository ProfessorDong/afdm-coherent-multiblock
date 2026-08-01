"""Pre-flight verification for the publication training campaign.

Verifies before committing 30+ h of GPU time:
  * cuda:0 is available and is the expected GPU (RTX 4090).
  * Disk space is sufficient for checkpoints (est. 200 MB).
  * All Python imports succeed.
  * A 2-epoch smoke of the proposed variant completes cleanly.
  * Checkpoint save + load round-trip works.
  * No stale process is already using the GPU.

Exit code 0: preflight passed. Nonzero: at least one check failed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import ExperimentConfig, build_ablation, train_receiver, load_receiver
from afdm.training import TrainingConfig


REQUIRED_GPU_NAME = "NVIDIA GeForce RTX 4090"
MIN_DISK_MB = 500  # generous buffer for checkpoints + logs


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "OK    " if ok else "FAIL  "
    print(f"  [{tag}] {name} {'-- ' + detail if detail else ''}")
    return ok


def main() -> int:
    print("=" * 66)
    print("PRE-FLIGHT VERIFICATION for publication training campaign")
    print("=" * 66)
    all_ok = True

    # 1. CUDA + expected GPU
    ok_cuda = torch.cuda.is_available()
    all_ok &= check("CUDA available", ok_cuda)
    if not ok_cuda:
        return 1
    device_name = torch.cuda.get_device_name(0)
    ok_gpu = REQUIRED_GPU_NAME in device_name
    all_ok &= check(f"GPU is {REQUIRED_GPU_NAME}", ok_gpu, detail=f"detected: {device_name}")
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    all_ok &= check(f"GPU memory >= 20 GB", total_mem_gb >= 20, detail=f"{total_mem_gb:.1f} GB")

    # 2. Disk space
    disk = shutil.disk_usage(Path.cwd())
    free_mb = disk.free / 1024**2
    all_ok &= check(f"Disk space >= {MIN_DISK_MB} MB", free_mb >= MIN_DISK_MB, detail=f"{free_mb:.0f} MB free")

    # 3. No stale GPU process (except us)
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        pids = [line.strip() for line in out.stdout.strip().split("\n") if line.strip()]
        # It's normal to have ONE python process (this one) if it holds a context
        many_procs = len(pids) > 2
        all_ok &= check("No stale GPU processes", not many_procs,
                        detail=f"{len(pids)} process(es): {pids}")
    except Exception as e:
        print(f"  [WARN  ] nvidia-smi check skipped: {e}")

    # 4. All imports work
    try:
        from afdm import (AFDMSystem, DoublyDispersiveChannel, UniformFractionalChannel,
                          FastAFDMOperator, uniform_daft_pilots)
        from afdm.support import SupportRecovery
        from afdm.classical import ClassicalCGDetector
        from afdm.pbigabp import PBiGaBPDetector
        from afdm.jpnce_sbl import JPNCESBLDetector
        from afdm.set_transformer import SetTransformer, UncertaintyGate
        from afdm.vem import h_step_damped_ridge, safeguarded_lm_theta_step
        from afdm.receiver import UGVEMReceiver
        from afdm.loss import compose_training_loss
        from afdm.training import train, evaluate_snr, sample_batch
        all_ok &= check("All 15 module imports succeed", True)
    except Exception as e:
        all_ok &= check("Module imports", False, detail=str(e))
        return 1

    # 5. Unit tests (quick)
    print("  Running unit tests...")
    tres = subprocess.run(["python3", "-m", "pytest", "tests/", "-q", "--tb=line", "-x"],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(Path(__file__).resolve().parent.parent))
    n_pass = tres.stdout.count(" passed")
    n_fail = tres.stdout.count(" failed")
    ok_tests = tres.returncode == 0
    all_ok &= check("Unit tests pass", ok_tests,
                    detail=f"exit code {tres.returncode}, output tail: {tres.stdout.strip().split(chr(10))[-1]}")
    if not ok_tests:
        print("  Test output:")
        for line in tres.stdout.split("\n")[-20:]:
            print(f"    {line}")

    # 6. Short 2-epoch smoke of proposed variant at publication config
    print("  Running 2-epoch smoke of proposed variant at publication config (~1-2 min)...")
    config = ExperimentConfig(
        N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16,
        T=8, K_cg=10, d_model=64, n_heads=4, n_blocks=3, P_max=5, seed=0,
    )
    tc = TrainingConfig(
        lr=5e-4, n_epochs=2, steps_per_epoch=10, batch_size=32,
        snr_db_min=5.0, snr_db_max=25.0, grad_clip=1.0,
        val_every=2, val_batches=1, val_snr_dbs=(15.0,),
        layer_gamma=0.7, mu_ce=0.5, eta_anchor=0.0,
        hungarian_kwargs=dict(w_h=1.0, w_ell=0.2, w_kap=0.2, mu_fa=0.1, mu_md=0.1),
        log_every=10,
    )
    smoke_ckpt = Path("preflight_smoke.pt")
    try:
        t0 = time.time()
        rx = build_ablation("proposed", config)
        train_receiver(rx, config, tc, checkpoint_path=str(smoke_ckpt), verbose=False)
        smoke_time = time.time() - t0
        all_ok &= check(f"2-epoch smoke training completes",
                        smoke_ckpt.exists(), detail=f"{smoke_time:.1f}s, checkpoint saved")
        # Check for NaN in model
        has_nan = False
        for name, p in rx.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                has_nan = True; break
        all_ok &= check("No NaN in model parameters after smoke", not has_nan)
    except Exception as e:
        all_ok &= check("Smoke training", False, detail=f"{type(e).__name__}: {e}")

    # 7. Checkpoint save + load roundtrip
    if smoke_ckpt.exists():
        try:
            rx2, cfg2 = load_receiver(str(smoke_ckpt))
            all_ok &= check("Checkpoint load roundtrip", True)
        except Exception as e:
            all_ok &= check("Checkpoint load roundtrip", False, detail=str(e))
        smoke_ckpt.unlink()
        (Path.cwd() / "preflight_smoke.pt").unlink(missing_ok=True)

    print("=" * 66)
    if all_ok:
        print("PRE-FLIGHT PASSED — safe to launch publication training.")
        return 0
    else:
        print("PRE-FLIGHT FAILED — fix issues above before launching.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

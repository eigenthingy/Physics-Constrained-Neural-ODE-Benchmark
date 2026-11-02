"""60-second demo: train all four models on the pendulum and print the table.

    python scripts/demo.py

This is a small-budget version of the real study (fewer steps, one seed) meant
to make the mechanism visible, not to produce publishable numbers. For those,
run the grid: `python scripts/run_experiments.py --shard 0 --n-shards 1`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcnn.data import make_data  # noqa: E402
from pcnn.metrics import evaluate, make_reference  # noqa: E402
from pcnn.models import MODELS, make_model  # noqa: E402
from pcnn.systems import make_system  # noqa: E402
from pcnn.train import TrainConfig, train  # noqa: E402

STEPS, LAM_B, SEED = 1500, 1.0, 0

if __name__ == "__main__":
    # the table uses ∇ and Δ; a default Windows console (cp1252) cannot print them
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(4)
    system = make_system("pendulum")
    data = make_data(system, seed=SEED, n_traj=40, T=6.0, dt=0.05, noise=0.01,
                     e_train=(0.2, 1.0), e_extrap=(1.2, 1.8))
    ref_in = make_reference(system, data.test_ic, 60.0, 0.05)
    ref_out = make_reference(system, data.test_ic_out, 60.0, 0.05)

    print(f"\nnonlinear pendulum · {len(data.train)} noisy samples · "
          f"derivative SNR {data.train.target_snr:.1f}\n")
    hdr = f"{'model':<22}{'params':>8}{'err(in)':>11}{'err(unseen)':>13}" \
          f"{'|cos(∇H,f)|':>13}{'ΔE':>11}{'valid T':>9}"
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for key, cls in MODELS.items():
        m = make_model(key, system, seed=SEED)
        t0 = time.time()
        train(m, system, data, TrainConfig(steps=STEPS, lam=LAM_B if key == "B" else 0.0,
                                           seed=SEED))
        ev = evaluate(m, system, data, ref_in, ref_out)
        rows[key] = (m, ev)
        print(f"{key + ': ' + cls.enforcement:<22}{ev['n_params']:>8}"
              f"{ev['vf_mse_in']:>11.2e}{ev['vf_mse_out']:>13.2e}"
              f"{ev['residual_cos_in']:>13.2e}{ev['energy_drift_rk4']:>11.2e}"
              f"{ev['valid_time_rk4']:>9.1f}   ({time.time() - t0:.0f}s)")

    r2 = rows["C"][1]["invariant_r2"]
    print(f"""
Read the |cos(∇H,f)| column first. That is dH/dt along the learned flow:
  A  no constraint          -> violates the conservation law freely
  B  soft penalty, λ={LAM_B:<6g} -> smaller, but never zero, and λ had to be chosen
  C  Hamiltonian net        -> conserves ITS OWN invariant exactly; the number you
                               see is how far that invariant is from the physical
                               energy (R² vs true energy = {r2:.4f})
  D  tangent projection     -> zero to machine precision, by construction

Then run `python scripts/run_experiments.py --quick` and `scripts/make_figures.py`
for the version with error bars, and read FINDINGS.md for where each one breaks.
""")

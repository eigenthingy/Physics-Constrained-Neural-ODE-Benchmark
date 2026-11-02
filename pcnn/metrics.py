"""Scoring.

Four families, matching the four questions the study asks:

  error        vf_mse_in            one-step accuracy where the data lives
  extrapolation vf_mse_out          same, on an energy shell never seen
  stability    valid_time,          how long a rollout tracks truth before
               energy_drift,        losing it, and whether it stays physical
               blowup
  consistency  residual_cos         does the flow actually conserve H?

Plus one metric that exists only because of this repo's thesis:

  invariant_r2 -- for the Hamiltonian model, how well its *learned* invariant
  matches the physical energy (R^2 of the best affine fit). An exactly
  conserved quantity that is not the energy is a guarantee about nothing.

`valid_time` uses a threshold relative to the trajectory's own scale, so the
number is comparable between the pendulum and Henon-Heiles.
"""

from __future__ import annotations

import numpy as np
import torch

from .integrators import ROLLOUTS, reference_rollout
from .losses import residual_cosine
from .models import ModelC_Hamiltonian
from .systems import System

BLOWUP = 1e3


def make_reference(system: System, ic: torch.Tensor, T: float, dt: float) -> dict:
    """Truth trajectories for a fixed set of ICs. Computed once, reused by all
    models so every model is scored against exactly the same yardstick."""
    t = np.arange(0.0, T + 0.5 * dt, dt)
    traj = reference_rollout(system, ic.numpy(), t)
    return dict(t=t, traj=torch.tensor(traj, dtype=torch.get_default_dtype()), dt=dt)


def _valid_time(err: torch.Tensor, ref: torch.Tensor, t: np.ndarray) -> float:
    """First time the state error exceeds 20% of the trajectory's RMS norm.

    Median over initial conditions; capped at the horizon (so a model that
    never fails reports the full horizon, not infinity)."""
    scale = ref.norm(dim=-1).mean(dim=1, keepdim=True).clamp_min(1e-8)  # (n_ic, 1)
    bad = err > 0.2 * scale
    horizon = float(t[-1])
    out = []
    for row in bad:
        idx = torch.nonzero(row, as_tuple=False)
        out.append(float(t[int(idx[0])]) if len(idx) else horizon)
    return float(np.median(out))


@torch.no_grad()
def rollout_scores(model, system, ref, integrator: str) -> dict:
    t, truth, dt = ref["t"], ref["traj"], ref["dt"]
    z0 = truth[:, 0, :]
    pred = ROLLOUTS[integrator](model, z0, dt, len(t) - 1)

    err = (pred - truth).norm(dim=-1)  # (n_ic, n_t)
    blow = (pred.norm(dim=-1).max(dim=1).values > BLOWUP).float().mean().item()

    e_true = system.H_torch(truth[:, 0, :])
    e_pred = system.H_torch(pred)
    scale = e_true.abs().mean().clamp_min(1e-8)
    drift = ((e_pred[:, -1] - e_true).abs() / scale).median().item()
    drift_max = ((e_pred - e_true[:, None]).abs().max(dim=1).values / scale).median().item()

    half = len(t) // 2
    return {
        f"rollout_rmse_half_{integrator}": err[:, half].median().item(),
        f"rollout_rmse_end_{integrator}": err[:, -1].median().item(),
        f"valid_time_{integrator}": _valid_time(err, truth, t),
        f"energy_drift_{integrator}": drift,
        f"energy_drift_max_{integrator}": drift_max,
        f"blowup_{integrator}": blow,
    }


@torch.no_grad()
def _invariant_r2(model, system, z) -> float:
    """R^2 of the affine fit  H_learned ~ a * H_true + b.

    Affine, not linear: a Hamiltonian is only defined up to scale and offset,
    so penalising those would be unfair. What is left is the question that
    matters -- is it the *same function* of state?
    """
    h_hat = model.H(z).double().numpy()
    h_true = system.H_torch(z).double().numpy()
    A = np.stack([h_true, np.ones_like(h_true)], axis=1)
    coef, *_ = np.linalg.lstsq(A, h_hat, rcond=None)
    resid = h_hat - A @ coef
    ss_tot = float(((h_hat - h_hat.mean()) ** 2).sum())
    return float(1.0 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def evaluate(
    model,
    system: System,
    data,
    ref_in: dict,
    ref_out: dict,
    integrators=("rk4",),
) -> dict:
    model.eval()
    out: dict = {}

    with torch.no_grad():
        out["vf_mse_in"] = torch.nn.functional.mse_loss(
            model(data.test_in.z), data.test_in.dz
        ).item()
        out["vf_mse_out"] = torch.nn.functional.mse_loss(
            model(data.test_out.z), data.test_out.dz
        ).item()

    out["residual_cos_in"] = residual_cosine(model, system, data.test_in.z)
    out["residual_cos_out"] = residual_cosine(model, system, data.test_out.z)

    for integ in integrators:
        out.update(rollout_scores(model, system, ref_in, integ))
        out.update(
            {k + "_extrap": v for k, v in rollout_scores(model, system, ref_out, integ).items()}
        )

    if isinstance(model, ModelC_Hamiltonian):
        out["invariant_r2"] = _invariant_r2(model, system, data.test_in.z)
        out["invariant_r2_out"] = _invariant_r2(model, system, data.test_out.z)
    else:
        out["invariant_r2"] = float("nan")
        out["invariant_r2_out"] = float("nan")

    out["n_params"] = model.n_parameters()
    return out


def bootstrap_ci(values, n_boot=4000, alpha=0.05, stat=np.median, seed=0):
    """Percentile bootstrap CI of a statistic across seeds."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = stat(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    return float(stat(v)), float(np.percentile(draws, 100 * alpha / 2)), float(
        np.percentile(draws, 100 * (1 - alpha / 2))
    )

"""Data generation.

One deliberate choice drives everything here: **the models never receive exact
derivatives.** We observe a noisy trajectory at spacing `dt` and estimate dz/dt
by central differences, the way anyone working from measurements would.

That single decision is what makes the noise study meaningful. Central
differencing amplifies observation noise by 1/(sqrt(2) dt): at dt = 0.05, a
sigma = 0.03 wobble on the state becomes a sigma ~ 0.42 wobble on the target,
against a signal of order 1. Handing models analytic derivatives -- as most
PINN demos do -- hides exactly the regime where a physics prior is supposed to
earn its keep.

Train and test regions are separated by *energy*, not by a bounding box, so
"extrapolation" means a physically meaningful shell the model never saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .integrators import reference_rollout
from .systems import System


@dataclass
class Batch:
    """Supervised pairs plus the clean counterparts (for diagnostics only --
    nothing downstream of training is allowed to look at these)."""

    z: torch.Tensor
    dz: torch.Tensor
    z_clean: torch.Tensor
    dz_exact: torch.Tensor

    def __len__(self):
        return self.z.shape[0]

    @property
    def target_snr(self) -> float:
        """Signal-to-noise of the finite-difference targets. Reported in
        results so a reader can see how hard each noise level actually is."""
        num = self.dz_exact.pow(2).mean().item()
        den = (self.dz - self.dz_exact).pow(2).mean().item()
        return float("inf") if den == 0 else float(np.sqrt(num / den))


@dataclass
class DataBundle:
    train: Batch
    val: Batch
    test_in: Batch
    test_out: Batch
    colloc: torch.Tensor
    test_ic: torch.Tensor  #: initial conditions for rollout scoring (in-dist)
    test_ic_out: torch.Tensor  #: ditto, extrapolation shell
    meta: dict


def _trajectories(system, ic, T, dt):
    t = np.arange(0.0, T + 0.5 * dt, dt)
    return reference_rollout(system, ic, t), t


def _finite_difference(traj_noisy, traj_clean, system, dt):
    """Central differences on the noisy samples; inputs are the noisy midpoints.

    Inputs and targets therefore share a noise realisation, as they would in
    practice -- an errors-in-variables problem, not a clean regression.
    """
    z = traj_noisy[:, 1:-1, :]
    dz = (traj_noisy[:, 2:, :] - traj_noisy[:, :-2, :]) / (2.0 * dt)
    zc = traj_clean[:, 1:-1, :]
    dz_exact = system.f_np(zc)
    flat = lambda a: torch.tensor(a.reshape(-1, a.shape[-1]), dtype=torch.get_default_dtype())
    return Batch(flat(z), flat(dz), flat(zc), flat(dz_exact))


def make_data(
    system: System,
    *,
    seed: int = 0,
    n_traj: int = 40,
    T: float = 6.0,
    dt: float = 0.05,
    noise: float = 0.0,
    e_train: tuple[float, float] = (0.2, 1.0),
    e_extrap: tuple[float, float] = (1.2, 1.8),
    n_colloc: int = 4096,
    n_test_traj: int = 24,
    val_frac: float = 0.2,
) -> DataBundle:
    rng = np.random.default_rng(seed)

    # --- training / validation ---------------------------------------------
    ic = system.sample_states(n_traj, rng, *e_train)
    traj, _ = _trajectories(system, ic, T, dt)
    traj_noisy = traj + noise * rng.standard_normal(traj.shape)
    full = _finite_difference(traj_noisy, traj, system, dt)

    n = len(full)
    perm = torch.tensor(rng.permutation(n))
    n_val = int(val_frac * n)
    vi, ti = perm[:n_val], perm[n_val:]
    train = Batch(full.z[ti], full.dz[ti], full.z_clean[ti], full.dz_exact[ti])
    val = Batch(full.z[vi], full.dz[vi], full.z_clean[vi], full.dz_exact[vi])

    # --- test sets: clean, exact derivatives, never used for fitting -------
    def clean_batch(e_range, m):
        s = system.sample_states(m, rng, *e_range)
        st = torch.tensor(s, dtype=torch.get_default_dtype())
        dt_ = torch.tensor(system.f_np(s), dtype=torch.get_default_dtype())
        return Batch(st, dt_, st, dt_)

    test_in = clean_batch(e_train, 4000)
    test_out = clean_batch(e_extrap, 4000)

    # --- collocation points for the soft penalty ---------------------------
    # Unlabelled, so they are "free" in the PINN sense: the residual needs no
    # ground truth. They cover the *training* shell only; giving B the
    # extrapolation region would be leaking the test set.
    colloc = torch.tensor(
        system.sample_states(n_colloc, rng, *e_train), dtype=torch.get_default_dtype()
    )

    test_ic = torch.tensor(
        system.sample_states(n_test_traj, rng, *e_train), dtype=torch.get_default_dtype()
    )
    test_ic_out = torch.tensor(
        system.sample_states(n_test_traj, rng, *e_extrap), dtype=torch.get_default_dtype()
    )

    meta = dict(
        system=system.name,
        seed=seed,
        n_traj=n_traj,
        T=T,
        dt=dt,
        noise=noise,
        n_train=len(train),
        e_train=list(e_train),
        e_extrap=list(e_extrap),
        target_snr=train.target_snr,
    )
    return DataBundle(train, val, test_in, test_out, colloc, test_ic, test_ic_out, meta)

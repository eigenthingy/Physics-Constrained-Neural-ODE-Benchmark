"""Integrators.

Two roles, deliberately kept apart:

* `reference_rollout` -- SciPy DOP853 at rtol=atol=1e-11 on the *true* vector
  field. This is the yardstick. It never touches a model.
* `rk4_rollout` / `implicit_midpoint_rollout` -- Torch, batched, used to roll
  models forward.

The second pair exists in two flavours on purpose. A structural conservation
guarantee lives in the continuous-time vector field; a non-symplectic
integrator can spend it. Comparing RK4 against implicit midpoint on the *same*
trained model isolates how much of the observed energy drift is the model and
how much is the solver -- a distinction most PINN write-ups skip.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------- #
# reference (NumPy / SciPy)
# --------------------------------------------------------------------------- #


def reference_rollout(system, z0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Integrate a *batch* of initial conditions with one stiff-tolerance solve.

    All trajectories are stacked into a single ODE of size (n_ic * d) so the
    adaptive stepper sees them at once. Returns (n_ic, n_t, d).
    """
    z0 = np.atleast_2d(np.asarray(z0, dtype=np.float64))
    n, d = z0.shape

    def rhs(_t, y):
        return system.f_np(y.reshape(n, d)).reshape(-1)

    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        z0.reshape(-1),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-11,
        atol=1e-11,
        max_step=0.5,
    )
    if not sol.success:
        raise RuntimeError(f"reference integration failed: {sol.message}")
    return sol.y.reshape(n, d, -1).transpose(0, 2, 1)


# --------------------------------------------------------------------------- #
# model rollouts (Torch)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def rk4_rollout(
    f: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    dt: float,
    n_steps: int,
    blowup: float = 1e4,
) -> torch.Tensor:
    """Classic RK4. Returns (n_ic, n_steps + 1, d).

    Diverged trajectories are frozen rather than allowed to become inf/nan, so
    one exploding initial condition cannot poison a batch statistic. The
    `blowup` fraction is reported separately in `metrics.py`.
    """
    zs = [z0]
    z = z0
    for _ in range(n_steps):
        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        z_next = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z_next = torch.nan_to_num(z_next, nan=blowup, posinf=blowup, neginf=-blowup)
        dead = z.norm(dim=-1, keepdim=True) > blowup
        z = torch.where(dead, z, z_next)
        zs.append(z)
    return torch.stack(zs, dim=1)


@torch.no_grad()
def implicit_midpoint_rollout(
    f: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    dt: float,
    n_steps: int,
    n_iter: int = 12,
    blowup: float = 1e4,
) -> torch.Tensor:
    """Implicit midpoint, solved by fixed-point iteration.

    Symplectic for *any* Hamiltonian vector field (unlike leapfrog, which needs
    a separable H -- a learned H is not separable). This is the integrator that
    lets a Hamiltonian architecture actually cash in its guarantee.
    """
    zs = [z0]
    z = z0
    for _ in range(n_steps):
        z_mid = z
        for _ in range(n_iter):
            z_mid = z + 0.5 * dt * f(z_mid)
        z_next = z + dt * f(z_mid)
        z_next = torch.nan_to_num(z_next, nan=blowup, posinf=blowup, neginf=-blowup)
        dead = z.norm(dim=-1, keepdim=True) > blowup
        z = torch.where(dead, z, z_next)
        zs.append(z)
    return torch.stack(zs, dim=1)


ROLLOUTS = {"rk4": rk4_rollout, "midpoint": implicit_midpoint_rollout}

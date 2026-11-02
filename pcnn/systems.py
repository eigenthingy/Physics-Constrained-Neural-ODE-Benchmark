"""Ground-truth dynamical systems.

Each system exposes the same interface twice: a NumPy side (used by SciPy to
produce reference trajectories at tight tolerance) and a Torch side (used inside
training, where we need autograd and batching).

The split matters for honesty: the "truth" a model is scored against must never
be produced by the same code path the model uses to roll out. `tests/` asserts
the two sides agree to 1e-10.

State layout is always `z = [q (dof), p (dof)]`, so `d = 2 * dof`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #


@dataclass
class System:
    """A dynamical system with a (possibly only *assumed*) conserved quantity.

    `is_hamiltonian` is the ground truth about the system. `H` is the quantity a
    physics-informed model is *told* to conserve. For the damped pendulum those
    two disagree on purpose -- that is the misspecification experiment.
    """

    name: str
    dof: int

    @property
    def dim(self) -> int:
        return 2 * self.dof

    is_hamiltonian: bool = True

    # --- numpy side (reference integration) -------------------------------- #
    def f_np(self, z: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def H_np(self, z: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- torch side (models, losses, metrics) ------------------------------ #
    def f_torch(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def H_torch(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def gradH_torch(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Analytic dH/dz, shape (N, d). Analytic (not autograd) so that the
        hard-constraint model D stays cheap and exactly differentiable."""
        raise NotImplementedError

    # --- sampling ---------------------------------------------------------- #
    def sample_states(
        self, n: int, rng: np.random.Generator, e_lo: float, e_hi: float
    ) -> np.ndarray:
        """Sample states uniformly-ish inside the energy shell [e_lo, e_hi].

        Sampling *by energy* (rather than in a box) is what makes the
        extrapolation study clean: train and test regions are separated by a
        physically meaningful coordinate, not by an arbitrary bounding box.
        """
        out, tries = [], 0
        while len(out) < n and tries < 400:
            tries += 1
            cand = self._propose(int(2.5 * (n - len(out))) + 64, rng, e_lo, e_hi)
            out.extend(cand)
        if len(out) < n:
            raise RuntimeError(f"{self.name}: could not sample energy shell [{e_lo},{e_hi}]")
        return np.asarray(out[:n], dtype=np.float64)

    def _propose(self, m, rng, e_lo, e_hi) -> list:  # pragma: no cover - abstract
        raise NotImplementedError


def _symplectic_J(d: int, device, dtype) -> torch.Tensor:
    """The canonical symplectic form J = [[0, I], [-I, 0]]."""
    n = d // 2
    J = torch.zeros(d, d, device=device, dtype=dtype)
    J[:n, n:] = torch.eye(n, device=device, dtype=dtype)
    J[n:, :n] = -torch.eye(n, device=device, dtype=dtype)
    return J


# --------------------------------------------------------------------------- #
# 1 DOF: the nonlinear pendulum
# --------------------------------------------------------------------------- #


@dataclass
class Pendulum(System):
    """H = p^2/2 + (1 - cos q).  Separatrix at E = 2."""

    name: str = "pendulum"
    dof: int = 1
    is_hamiltonian: bool = True

    def f_np(self, z):
        q, p = z[..., 0], z[..., 1]
        return np.stack([p, -np.sin(q)], axis=-1)

    def H_np(self, z):
        q, p = z[..., 0], z[..., 1]
        return 0.5 * p**2 + (1.0 - np.cos(q))

    def f_torch(self, z):
        q, p = z[..., 0], z[..., 1]
        return torch.stack([p, -torch.sin(q)], dim=-1)

    def H_torch(self, z):
        q, p = z[..., 0], z[..., 1]
        return 0.5 * p**2 + (1.0 - torch.cos(q))

    def gradH_torch(self, z):
        q, p = z[..., 0], z[..., 1]
        return torch.stack([torch.sin(q), p], dim=-1)

    def _propose(self, m, rng, e_lo, e_hi):
        q = rng.uniform(-np.pi, np.pi, size=m)
        e = rng.uniform(e_lo, e_hi, size=m)
        v = 1.0 - np.cos(q)
        ok = e > v + 1e-9
        q, e, v = q[ok], e[ok], v[ok]
        p = np.sqrt(2.0 * (e - v)) * rng.choice([-1.0, 1.0], size=q.shape[0])
        return list(np.stack([q, p], axis=-1))


@dataclass
class DampedPendulum(Pendulum):
    """Same H, but the true flow dissipates: p' = -sin q - alpha * p.

    Models B/C/D are handed the *undamped* H and told to conserve it. They are
    therefore wrong by construction. This is the point of the system.
    """

    name: str = "damped_pendulum"
    is_hamiltonian: bool = False
    alpha: float = 0.15

    def f_np(self, z):
        q, p = z[..., 0], z[..., 1]
        return np.stack([p, -np.sin(q) - self.alpha * p], axis=-1)

    def f_torch(self, z):
        q, p = z[..., 0], z[..., 1]
        return torch.stack([p, -torch.sin(q) - self.alpha * p], dim=-1)


# --------------------------------------------------------------------------- #
# 2 DOF: Henon-Heiles (chaotic above E = 1/6)
# --------------------------------------------------------------------------- #


@dataclass
class HenonHeiles(System):
    """H = (p1^2+p2^2)/2 + (q1^2+q2^2)/2 + q1^2 q2 - q2^3/3.

    Bounded below the escape energy E* = 1/6 ~ 0.1667; largely regular below
    E ~ 0.08 and strongly chaotic near E ~ 0.13. That gradient is exactly what
    makes it a fair stability / extrapolation test rather than a toy.
    """

    name: str = "henon_heiles"
    dof: int = 2
    is_hamiltonian: bool = True

    E_ESCAPE: ClassVar[float] = 1.0 / 6.0

    @staticmethod
    def _V_np(q1, q2):
        return 0.5 * (q1**2 + q2**2) + q1**2 * q2 - q2**3 / 3.0

    def f_np(self, z):
        q1, q2, p1, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        return np.stack(
            [p1, p2, -(q1 + 2.0 * q1 * q2), -(q2 + q1**2 - q2**2)], axis=-1
        )

    def H_np(self, z):
        q1, q2, p1, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        return 0.5 * (p1**2 + p2**2) + self._V_np(q1, q2)

    def f_torch(self, z):
        q1, q2, p1, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        return torch.stack(
            [p1, p2, -(q1 + 2.0 * q1 * q2), -(q2 + q1**2 - q2**2)], dim=-1
        )

    def H_torch(self, z):
        q1, q2, p1, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        v = 0.5 * (q1**2 + q2**2) + q1**2 * q2 - q2**3 / 3.0
        return 0.5 * (p1**2 + p2**2) + v

    def gradH_torch(self, z):
        q1, q2, p1, p2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        return torch.stack(
            [q1 + 2.0 * q1 * q2, q2 + q1**2 - q2**2, p1, p2], dim=-1
        )

    def _propose(self, m, rng, e_lo, e_hi):
        q1 = rng.uniform(-0.55, 0.55, size=m)
        q2 = rng.uniform(-0.55, 0.75, size=m)
        e = rng.uniform(e_lo, e_hi, size=m)
        v = self._V_np(q1, q2)
        ok = e > v + 1e-9
        q1, q2, e, v = q1[ok], q2[ok], e[ok], v[ok]
        speed = np.sqrt(2.0 * (e - v))
        th = rng.uniform(0.0, 2.0 * np.pi, size=q1.shape[0])
        return list(np.stack([q1, q2, speed * np.cos(th), speed * np.sin(th)], axis=-1))


# --------------------------------------------------------------------------- #

REGISTRY = {
    "pendulum": Pendulum,
    "damped_pendulum": DampedPendulum,
    "henon_heiles": HenonHeiles,
}


def make_system(name: str, **kw) -> System:
    if name not in REGISTRY:
        raise KeyError(f"unknown system {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)

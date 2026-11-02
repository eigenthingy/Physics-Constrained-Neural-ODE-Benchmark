"""Objective terms.

`physics_residual` is the quantity every model in the study is measured on and
the quantity model B is trained on:

    r(z) = < grad H(z), f_theta(z) >          ( = dH/dt along the learned flow )

It is zero iff the learned flow is tangent to the level set of H. Two versions:

* `physics_residual`  -- raw, squared, for the loss. lambda absorbs the scale.
* `residual_cosine`   -- normalised to [0, 1], for reporting. Dimensionless, so
  it is comparable across systems and across noise levels, and it cannot be
  gamed by a model that simply predicts a very small vector field.

The normalised version matters more than it sounds: a network that collapses to
f ~ 0 has a tiny raw residual and looks beautifully "physics-consistent" while
being useless. Any paper reporting only the raw penalty cannot distinguish the
two. We report both.
"""

from __future__ import annotations

import torch

from .systems import System


def physics_residual(model, system: System, z: torch.Tensor) -> torch.Tensor:
    """Mean squared dH/dt along the learned flow. The PINN penalty term."""
    f = model(z)
    n = system.gradH_torch(z)
    return ((n * f).sum(-1) ** 2).mean()


@torch.no_grad()
def residual_cosine(model, system: System, z: torch.Tensor) -> float:
    """|cos angle between f_theta and grad H|, averaged. Scale-free in [0, 1].

    0 = flow exactly tangent to the energy level set (conservation).
    """
    f = model(z)
    n = system.gradH_torch(z)
    num = (n * f).sum(-1).abs()
    den = n.norm(dim=-1) * f.norm(dim=-1) + 1e-12
    return float((num / den).mean())

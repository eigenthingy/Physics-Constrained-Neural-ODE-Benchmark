"""The four ways of putting physics into a network.

The whole study hangs on one taxonomy. A physics prior has two independent
axes, and the usual "PINN vs baseline" framing collapses them into one:

                        | knows the invariant H     | knows only the *structure*
  ----------------------+---------------------------+---------------------------
  enforced SOFTLY       | B  penalty  dH/dt -> 0     | (open cell, see README)
  (loss term)           |                           |
  ----------------------+---------------------------+---------------------------
  enforced HARD         | D  tangent projection      | C  Hamiltonian net
  (architecture)        |    conserves the TRUE H    |    conserves a LEARNED H
                        |    exactly                 |    exactly

Model A sits outside the table: no physics at all.

Read the bottom row carefully, because it is the finding the repo exists for:
**C and D both give you an exact conservation law, but not of the same thing.**
D conserves the energy you handed it. C conserves whatever scalar its network
happened to converge to -- exactly, to machine precision, and possibly not the
physical energy. "Exactly conserved" and "conserves the right thing" are
different claims, and only one of them is free.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .systems import System, _symplectic_J


# --------------------------------------------------------------------------- #
# shared trunk
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
    """Plain tanh MLP. Every model in the study uses this and only this, so a
    performance gap is attributable to the constraint, not to the trunk."""

    def __init__(self, d_in: int, d_out: int, width: int = 128, depth: int = 3):
        super().__init__()
        layers, prev = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(prev, width), nn.Tanh()]
            prev = width
        layers += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=5.0 / 3.0)  # tanh gain
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def _n_params(d_in: int, d_out: int, width: int, depth: int) -> int:
    n = d_in * width + width
    n += (depth - 1) * (width * width + width)
    n += width * d_out + d_out
    return n


def width_for_parity(d_in: int, d_out_ref: int, width_ref: int, depth: int) -> int:
    """Width a scalar-output net needs to match a vector-output net's budget.

    Without this, the Hamiltonian model quietly runs with fewer parameters than
    the baselines and any win is contaminated. Search is exhaustive and cheap.
    """
    target = _n_params(d_in, d_out_ref, width_ref, depth)
    cands = range(max(4, width_ref - 12), width_ref + 13)
    return min(cands, key=lambda w: abs(_n_params(d_in, 1, w, depth) - target))


# --------------------------------------------------------------------------- #
# the four models
# --------------------------------------------------------------------------- #


class PhysicsModel(nn.Module):
    """Common interface: `forward(z) -> dz/dt`."""

    #: what the model is told about the physics, for the README taxonomy table
    knowledge = "none"
    #: how the constraint is imposed
    enforcement = "none"
    label = "?"

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ModelA_Unconstrained(PhysicsModel):
    """A -- black box. Learns dz/dt directly. No physics anywhere."""

    knowledge, enforcement, label = "none", "none", "A: unconstrained"

    def __init__(self, system: System, width=128, depth=3):
        super().__init__()
        self.net = MLP(system.dim, system.dim, width, depth)

    def forward(self, z):
        return self.net(z)


class ModelB_SoftPenalty(ModelA_Unconstrained):
    """B -- the standard PINN recipe: same net, plus lambda * physics residual.

    The penalty itself lives in `losses.physics_residual` because it is a
    property of the *training objective*, not of the architecture. That is
    precisely what makes B different in kind from C and D: at inference time B
    is byte-for-byte model A. Nothing about the deployed network knows physics.
    """

    knowledge, enforcement, label = "explicit H", "soft (loss)", "B: soft penalty"


class ModelC_Hamiltonian(PhysicsModel):
    """C -- Hamiltonian network. The net outputs a scalar H_theta; the vector
    field is dz/dt = J grad H_theta.

    Guarantee (exact, provable, unit-tested): grad H_theta . dz/dt = 0 for every
    input, because J is antisymmetric. The network *cannot* express a flow that
    changes its own H. It knows only that a conserved quantity exists -- not
    which one.
    """

    knowledge, enforcement, label = "structure only", "hard (architecture)", "C: Hamiltonian net"

    def __init__(self, system: System, width=128, depth=3):
        super().__init__()
        w = width_for_parity(system.dim, system.dim, width, depth)
        self.net = MLP(system.dim, 1, w, depth)
        self.register_buffer("J", _symplectic_J(system.dim, "cpu", torch.get_default_dtype()))

    def H(self, z):
        return self.net(z).squeeze(-1)

    def forward(self, z):
        # autograd through the input, and re-enabled explicitly so the model
        # still works inside a no_grad rollout.
        with torch.enable_grad():
            zg = z if z.requires_grad else z.clone().requires_grad_(True)
            h = self.net(zg).sum()
            (dH,) = torch.autograd.grad(h, zg, create_graph=self.training)
        if not self.training:
            dH = dH.detach()
        return dH @ self.J.T


class ModelD_Projected(PhysicsModel):
    """D -- hard constraint on the *true* invariant, by projection.

    A free net proposes g(z); the output is g projected onto the tangent space
    of the level set of the known H:

        f(z) = g(z) - <g(z), n(z)> / <n(z), n(z)> * n(z),     n = grad H_true

    so <grad H_true, f> = 0 identically. Exact conservation of the *physical*
    energy, at the cost of needing H in closed form.

    Watch the dimension count. The constraint removes one direction out of d.
    In 1 DOF (d = 2) that leaves a single free direction, so D can only learn a
    scalar speed along the true flow -- almost the answer for free. In 2 DOF
    (d = 4) three directions survive and the prior is genuinely weak. The value
    of a hard constraint scales with its codimension relative to d, and the
    pendulum flatters it in a way the 4-D system does not.
    """

    knowledge, enforcement, label = "explicit H", "hard (architecture)", "D: projected"

    def __init__(self, system: System, width=128, depth=3):
        super().__init__()
        self.system = system
        self.net = MLP(system.dim, system.dim, width, depth)

    def forward(self, z):
        g = self.net(z)
        n = self.system.gradH_torch(z)
        denom = (n * n).sum(-1, keepdim=True) + 1e-10
        return g - (g * n).sum(-1, keepdim=True) / denom * n


MODELS = {
    "A": ModelA_Unconstrained,
    "B": ModelB_SoftPenalty,
    "C": ModelC_Hamiltonian,
    "D": ModelD_Projected,
}

#: models whose objective carries the physics penalty
SOFT_MODELS = {"B"}


def make_model(key: str, system: System, width=128, depth=3, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; have {sorted(MODELS)}")
    return MODELS[key](system, width=width, depth=depth)

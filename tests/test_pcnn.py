"""Tests.

Most of these are ordinary sanity checks. Four of them are the load-bearing
ones, because they are what turns a claim in the README into something a reader
can execute:

  test_model_C_conserves_its_own_invariant_exactly
  test_model_D_conserves_the_true_energy_exactly
  test_unconstrained_model_does_not_conserve      (proves the above aren't vacuous)
  test_parameter_budgets_match                    (proves the comparison is fair)

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcnn.data import make_data
from pcnn.integrators import implicit_midpoint_rollout, reference_rollout, rk4_rollout
from pcnn.losses import residual_cosine
from pcnn.models import make_model
from pcnn.systems import REGISTRY, make_system
from pcnn.train import TrainConfig, train

ALL = sorted(REGISTRY)
TOL_EXACT = 1e-5  # float32 "machine precision" on a normalised dot product


def _states(system, n=512, seed=0):
    rng = np.random.default_rng(seed)
    e = (0.2, 1.0) if system.dof == 1 else (0.02, 0.08)
    return torch.tensor(system.sample_states(n, rng, *e), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# systems
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL)
def test_numpy_and_torch_sides_agree(name):
    """The reference integrator and the training code must describe the same
    system, or every number in the study compares a model to the wrong truth."""
    s = make_system(name)
    z = _states(s).double()
    assert np.allclose(s.f_np(z.numpy()), s.f_torch(z).numpy(), atol=1e-12)
    assert np.allclose(s.H_np(z.numpy()), s.H_torch(z).numpy(), atol=1e-12)


@pytest.mark.parametrize("name", ALL)
def test_analytic_gradient_matches_autograd(name):
    s = make_system(name)
    z = _states(s, 64).double().requires_grad_(True)
    (auto,) = torch.autograd.grad(s.H_torch(z).sum(), z)
    assert torch.allclose(auto, s.gradH_torch(z), atol=1e-10)


@pytest.mark.parametrize("name", ALL)
def test_energy_is_conserved_exactly_when_it_should_be(name):
    s = make_system(name)
    z = _states(s).double()
    r = (s.gradH_torch(z) * s.f_torch(z)).sum(-1).abs().max().item()
    if s.is_hamiltonian:
        assert r < 1e-12
    else:
        assert r > 1e-3, "the damped system must actually violate the assumed invariant"


@pytest.mark.parametrize("name", ALL)
def test_sampling_respects_the_energy_shell(name):
    s = make_system(name)
    lo, hi = (0.2, 1.0) if s.dof == 1 else (0.02, 0.08)
    z = s.sample_states(400, np.random.default_rng(1), lo, hi)
    e = s.H_np(z)
    assert e.min() >= lo - 1e-9 and e.max() <= hi + 1e-9


# --------------------------------------------------------------------------- #
# integrators
# --------------------------------------------------------------------------- #


def test_reference_integrator_is_a_trustworthy_yardstick():
    s = make_system("pendulum")
    z0 = _states(s, 16).double().numpy()
    traj = reference_rollout(s, z0, np.linspace(0, 50, 501))
    drift = np.abs(s.H_np(traj) - s.H_np(traj[:, :1])).max()
    assert drift < 1e-9


def test_implicit_midpoint_conserves_quadratic_invariants():
    """Implicit midpoint conserves quadratic first integrals exactly. On the
    harmonic oscillator that is the energy, at any step size -- which is why it
    is the right solver for cashing in a Hamiltonian architecture."""
    f = lambda z: torch.stack([z[..., 1], -z[..., 0]], dim=-1)
    z0 = torch.tensor([[1.0, 0.0], [0.3, 0.9]], dtype=torch.float64)
    e0 = 0.5 * (z0**2).sum(-1)

    mid = implicit_midpoint_rollout(f, z0, 0.4, 1500, n_iter=40)
    rk4 = rk4_rollout(f, z0, 0.4, 1500)
    drift = lambda tr: ((0.5 * (tr**2).sum(-1) - e0[:, None]).abs() / e0[:, None]).max().item()

    assert drift(mid) < 1e-10
    assert drift(mid) < drift(rk4)


# --------------------------------------------------------------------------- #
# the guarantees -- the point of the repo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["pendulum", "henon_heiles"])
def test_model_C_conserves_its_own_invariant_exactly(name):
    """dH_theta/dt = <grad H_theta, J grad H_theta> = 0 by antisymmetry of J.

    Untrained weights on purpose: the guarantee is structural, so it must hold
    before a single gradient step and for every input.
    """
    s = make_system(name)
    m = make_model("C", s, width=32, seed=0)
    z = _states(s, 256).requires_grad_(True)
    f = m(z)
    (gH,) = torch.autograd.grad(m.H(z).sum(), z)
    cos = ((gH * f).sum(-1).abs() / (gH.norm(dim=-1) * f.norm(dim=-1) + 1e-12)).max()
    assert cos.item() < TOL_EXACT


@pytest.mark.parametrize("name", ALL)
def test_model_D_conserves_the_true_energy_exactly(name):
    """<grad H_true, f_theta> = 0 identically, by construction of the projection."""
    s = make_system(name)
    m = make_model("D", s, width=32, seed=0)
    assert residual_cosine(m, s, _states(s, 512)) < TOL_EXACT


@pytest.mark.parametrize("name", ["pendulum", "henon_heiles"])
def test_unconstrained_model_does_not_conserve(name):
    """Guards against a vacuous test above: an unconstrained net at the same
    width violates the law by a wide margin."""
    s = make_system(name)
    m = make_model("A", s, width=32, seed=0)
    assert residual_cosine(m, s, _states(s, 512)) > 1e-2


@pytest.mark.parametrize("name", ["pendulum", "henon_heiles"])
def test_parameter_budgets_match(name):
    """A win that comes from extra capacity is not a win for the constraint."""
    s = make_system(name)
    counts = {k: make_model(k, s, width=128, seed=0).n_parameters() for k in "ABCD"}
    ref = counts["A"]
    for k, v in counts.items():
        assert abs(v - ref) / ref < 0.01, f"model {k}: {v} vs {ref}"


def test_model_C_guarantee_survives_a_no_grad_rollout():
    """The Hamiltonian model differentiates its own input, so it must still work
    inside torch.no_grad() -- otherwise every rollout silently breaks."""
    s = make_system("pendulum")
    m = make_model("C", s, width=32, seed=0).eval()
    with torch.no_grad():
        tr = rk4_rollout(m, _states(s, 8), 0.05, 200)
    assert torch.isfinite(tr).all()


# --------------------------------------------------------------------------- #
# data & training
# --------------------------------------------------------------------------- #


def test_noise_degrades_the_finite_difference_targets():
    s = make_system("pendulum")
    clean = make_data(s, seed=0, n_traj=6, T=3.0, noise=0.0)
    noisy = make_data(s, seed=0, n_traj=6, T=3.0, noise=0.03)
    assert clean.train.target_snr > 10 * noisy.train.target_snr
    # train and extrapolation shells must not overlap, or "extrapolation" is a lie
    assert s.H_torch(noisy.test_out.z).min() > s.H_torch(noisy.test_in.z).max()


@pytest.mark.parametrize("key", list("ABCD"))
def test_training_actually_fits(key):
    s = make_system("pendulum")
    d = make_data(s, seed=0, n_traj=8, T=4.0, noise=0.0)
    m = make_model(key, s, width=32, seed=0)
    before = torch.nn.functional.mse_loss(m(d.val.z), d.val.dz).item()
    r = train(m, s, d, TrainConfig(steps=400, lam=1.0 if key == "B" else 0.0, seed=0))
    assert r.val_loss < 0.25 * before

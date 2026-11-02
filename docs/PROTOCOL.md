# Protocol

Everything a reader needs to decide whether to believe the results, and
everything needed to reproduce them. If a choice below looks arbitrary, the
reason it is written down is so you can argue with it.

## 1. What the models learn

A **vector field**, not a solution:

```
f_theta : R^d -> R^d       approximating    dz/dt = f(z)
```

This is the neural-ODE setting, not the "PINN solves a PDE on a space-time
collocation grid" setting. The choice is deliberate: it is the setting where all
four ways of imposing a conservation law can be written down side by side and
compared with nothing else changing. A conservation law is a statement about a
flow, so putting the flow at the centre keeps the comparison clean. The cost is
that conclusions here transfer to operator-learning / PDE-PINN settings by
analogy, not by proof.

## 2. Systems

| system | d | conserves H? | train shell | unseen shell | rollout horizon |
|---|---|---|---|---|---|
| `pendulum` | 2 | yes | E ∈ [0.2, 1.0] | E ∈ [1.2, 1.8] | T = 60 |
| `henon_heiles` | 4 | yes | E ∈ [0.02, 0.08] | E ∈ [0.10, 0.145] | T = 30 |
| `damped_pendulum` | 2 | **no** (α = 0.15) | E₀ ∈ [0.2, 1.0] | E₀ ∈ [1.2, 1.8] | T = 30 |

Pendulum: `H = p²/2 + (1 − cos q)`, separatrix at E = 2, so the unseen shell is
still librating but visibly closer to the separatrix — a genuinely different
regime, not a rescaling.

Hénon–Heiles: `H = ½(p₁²+p₂²) + ½(q₁²+q₂²) + q₁²q₂ − q₂³/3`, escape energy
E\* = 1/6 ≈ 0.167. Largely regular on the training shell, strongly chaotic on the
unseen one. This is why extrapolation there is hard for reasons that have
nothing to do with network capacity.

**Train and test regions are separated by energy, not by a bounding box.** A box
split lets a model interpolate across the boundary; an energy split does not.

## 3. Data

1. Sample `n_traj` initial conditions inside the training energy shell.
2. Integrate the **true** system with SciPy DOP853 at `rtol = atol = 1e-11`.
3. Sample at `dt = 0.05` over `T = 6`.
4. Add i.i.d. Gaussian noise `σ` to the observed **states**.
5. Estimate targets by **central differences on the noisy states**.

Step 5 is the one that matters. Noise on the state is amplified by
`1/(√2·dt) ≈ 14` on the target, and the input and the target share a noise
realisation (errors-in-variables). Reported per run as `target_snr`:

| σ | 0 | 0.003 | 0.01 | 0.03 |
|---|---|---|---|---|
| target SNR, pendulum | 2618 | 17.6 | 5.3 | 1.8 |
| target SNR, Hénon–Heiles | 1898 | 3.9 | 1.2 | 0.4 |

At σ = 0 the only error is the O(dt²) differencing bias, hence the very high
SNR — that column is the "clean data" control, not a noiseless one. Hénon–Heiles
is harder at equal σ because its vector field is smaller in magnitude: by
σ = 0.03 the targets carry more noise than signal.

Test sets are clean and use analytic derivatives, drawn fresh from each shell.
Collocation points for the soft penalty are unlabelled and drawn from the
**training** shell only — extending them into the unseen shell would leak the
test set.

## 4. Models

See the table in `pcnn/models.py`. All four share the same trunk: tanh MLP,
3 hidden layers, width 128, Xavier-normal init with tanh gain.

| | knows | enforcement | guarantee |
|---|---|---|---|
| A | nothing | none | none |
| B | H analytically | loss penalty | approximate, λ-dependent |
| C | *that* an invariant exists | architecture | `dH_θ/dt = 0` exactly |
| D | H analytically | architecture | `dH_true/dt = 0` exactly |

The fourth cell of the 2×2 — a soft penalty on a *learned* invariant — is not
implemented. It is the obvious extension and is flagged in the README.

## 5. Training budget (identical for every model)

| | |
|---|---|
| optimizer | AdamW, lr 3e-3, cosine to 1 % , weight decay 1e-6 |
| budget | **4000 optimizer steps** (steps, not epochs) |
| batch | 512, sampled with a per-seed generator so batch order matches across models |
| model selection | best **validation** MSE, evaluated every 250 steps |
| validation targets | the **noisy** finite-difference ones |
| parameter parity | C's width is solved for so all four counts agree to < 1 % (asserted in tests) |

Selecting on clean validation data would leak the answer into a study whose
whole subject is noise. Counting epochs instead of steps would hand extra
updates to whichever model has the cheaper forward pass.

## 6. Choosing λ for the soft model

Every model-B entry outside the λ study is trained at λ ∈ {0.1, 1, 10} and the
**best validation loss** wins. Reported alongside the result as `lam_selected`.

- Fixing one λ → straw man.
- Choosing λ by the test metric → oracle.
- Choosing λ by validation → what a practitioner can actually do.

The λ study itself sweeps λ ∈ {1e-3 … 1e3} and reports the whole front.

## 7. Metrics

| name | meaning |
|---|---|
| `vf_mse_in` / `vf_mse_out` | one-step vector-field MSE, clean test set, train / unseen shell |
| `valid_time_rk4` | first time the rollout error exceeds 20 % of the trajectory's own mean norm ‖z‖; median over 24 ICs, capped at the horizon |
| `energy_drift_rk4` | \|H(z_T) − H(z_0)\| / mean\|H(z_0)\|, median over ICs |
| `residual_cos_in` | \|cos(∇H, f_θ)\|, mean. Scale-free physics violation in [0, 1] |
| `blowup_rk4` | fraction of rollouts exceeding ‖z‖ = 10³ |
| `invariant_r2` | R² of the affine fit `H_θ ≈ a·H_true + b` (model C only) |

`residual_cos` is normalised on purpose. The raw penalty `(∇H·f)²` is minimised
by a network that predicts `f ≈ 0` — perfectly "physics-consistent" and useless.
A normalised residual cannot be gamed that way. Only the normalised one is
recorded; the raw one exists only as model B's training loss.

`invariant_r2` uses an **affine** fit because a Hamiltonian is only defined up to
scale and offset; penalising those would be unfair. What remains is the question
that matters: is it the same *function of state*?

## 8. Statistics

5 seeds. A seed fixes the data, the initial conditions, the batch order and the
weight init — **all four models see identical everything**. Reported numbers are
medians with percentile-bootstrap 95 % CIs (4000 resamples). Comparisons against
model A are paired per seed, so seed-to-seed variance cancels.

With 5 seeds we can detect consistent effects, not small ones. Where intervals
overlap, `FINDINGS.md` says so instead of ranking.

## 9. Rollout and the integrator

Everything except the integrator study rolls out with **RK4 at dt = 0.05**, fine
enough that solver truncation error is orders of magnitude below model error —
so every drift number elsewhere is the model's fault, not the solver's.

The integrator study re-rolls the *same trained weights* with RK4 and with
implicit midpoint (symplectic for any Hamiltonian field, unlike leapfrog which
needs a separable H — a learned H is not separable), at a fine step and at a
coarse one (dt = 0.4 pendulum, 0.25 Hénon–Heiles).

## 10. Reproducing

```bash
pip install -e ".[dev]"
pytest -q                                   # 29 tests (~10 s), incl. the guarantees
python scripts/demo.py                      # 60 s, one system, one seed
python scripts/run_experiments.py --quick   # ~5 min reduced grid
make grid SHARDS=8                          # the full 560-run study
python scripts/make_figures.py
```

The grid is sharded, append-only and resumable: re-running a shard skips jobs
already in its JSONL, provided the shard count is unchanged (the committed
results used 20 shards). Raw per-run records live in `results/raw_shard*.jsonl`,
aggregated tables in `results/table_*.csv`.

## 11. Determinism and machine dependence

Every source of randomness is seeded (data, initial conditions, weight init,
batch order). On a given machine and software stack a rerun is **bit-for-bit
identical** — verified by rerunning recorded jobs with 1 and 4 threads.

Across machines it is not, for three compounding reasons:

1. **Float addition is not associative.** The summation order inside a matrix
   product depends on the CPU instruction set (SSE / AVX2 / AVX-512), the BLAS
   library (MKL / OpenBLAS / Accelerate) and the PyTorch version, so the last
   bit of every layer output can differ.
2. **Training amplifies it.** 4000 nonlinear optimiser steps turn a last-bit
   difference into slightly different weights.
3. **Chaos amplifies it again.** Hénon–Heiles rollouts diverge exponentially,
   so valid times there are the most sensitive quantity.

Separately, model D's energy drift (~5e-7) *is* the float32 round-off floor,
so its exact value is a property of the hardware rather than of the model.

Measured: forcing MKL onto SSE4.2 (`MKL_CBWR=SSE4_2`) changed losses and errors
by 1e-7 to 4e-6 relative, left valid times unchanged, and moved D's drift from
6.5e-7 to 1.1e-6. The study's conclusions rest on ratios between 1.5× and 10⁶×;
they should reproduce anywhere, the last digits should not be expected to.

Committed results: PyTorch 2.7.1, CPU, x86-64 with AVX-512 and MKL, one thread
per shard, 20 shards.

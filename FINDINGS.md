# Findings

Long form. Every number is a median over 5 seeds; brackets are percentile-bootstrap
95 % CIs. `k/5` is a **paired** win count against baseline A — same data, same
initial conditions, same batch order, same init, so seed variance cancels.

Where intervals overlap, this document says so instead of ranking. That happens
more often than the headline figure suggests, and it is the reason the paired
counts are reported alongside.

Reference setting unless stated: 40 trajectories, σ = 0.01, derivative SNR 5.3
(pendulum) / 1.2 (Hénon–Heiles), 4000 optimizer steps.

---

## 0. The reference table

### Nonlinear pendulum (d = 2)

| metric | A · none | B · soft | C · Hamiltonian | D · projected |
|---|---|---|---|---|
| one-step error, in-dist | 6.0e-5 `[4.5, 7.6]e-5` | 5.9e-5 `[4.5e-5, 2.1e-4]` | 4.5e-5 `[3.9, 8.1]e-5` | **4.4e-5** `[3.5e-5, 1.1e-4]` |
| one-step error, unseen shell | 5.1e-3 `[3.5, 5.5]e-3` | 6.1e-3 `[4.0e-3, 1.2e-2]` | 5.6e-3 `[4.9, 7.2]e-3` | **3.5e-3** `[2.6, 5.7]e-3` |
| valid time (cap 60) | 35.3 `[20.3, 54.3]` | 38.3 `[30.2, 42.8]` | **60.0** `[54.2, 60]` | **60.0** `[54.6, 60]` |
| energy drift | 0.129 `[0.072, 0.421]` | 0.136 `[0.079, 0.184]` | 6.2e-3 `[2.2, 7.7]e-3` | **6.5e-7** `[4.9e-7, 1.0e-6]` |
| residual \|cos(∇H,f)\| | 5.6e-3 `[4.8, 6.6]e-3` | 2.5e-3 `[1.1, 8.7]e-3` | 4.9e-3 `[3.9, 6.0]e-3` | **1.8e-8** `[1.5, 2.1]e-8` |
| R² of learned invariant | — | — | 0.99963 | — |

### Hénon–Heiles (d = 4, chaotic)

| metric | A · none | B · soft | C · Hamiltonian | D · projected |
|---|---|---|---|---|
| one-step error, in-dist | 5.2e-5 `[4.5, 9.6]e-5` | **2.9e-5** `[2.7, 3.7]e-5` | 3.8e-5 `[3.5, 9.4]e-5` | 5.6e-5 `[4.1e-5, 1.1e-4]` |
| one-step error, unseen shell | 6.4e-4 `[5.4e-4, 1.2e-3]` | **4.4e-4** `[4.2, 5.0]e-4` | 6.4e-4 `[5.0, 7.4]e-4` | 6.4e-4 `[5.4e-4, 1.4e-3]` |
| valid time (cap 30) | 11.3 `[7.0, 16.5]` | **12.3** `[7.9, 22.6]` | 9.4 `[7.9, 12.2]` | 8.4 `[6.4, 14.2]` |
| energy drift | 0.318 `[0.132, 0.366]` | 0.121 `[0.048, 0.153]` | 0.015 `[0.010, 0.064]` | **4.8e-7** `[3.7, 7.1]e-7` |
| residual \|cos(∇H,f)\| | 1.9e-2 `[1.5, 2.1]e-2` | 7.4e-3 `[5.9, 7.8]e-3` | 1.1e-2 `[1.1, 2.2]e-2` | **1.6e-8** `[1.5, 1.7]e-8` |
| R² of learned invariant | — | — | 0.99577 | — |

**Read the one-step-error rows first, because they are the null result that
frames everything else.** On the pendulum all four CIs overlap: no model has a
detectable one-step accuracy advantage. On Hénon–Heiles only B separates from A
(`[2.7, 3.7]e-5` vs `[4.5, 9.6]e-5`). Physics constraints, of either kind, are
**not** primarily an accuracy technology at this data scale. What separates them
is everything downstream of one step.

### Paired comparison against A (5 seeds)

| system | model | metric | ratio vs A | wins |
|---|---|---|---|---|
| pendulum | C | valid time | **1.70×** | 5/5 |
| pendulum | D | valid time | **1.55×** | 5/5 |
| pendulum | B | valid time | 1.09× | 3/5 |
| pendulum | C | energy drift | **0.033×** | 5/5 |
| pendulum | D | energy drift | **0.000×** | 5/5 |
| pendulum | B | energy drift | 1.05× | 2/5 |
| Hénon–Heiles | B | one-step error | **0.55×** | 5/5 |
| Hénon–Heiles | C | one-step error | **0.71×** | 5/5 |
| Hénon–Heiles | D | one-step error | 1.10× | 2/5 |
| Hénon–Heiles | B | energy drift | **0.36×** | 5/5 |
| Hénon–Heiles | C | energy drift | **0.074×** | 5/5 |
| Hénon–Heiles | D | energy drift | **0.000×** | 5/5 |
| Hénon–Heiles | C | valid time | 0.83× | **1/5** |
| Hénon–Heiles | D | valid time | 0.83× | 2/5 |

The two rows that matter most are the last two. In 4 dimensions, the models with
an exact conservation guarantee track the true trajectory *worse* than the model
with no physics at all — consistently, not by chance (C loses on 4 of 5 seeds).

---

## 1. Conservation and accuracy are orthogonal

D's energy drift on Hénon–Heiles is `4.8e-7` against A's `0.318`: six orders of
magnitude, 5/5 seeds, no overlap. D's valid time is `8.4` against A's `11.3`.

A trajectory can stay exactly on its energy shell and still be in the wrong place
on it. In a chaotic system that is the normal case: the constraint fixes one
scalar out of four state dimensions and says nothing about the phase.

**Consequence for practice.** Energy drift is the metric everyone plots because
it is the metric a conservation prior is guaranteed to win. It is not evidence
that the model predicts better. Report a trajectory metric next to it, or the
plot is decoration.

---

## 2. What the soft penalty actually does

### It does not deliver conservation

Pendulum, λ chosen by validation loss:

| | A | B |
|---|---|---|
| residual | 5.6e-3 | 2.5e-3 (2.3× better) |
| energy drift over T=60 | 0.129 | 0.136 (**2/5 seeds**) |

The penalty is applied pointwise at collocation points, and it does reduce the
pointwise violation. Over a long rollout the reduction does not accumulate into
anything. The hard constraints, on the same system, reduce drift by 21× (C) and
198 000× (D).

On Hénon–Heiles the penalty does help drift (`0.36×`, 5/5) — but still lands two
orders of magnitude short of C and six short of D.

### It does deliver regularisation

This is the part that a "hard constraints win" narrative would suppress. On
Hénon–Heiles, B has the best one-step error (`0.55×` A, 5/5) **and** the best
extrapolation error (`4.4e-4` vs `6.4e-4` for every other model), and is the only
model whose CI separates from A's.

Against finite-difference targets with SNR 1.2, the residual term acts as a
smoothness prior. That is a real, reproducible benefit. It is simply not the
benefit "physics-informed" is usually sold on.

### The λ front

Pendulum, full sweep, medians over 5 seeds:

| λ | one-step error | residual | energy drift |
|---|---|---|---|
| 1e-3 | 6.1e-5 | 6.0e-3 | 0.101 |
| 1e-2 | 6.1e-5 | 5.9e-3 | 0.100 |
| 1e-1 | 5.4e-5 | 5.5e-3 | 0.279 |
| 1 | 1.0e-4 | 6.4e-3 | 0.124 |
| 10 | 5.5e-5 | 2.0e-3 | 0.123 |
| 100 | 5.2e-4 | 9.5e-4 | 6.4e-3 |
| 1000 | 4.4e-3 | 8.0e-4 | 1.8e-2 |
| **D** | **4.4e-5** | **1.8e-8** | **6.5e-7** |

To reach C's energy drift the penalty needs λ = 100, and pays 9× in one-step
error and 3.5× in extrapolation error. To approach D's residual it would need
λ → ∞. D is not a point on this front; it is off it.

On Hénon–Heiles the front is friendlier — λ = 10 improves accuracy *and* residual
*and* drift simultaneously. **One system is not a benchmark**: the pendulum alone
would have said "λ is a painful trade-off", Hénon–Heiles alone would have said
"λ is free". Both are true, of different systems.

---

## 3. The value of a hard constraint scales with its codimension

The projection removes one direction out of `d`. What is left to learn is a
`(d−1)`-dimensional field.

| | d | free directions after constraint | D's valid time vs A |
|---|---|---|---|
| pendulum | 2 | 1 | **1.55×**, 5/5 |
| Hénon–Heiles | 4 | 3 | 0.83×, 2/5 |

In `d = 2` the tangent space of an energy level set is one-dimensional, so D can
only learn a scalar speed along the true flow. That is nearly the answer for
free, and D behaves accordingly — it hits the rollout-horizon cap. In `d = 4` the
same constraint leaves three-quarters of the directions free and the prior stops
carrying the model.

C follows the same pattern (`1.70×` → `0.83×`).

**This is why single-pendulum demos are misleading.** A structural conservation
prior looks spectacular at `d = 2` for a reason that has nothing to do with
conservation being a good prior in general, and everything to do with there being
almost nothing left to learn. Two points do not establish a law, but they
establish that the effect is not dimension-free.

---

## 4. Model C conserves something exactly — and it drifts away from the energy

C's guarantee is structural: `⟨∇H_θ, J∇H_θ⟩ = 0` by antisymmetry of `J`, at
machine precision, for untrained weights and every input
(`tests/test_pcnn.py::test_model_C_conserves_its_own_invariant_exactly`).

What is *not* guaranteed is that `H_θ` is the energy. R² of the affine fit
`H_θ ≈ a·H_true + b`:

| Hénon–Heiles | σ=0 | σ=0.003 | σ=0.01 | σ=0.03 |
|---|---|---|---|---|
| R² | 0.99975 | 0.99914 | 0.99577 | **0.936** |

| Hénon–Heiles | 5 traj | 10 | 20 | 40 |
|---|---|---|---|---|
| R² | **0.906** | 0.982 | 0.995 | 0.996 |

The guarantee's *precision* is constant; its *referent* degrades exactly when the
problem gets hard. And measured against the physical energy, C is not better than
having no prior at all:

| residual vs **true** H, Hénon–Heiles | A | B | C | D |
|---|---|---|---|---|
| | 1.9e-2 | 7.4e-3 | 1.1e-2 | 1.6e-8 |

C's CI `[1.1e-2, 2.2e-2]` overlaps A's `[1.5e-2, 2.1e-2]`. **A model advertised as
conserving energy is, with respect to the actual energy, statistically
indistinguishable from a model with no physics in it.** It conserves *an*
invariant perfectly. Whether that is worth anything depends on what you wanted
the guarantee for.

If you need the physical energy conserved, C is the wrong tool and D is the right
one. If you only need *some* invariant — long-term boundedness, no secular blowup
— C delivers, and without needing `H` in closed form.

---

## 5. The structural prior is a small-data effect, and it reverses

Hénon–Heiles, valid time:

| trajectories | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| A | 1.65 | 3.05 | 5.45 | **11.28** |
| C | **2.38** | **6.15** | **10.88** | 9.40 |

One-step error, same axis:

| trajectories | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| A | 1.38e-3 | 3.0e-4 | 1.54e-4 | **5.2e-5** |
| C | **3.2e-4** | **1.3e-4** | **4.3e-5** | 3.8e-5 |

At n = 5, C is 4.3× more accurate and 1.4× more stable. At n = 40 it has lost the
stability advantage entirely. The same reversal appears on the noise axis: with
**clean** data (σ = 0) C is the *worst* model on Hénon–Heiles — valid time 15.3
against 30.0 (the cap) for A, B and D, and one-step error 1.1e-5 against 2.6–3.1e-6.

A hard architectural prior is a **bias**. When the data is scarce or noisy the
bias pays for itself; when the data is clean and plentiful you are paying for a
constraint you no longer need, and the bill shows up as accuracy.

**Consequence for practice.** A structural-prior result reported at one dataset
size is a report of where the author happened to stand on this curve. Both the
"structural priors win" and "structural priors are overrated" papers can be
generated from this single sweep by choosing n.

---

## 6. Noise: the constraints pay in one corner of the space, not everywhere

Targets are central differences on noisy states, so σ on the state lands as ≈14σ
on the target. Derivative SNR (median over seeds): pendulum 2618 / 17.6 / 5.3 / 1.8 and
Hénon–Heiles 1898 / 3.9 / 1.2 / 0.4 for σ = 0 / 0.003 / 0.01 / 0.03.

### Hénon–Heiles, one-step error

| σ | A | B | C | D |
|---|---|---|---|---|
| 0 | 3.0e-6 | 3.1e-6 | 1.1e-5 | **2.6e-6** |
| 0.003 | 1.2e-5 | **9.8e-6** | 1.5e-5 | 1.0e-5 |
| 0.01 | 5.2e-5 | **2.9e-5** | 3.8e-5 | 5.6e-5 |
| 0.03 | 9.3e-4 | **4.0e-4** | 4.1e-4 | 5.2e-4 |

### Hénon–Heiles, energy drift

| σ | A | B | C | D |
|---|---|---|---|---|
| 0 | 0.034 | 0.029 | 2.9e-3 | **5.4e-7** |
| 0.01 | 0.318 | 0.121 | 0.015 | **4.8e-7** |
| 0.03 | **3.78** | 0.481 | 0.046 | **4.5e-7** |

With clean data, D is also the most accurate model one step ahead: it beats A on
5/5 seeds at σ = 0 on both systems (`0.84×` here, `0.66×` on the pendulum). The
exact constraint costs nothing when the data is good, *provided you know H*.

At σ = 0.03 the unconstrained model does not merely degrade: its rollout ends
with nearly four times the initial energy. C stays 82× closer to physical, and
B 7.9× — the constraints are doing real work here.

### The same noise on the pendulum

| σ = 0.03, pendulum | A | B | C | D |
|---|---|---|---|---|
| one-step error | **2.9e-4** | 3.6e-4 | 4.6e-4 | 3.9e-4 |
| valid time (cap 60) | **32.2** | 11.1 | 14.0 | 22.8 |

Here the unconstrained model is the *most* accurate and the *most* stable, and
the Hamiltonian model is the worst on both.

And with clean data on Hénon–Heiles, C is the worst model in the entire study:
valid time **15.3** against 30.0 (the cap) for A, B and D, one-step error 3.6–4.4×
the others.

**The constraint pays when the data is bad *and* the system is hard.** Neither
condition alone suffices. In the other three quadrants you are paying for a bias
you do not need, and finding 5's reversal is the same effect seen along the data
axis instead of the noise axis.

---

## 7. Misspecification: the ranking inverts and the metric changes sign

Damped pendulum, `p' = −sin q − 0.15 p`. Not Hamiltonian. All models are still
handed the undamped energy and told to conserve it. Over the T = 30 horizon the
**true** system dissipates ~99 % of each trajectory's initial energy. Measured
with the same drift metric as the models (median over initial conditions of
`|H(z_T) − H(z_0)| / mean H(z_0)`), the truth scores **≈ 0.94** (median over
seeds; 0.86–1.01 across seeds).

| | A | B | C | D |
|---|---|---|---|---|
| energy "drift" (truth ≈ 0.94) | **0.939** | 0.932 | 0.033 | 6e-7 |
| valid time (cap 30) | **30.0** | 10.6 | 1.1 | 0.9 |
| one-step error | **1.1e-4** | 2.1e-3 | 4.3e-3 | 7.7e-3 |
| extrapolation error | **6.9e-3** | 3.3e-2 | 2.3e-2 | 2.2e-2 |

The energy-drift row is the same column that made D look six orders of magnitude
better than everyone else in every other experiment. Here the correct value is
**≈ 0.94**, and D scores **6e-7** — it is not merely wrong, it is maximally wrong,
and it is wrong *by construction*, so no amount of data can fix it. A scores 0.94,
i.e. it reproduces the true dissipation, and holds the trajectory for the entire
horizon.

Valid time collapses 33× for D and 27× for C.

**B is the only model that degrades gracefully**, and only because λ is a dial: at
the validation-selected λ = 0.1 it keeps a usable 10.6 valid time. Push λ up and
it converges to C and D's failure; push it down and it converges to A. That
tunability is the actual argument for soft constraints, and it is invisible in
any experiment where the assumed physics is correct.

**Consequence for practice.** A hard constraint is a bet that your physics is
exactly right. It has no failure mode between "correct" and "catastrophic". A
soft penalty is a bet with a hedge whose size you choose. Nearly all published
comparisons run only the experiment where the bet pays.

---

## 8. Negative result: the symplectic integrator did not matter

Common claim: a Hamiltonian architecture needs a symplectic integrator, or the
solver spends the guarantee. Same trained weights, four rollouts — RK4 and
implicit midpoint, at a fine step (dt = 0.05) and a coarse one (dt = 0.4 / 0.25).

Energy drift, pendulum:

| model | RK4 fine | midpoint fine | RK4 coarse | midpoint coarse |
|---|---|---|---|---|
| A | 1.29e-1 | 1.29e-1 | 1.33e-1 | 1.28e-1 |
| B | 1.24e-1 | 1.23e-1 | 1.28e-1 | 1.08e-1 |
| C | 6.23e-3 | 6.16e-3 | 5.80e-3 | 5.84e-3 |
| D | **6.55e-7** | 2.41e-5 | 5.43e-3 | 1.83e-3 |

(B is fixed at λ = 1 in this study rather than tuned, which is why its drift
differs from the reference table.)

For A, B and C the solver changes the answer by about 1 % at the fine step and by
at most 16 % at the coarse one (B, pendulum): **the drift is model error, not
solver error**, and no integrator can fix a vector field that is already wrong
by more than the discretisation. Only D is precise enough for the solver to
become visible — and at the fine step, plain RK4 wins (`6.6e-7` vs `2.4e-5`). At
the coarse step midpoint wins on the pendulum (3×) and *loses* on Hénon–Heiles
(`1.0e-3` vs `5.5e-4`); on the pendulum, coarse midpoint also halves C's and D's
valid time (60 → ~30), whereas coarse RK4 keeps them at the cap. The coarse step
mostly hurts D's conservation; the solver choice redistributes the damage rather
than removing it.

**Why midpoint loses for D at the fine step.** Implicit midpoint is symplectic
but only second-order. Symplecticity keeps the energy error of a non-quadratic
`H` (the pendulum's `1 − cos q`) *bounded*, not zero — at O(dt²) — whereas RK4's
error is O(dt⁴). We checked that this is truncation and not an unconverged
solve: on seed 0, 12 and 40 fixed-point iterations give the same drift
(`4.5e-5`) to every printed digit, and halving dt divides it by 3.9 ≈ 2². A
Newton solve would not change it. (Implicit midpoint conserves *quadratic*
invariants exactly, which is what `test_implicit_midpoint_conserves_quadratic_invariants`
checks on the harmonic oscillator; the pendulum energy is not quadratic.)

The defensible version of the folklore: *the integrator matters once your model
error drops below your discretisation error.* For three of four models here, it
never does.

---

## Threats to validity

- **Two dimensionalities.** The codimension argument rests on `d = 2` and `d = 4`.
  It predicts the trend; it does not measure a slope.
- **5 seeds.** Enough for consistent effects (the 5/5 rows), not for small ones.
  Every "no difference" here means "not detectable at n = 5", not "equal".
- **λ over three values** outside the dedicated sweep. A finer grid could improve
  B's numbers; it would not create the conservation it fails to deliver, since
  the full sweep shows what the whole front looks like.
- **One invariant, exactly known.** D is given `H` in closed form. Partial or
  approximate knowledge of the invariant is the realistic case and is untested.
- **Neural-ODE setting.** Vector-field regression, not PDE collocation. Findings
  about *penalty vs architecture* should transfer in spirit; nothing here
  measures a space-time PINN.
- **Fixed trunk.** All results at width 128, depth 3, tanh. A capacity sweep
  might move the point at which the structural prior stops paying (finding 5),
  since that point is about the bias/variance balance.
- **Machine-dependent last digits.** Runs are bit-for-bit reproducible on one
  machine but not across CPUs, BLAS libraries or PyTorch versions, because the
  order of floating-point sums changes and training (and Hénon–Heiles chaos)
  amplifies the difference. Forcing a different instruction set moved errors by
  ≤ 4e-6 relative and D's float32-floor drift from `6.5e-7` to `1.1e-6`; no
  comparison above is anywhere near that scale. See the README section "Your
  numbers will differ in the last digits".

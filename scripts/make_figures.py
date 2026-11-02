"""Build every figure in `figures/` from `results/`.

    python scripts/make_figures.py

Aggregate figures come from the sharded grid. The four figures that need actual
weights (phase portraits, energy curves, the learned-invariant map, the
misspecification portraits) retrain the reference configuration at seed 0 and
cache the weights under `results/models/`, so re-plotting is instant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcnn import analysis, plotting as P  # noqa: E402
from pcnn.data import make_data  # noqa: E402
from pcnn.experiments import ARCH, DT_ROLL, LAMBDA_TUNE, REFERENCE, STEPS, SYSTEM_CFG  # noqa: E402
from pcnn.integrators import ROLLOUTS  # noqa: E402
from pcnn.metrics import make_reference  # noqa: E402
from pcnn.models import make_model  # noqa: E402
from pcnn.systems import make_system  # noqa: E402
from pcnn.train import TrainConfig, train  # noqa: E402

SEED = 0
MODEL_DIR = ROOT / "results" / "models"
FIG = str(ROOT / "figures")


# --------------------------------------------------------------------------- #
# reference models (cached)
# --------------------------------------------------------------------------- #


def reference_models(sysname: str):
    system = make_system(sysname)
    scfg = SYSTEM_CFG[sysname]
    data = make_data(
        system, seed=SEED, n_traj=REFERENCE["n_traj"], T=REFERENCE["T"], dt=REFERENCE["dt"],
        noise=REFERENCE["noise"], e_train=scfg["e_train"], e_extrap=scfg["e_extrap"],
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models, meta_path = {}, MODEL_DIR / f"{sysname}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for key in P.ORDER:
        ckpt = MODEL_DIR / f"{sysname}_{key}.pt"
        m = make_model(key, system, seed=SEED, **ARCH)
        if ckpt.exists():
            m.load_state_dict(torch.load(ckpt, weights_only=True))
            m.eval()
        else:
            if key == "B":
                # honest tuning: best validation loss over the candidate lambdas
                best = None
                for lam in LAMBDA_TUNE:
                    cand = make_model(key, system, seed=SEED, **ARCH)
                    r = train(cand, system, data, TrainConfig(steps=STEPS, lam=lam, seed=SEED))
                    if best is None or r.val_loss < best[0]:
                        best = (r.val_loss, lam, cand)
                _, meta["lam_B"], m = best
            else:
                train(m, system, data, TrainConfig(steps=STEPS, lam=0.0, seed=SEED))
            torch.save(m.state_dict(), ckpt)
        models[key] = m
    meta_path.write_text(json.dumps(meta))
    return system, data, models, meta


@torch.no_grad()
def rollouts(system, models, ic, T, dt, integrator="rk4"):
    ref = make_reference(system, ic, T, dt)
    truth = ref["traj"]
    preds = {
        k: ROLLOUTS[integrator](m, truth[:, 0, :], dt, truth.shape[1] - 1)
        for k, m in models.items()
    }
    return ref["t"], truth, preds


# --------------------------------------------------------------------------- #


def main():
    P.use_style()
    tables = analysis.build_tables(str(ROOT / "results"), str(ROOT / "results"))
    systems = ["pendulum", "henon_heiles"]

    # ---- 1. headline ------------------------------------------------------
    P.fig_headline(tables["headline"], systems, FIG)

    # ---- aggregate sweeps -------------------------------------------------
    P.fig_sweep(
        tables["noise"], systems, "noise",
        [("vf_mse_in", "one-step error (in-dist)", True),
         ("energy_drift_rk4", "energy drift", True),
         ("valid_time_rk4", "valid rollout time", False)],
        "Sensitivity to observation noise",
        "Derivatives are estimated by central differences from noisy states, so noise on the "
        "state is amplified ~14x on the target. Bands are bootstrap 95% CI over 5 seeds. "
        "Note the middle row: the gap the constraints open is in conservation, not accuracy.",
        "fig4_noise.png", xlabel="observation noise σ on the state", outdir=FIG,
    )
    P.fig_sweep(
        tables["data"], systems, "n_traj",
        [("vf_mse_in", "one-step error (in-dist)", True),
         ("vf_mse_out", "one-step error (unseen shell)", True),
         ("valid_time_rk4", "valid rollout time", False)],
        "Data efficiency",
        "Number of training trajectories, at fixed noise σ = 0.01. A structural prior should pay "
        "off most when data is scarce. On Hénon–Heiles it does — and then the advantage "
        "evaporates and reverses by 40 trajectories.",
        "fig5_data_efficiency.png", logx=True, xlabel="training trajectories", outdir=FIG,
    )

    # ---- 6. lambda Pareto -------------------------------------------------
    head = tables["headline"]
    ref_points, selected = {}, {}
    for s in systems:
        ref_points[s] = {
            k: (P._pick(head, s, k, "residual_cos_in")[0], P._pick(head, s, k, "vf_mse_in")[0])
            for k in ("A", "D")
        }
        sel = tables["all_runs"]
        sel = sel[(sel.exp == "noise") & (sel.system == s) & (sel.model == "B")
                  & (sel.noise == REFERENCE["noise"]) & (sel.n_traj == REFERENCE["n_traj"])]
        selected[s] = float(sel["lam"].median()) if len(sel) else None
    P.fig_lambda(tables["lambda"], systems, ref_points, selected, FIG)

    # ---- 9. integrator ----------------------------------------------------
    P.fig_integrator(tables["integ"], systems, DT_ROLL,
                     {s: SYSTEM_CFG[s]["dt_coarse"] for s in systems}, FIG)

    # ---- model-dependent figures -----------------------------------------
    residuals, grids = {}, None
    energy_curves = {}

    for sysname in systems + ["damped_pendulum"]:
        system, data, models, meta = reference_models(sysname)
        scfg = SYSTEM_CFG[sysname]
        T = scfg["T_roll"]

        t, truth, preds = rollouts(system, models, data.test_ic[:8], T, DT_ROLL)

        if sysname == "pendulum":
            P.fig_phase_portraits(truth.numpy(), {k: v.numpy() for k, v in preds.items()},
                                  FIG, "fig2_phase_portraits.png", sysname)
        if sysname == "damped_pendulum":
            P.fig_phase_portraits(
                truth.numpy(), {k: v.numpy() for k, v in preds.items()},
                FIG, "fig8_misspecification.png", sysname,
                extra_note="The true system spirals into the origin. Models C and D are "
                           "architecturally incapable of losing energy, so they orbit forever — "
                           "a guarantee is only as good as the assumption behind it.",
            )
            P.fig_headline(
                tables["misspec"], ["damped_pendulum"], FIG, "fig8b_misspec_scores.png",
                title="Same four models, on a system that does not conserve energy",
                sub="The physics prior is now false. Everything that made C and D win before "
                    "makes them lose here — and the soft penalty degrades gracefully in between.",
            )

        if sysname in systems:
            e0 = system.H_torch(truth[:, 0, :])
            scale = e0.abs().mean().clamp_min(1e-8)
            energy_curves[sysname] = (
                t,
                {k: ((system.H_torch(v) - e0[:, None]).abs() / scale).median(dim=0).values.numpy()
                 for k, v in preds.items()},
            )

            # What invariant did C learn? Fit H_theta ~ a*H_true + b on the
            # TRAINING shell, then apply that same calibration to the unseen
            # shell: the residual then shows how the learned invariant drifts
            # off-shell instead of hiding it behind a refit.
            zc, zo = data.test_in.z, data.test_out.z
            h_hat_in = models["C"].H(zc).detach().numpy()
            h_hat_out = models["C"].H(zo).detach().numpy()
            h_true_in = system.H_torch(zc).numpy()
            h_true_out = system.H_torch(zo).numpy()

            def _fit(h_true, h_hat):
                M = np.stack([h_true, np.ones_like(h_true)], 1)
                coef, *_ = np.linalg.lstsq(M, h_hat, rcond=None)
                ss_res = ((h_hat - M @ coef) ** 2).sum()
                ss_tot = ((h_hat - h_hat.mean()) ** 2).sum()
                return coef, float(1 - ss_res / ss_tot)

            (a, b), r2 = _fit(h_true_in, h_hat_in)
            _, r2_out = _fit(h_true_out, h_hat_out)  # refit, matching metrics.py
            rescale = lambda h: (h - b) / a
            residuals[sysname] = dict(
                h_in=h_true_in, r_in=rescale(h_hat_in) - h_true_in,
                h_out=h_true_out, r_out=rescale(h_hat_out) - h_true_out,
                r2=r2, r2_out=r2_out,
            )

            if sysname == "pendulum":
                q = np.linspace(-np.pi, np.pi, 220)
                p = np.linspace(-2.2, 2.2, 220)
                Q, Pg = np.meshgrid(q, p)
                Z = torch.tensor(np.stack([Q.ravel(), Pg.ravel()], 1), dtype=torch.float32)
                Ht = system.H_torch(Z).numpy().reshape(Q.shape)
                Hl = rescale(models["C"].H(Z).detach().numpy()).reshape(Q.shape)
                grids = (Q, Pg, Ht, Hl)

    P.fig_energy_curves(energy_curves, FIG)
    P.fig_invariant(grids, residuals, FIG)
    print("\nall figures written to", FIG)


if __name__ == "__main__":
    main()

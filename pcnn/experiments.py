"""The experiment grid.

Five studies, one job list. Every job is a fully-specified dict, so the grid is
reproducible, shardable across processes, and resumable -- and so a reader can
see the entire protocol in one place instead of reverse-engineering it from
plotting code.

  noise    error vs observation noise (the headline comparison lives here,
           at the reference noise level)
  data     error vs number of training trajectories
  lambda   the soft model's Pareto front: data fit against physics residual
  misspec  the same protocol on a system that does NOT conserve the energy
           the models are told to conserve
  integ    RK4 vs symplectic rollout of the *same* trained weights

Protocol note on model B. Reporting a PINN at one arbitrary lambda makes it a
straw man. Every B entry outside the lambda study is trained at three lambdas
and the one with the best **validation** loss is kept -- validation on the
noisy targets, i.e. the only signal a practitioner actually has. Selecting on
clean data would leak the answer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .data import make_data
from .metrics import evaluate, make_reference, rollout_scores
from .models import make_model
from .systems import make_system
from .train import TrainConfig, train

# --------------------------------------------------------------------------- #
# protocol constants
# --------------------------------------------------------------------------- #

REFERENCE = dict(n_traj=40, T=6.0, dt=0.05, noise=0.01)
ARCH = dict(width=128, depth=3)
STEPS = 4000
SEEDS = [0, 1, 2, 3, 4]

#: lambdas tried for every model-B entry; best-by-validation is kept
LAMBDA_TUNE = [0.1, 1.0, 10.0]
#: full sweep for the Pareto-front study
LAMBDA_SWEEP = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

NOISES = [0.0, 0.003, 0.01, 0.03]
N_TRAJS = [5, 10, 20]

SYSTEM_CFG = {
    "pendulum": dict(e_train=(0.2, 1.0), e_extrap=(1.2, 1.8), T_roll=60.0, dt_coarse=0.4),
    "henon_heiles": dict(e_train=(0.02, 0.08), e_extrap=(0.10, 0.145), T_roll=30.0, dt_coarse=0.25),
    "damped_pendulum": dict(e_train=(0.2, 1.0), e_extrap=(1.2, 1.8), T_roll=30.0, dt_coarse=0.4),
}
#: rollout step used everywhere except the integrator study. Fine enough that
#: RK4 truncation error is far below model error -- which is the point: it means
#: every drift number elsewhere in the study is the *model's* fault, not the
#: solver's.
DT_ROLL = 0.05
MAIN_SYSTEMS = ["pendulum", "henon_heiles"]


def _quick(q: bool):
    if not q:
        return SEEDS, STEPS, NOISES, N_TRAJS, LAMBDA_SWEEP
    return [0, 1], 1200, [0.0, 0.01], [10], [1e-2, 1.0, 100.0]


# --------------------------------------------------------------------------- #
# job list
# --------------------------------------------------------------------------- #


def jobs(quick: bool = False) -> list[dict]:
    seeds, steps, noises, ntrajs, lam_sweep = _quick(quick)
    out: list[dict] = []

    def add(exp, system, model, seed, lam=0.0, **over):
        cfg = dict(REFERENCE)
        cfg.update(over)
        out.append(
            dict(exp=exp, system=system, model=model, seed=seed, lam=lam, steps=steps, **cfg)
        )

    def model_block(exp, system, seed, **over):
        for k in ("A", "C", "D"):
            add(exp, system, k, seed, 0.0, **over)
        for lam in LAMBDA_TUNE:
            add(exp, system, "B", seed, lam, **over)

    for s in MAIN_SYSTEMS:
        for seed in seeds:
            for nz in noises:
                model_block("noise", s, seed, noise=nz)
            for nt in ntrajs:
                model_block("data", s, seed, n_traj=nt)
            for lam in lam_sweep:
                add("lambda", s, "B", seed, lam)
            for k in ("A", "C", "D"):
                add("integ", s, k, seed, 0.0)
            add("integ", s, "B", seed, 1.0)

    for seed in seeds:
        model_block("misspec", "damped_pendulum", seed)

    return out


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def _data_key(j):
    return (j["system"], j["seed"], j["noise"], j["n_traj"], j["T"], j["dt"])


def run_job(job: dict, cache: dict) -> dict:
    sysname = job["system"]
    scfg = SYSTEM_CFG[sysname]
    system = make_system(sysname)

    key = _data_key(job)
    if cache.get("key") != key:
        data = make_data(
            system,
            seed=job["seed"],
            n_traj=job["n_traj"],
            T=job["T"],
            dt=job["dt"],
            noise=job["noise"],
            e_train=scfg["e_train"],
            e_extrap=scfg["e_extrap"],
        )
        cache.clear()
        cache.update(
            key=key,
            data=data,
            ref_in=make_reference(system, data.test_ic, scfg["T_roll"], DT_ROLL),
            ref_out=make_reference(system, data.test_ic_out, scfg["T_roll"], DT_ROLL),
        )
    data, ref_in, ref_out = cache["data"], cache["ref_in"], cache["ref_out"]

    model = make_model(job["model"], system, seed=job["seed"], **ARCH)
    res = train(
        model,
        system,
        data,
        TrainConfig(steps=job["steps"], lam=job["lam"], seed=job["seed"]),
    )

    # the symplectic comparison is expensive, so only the integrator study pays
    integrators = ("rk4", "midpoint") if job["exp"] == "integ" else ("rk4",)
    ev = evaluate(model, system, data, ref_in, ref_out, integrators=integrators)

    if job["exp"] == "integ":
        # Same weights, coarse step. A structural guarantee lives in the
        # continuous-time field; whether it survives discretisation is a
        # property of the solver. At DT_ROLL the two solvers agree, so the
        # comparison is only meaningful once the step is genuinely coarse.
        if "ref_coarse" not in cache:
            cache["ref_coarse"] = make_reference(
                system, data.test_ic, scfg["T_roll"], scfg["dt_coarse"]
            )
        for integ in ("rk4", "midpoint"):
            ev.update(
                {
                    f"{k}_coarse": v
                    for k, v in rollout_scores(model, system, cache["ref_coarse"], integ).items()
                }
            )
        ev["dt_coarse"] = scfg["dt_coarse"]

    return {
        **job,
        **ev,
        "val_loss": res.val_loss,
        "train_seconds": res.seconds,
        "target_snr": data.train.target_snr,
        "n_train": len(data.train),
    }


def main(shard: int = 0, n_shards: int = 1, out_dir: str = "results", quick: bool = False):
    torch.set_num_threads(int(os.environ.get("PCNN_THREADS", "2")))
    all_jobs = jobs(quick)
    # Shard by *data key*, not by index: every job sharing a dataset lands on
    # the same worker, which then builds that dataset and its reference
    # trajectories once. Round-robin over the sorted list would do the
    # opposite. Job definitions are seeded independently, so the assignment
    # changes nothing about the results -- only how often work is repeated.
    all_jobs.sort(key=lambda j: (_data_key(j), j["exp"], j["model"], j["lam"]))
    keys = sorted({_data_key(j) for j in all_jobs})
    owner = {k: i % n_shards for i, k in enumerate(keys)}
    mine = [j for j in all_jobs if owner[_data_key(j)] == shard]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"raw_shard{shard:02d}.jsonl"
    done = set()
    if path.exists():  # resume
        with open(path) as fh:
            for line in fh:
                try:
                    done.add(json.dumps(_sig(json.loads(line)), sort_keys=True))
                except Exception:
                    pass

    cache: dict = {}
    with open(path, "a") as fh:
        for i, job in enumerate(mine):
            if json.dumps(_sig(job), sort_keys=True) in done:
                continue
            rec = run_job(job, cache)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(
                f"[shard {shard}] {i + 1}/{len(mine)} {job['exp']:8s} {job['system']:15s} "
                f"{job['model']} lam={job['lam']:<7g} seed={job['seed']} "
                f"-> vf_in={rec['vf_mse_in']:.4g}",
                flush=True,
            )


def _sig(j):
    return {k: j[k] for k in ("exp", "system", "model", "seed", "lam", "noise", "n_traj", "steps")}

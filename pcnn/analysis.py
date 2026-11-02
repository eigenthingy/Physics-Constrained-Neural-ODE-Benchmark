"""Aggregation: raw JSONL -> tidy tables with uncertainty.

Two things here are load-bearing for the credibility of the whole repo.

1. **Lambda selection.** Model B exists at three lambdas everywhere. We keep the
   one with the lowest *validation* loss, per (study, system, seed, setting).
   Picking the lambda that happens to score best on the test metric would turn
   B into an oracle; picking one fixed lambda would turn it into a straw man.
   `select_lambda` does the honest thing and records which lambda won.

2. **Uncertainty.** Every number reported is a median over seeds with a
   percentile-bootstrap 95% CI. Single-seed deep-learning comparisons are
   noise; if two models' intervals overlap we say so rather than ranking them.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci

#: settings that identify one experimental cell (everything except the model)
CELL = ["exp", "system", "noise", "n_traj", "seed"]

HEADLINE_METRICS = [
    "vf_mse_in",
    "vf_mse_out",
    "valid_time_rk4",
    "valid_time_rk4_extrap",
    "energy_drift_rk4",
    "residual_cos_in",
    "blowup_rk4",
    "invariant_r2",
]


def load_raw(results_dir="results") -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(Path(results_dir) / "raw_shard*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"no results in {results_dir}; run scripts/run_experiments.py")
    df = pd.DataFrame(rows)
    return df.drop_duplicates(
        subset=["exp", "system", "model", "seed", "lam", "noise", "n_traj", "steps"], keep="last"
    )


def select_lambda(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse model B's lambda candidates to one row, by validation loss."""
    b = df[df.model == "B"].copy()
    rest = df[df.model != "B"].copy()

    sweep = b[b.exp == "lambda"].copy()  # the Pareto study keeps every lambda
    tuned = b[b.exp != "lambda"].copy()
    idx = tuned.groupby(CELL)["val_loss"].idxmin()
    tuned = tuned.loc[idx].copy()
    tuned["lam_selected"] = tuned["lam"]

    out = pd.concat([rest, tuned, sweep], ignore_index=True)
    return out.sort_values(["exp", "system", "model", "noise", "n_traj", "seed"])


def summarize(df: pd.DataFrame, by: list[str], metrics=None) -> pd.DataFrame:
    """Median + bootstrap 95% CI across seeds, long form."""
    metrics = metrics or [m for m in HEADLINE_METRICS if m in df.columns]
    recs = []
    for keys, g in df.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(by, keys))
        for m in metrics:
            if m not in g:
                continue
            med, lo, hi = bootstrap_ci(g[m].to_numpy())
            recs.append({**base, "metric": m, "median": med, "lo": lo, "hi": hi, "n_seeds": len(g)})
    return pd.DataFrame(recs)


def reference_slice(df: pd.DataFrame, noise=0.01, n_traj=40) -> pd.DataFrame:
    """The headline comparison: the 'noise' study at the reference setting."""
    return df[(df.exp == "noise") & (df.noise == noise) & (df.n_traj == n_traj)]


def significance(df: pd.DataFrame, metric: str, by=("system",), lower_is_better=True):
    """Paired, per-seed comparison of every model against baseline A.

    Paired because all four models see identical data and initial-condition
    draws for a given seed -- so the seed-to-seed variance cancels, and a
    small consistent win is detectable with five seeds.
    """
    recs = []
    for keys, g in df.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        piv = g.pivot_table(index="seed", columns="model", values=metric)
        if "A" not in piv:
            continue
        for m in [c for c in piv.columns if c != "A"]:
            d = (piv[m] - piv["A"]).dropna()
            if d.empty:
                continue
            ratio = (piv[m] / piv["A"]).dropna()
            wins = int((d < 0).sum()) if lower_is_better else int((d > 0).sum())
            recs.append(
                {
                    **dict(zip(by, keys)),
                    "model": m,
                    "metric": metric,
                    "median_ratio_vs_A": float(np.median(ratio)),
                    "wins_over_A": wins,
                    "n_seeds": int(len(d)),
                }
            )
    return pd.DataFrame(recs)


def build_tables(results_dir="results", out_dir="results") -> dict[str, pd.DataFrame]:
    raw = load_raw(results_dir)
    df = select_lambda(raw)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = {}
    tables["all_runs"] = df
    tables["headline"] = summarize(reference_slice(df), ["system", "model"])
    tables["noise"] = summarize(df[df.exp == "noise"], ["system", "model", "noise"])
    tables["data"] = summarize(
        df[(df.exp == "data") | ((df.exp == "noise") & (df.noise == 0.01))],
        ["system", "model", "n_traj"],
    )
    tables["lambda"] = summarize(
        df[df.exp == "lambda"], ["system", "lam"], metrics=["vf_mse_in", "vf_mse_out",
                                                            "residual_cos_in", "valid_time_rk4",
                                                            "energy_drift_rk4", "val_loss"]
    )
    tables["misspec"] = summarize(df[df.exp == "misspec"], ["system", "model"])
    integ_metrics = [
        c
        for c in df.columns
        if c.startswith(("energy_drift_", "valid_time_", "rollout_rmse_"))
        and not c.endswith("_extrap")
    ]
    tables["integ"] = summarize(df[df.exp == "integ"], ["system", "model"], metrics=integ_metrics)
    tables["significance"] = pd.concat(
        [
            significance(reference_slice(df), "vf_mse_in"),
            significance(reference_slice(df), "vf_mse_out"),
            significance(reference_slice(df), "energy_drift_rk4"),
            significance(reference_slice(df), "valid_time_rk4", lower_is_better=False),
        ],
        ignore_index=True,
    )

    for k, v in tables.items():
        v.to_csv(out / f"table_{k}.csv", index=False)
    return tables


if __name__ == "__main__":
    t = build_tables()
    pd.set_option("display.width", 160, "display.max_rows", 200)
    print(t["headline"].pivot_table(index=["system", "metric"], columns="model", values="median"))

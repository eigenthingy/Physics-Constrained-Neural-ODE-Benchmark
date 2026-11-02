"""Figures.

Palette is checked, not chosen by eye: the four series colours clear the
all-pairs OKLab separation gates for normal vision and for simulated
protan/deutan/tritan vision (`python scripts/validate_palette.py`). Aqua sits
below 3:1 contrast on the light surface, so every chart that uses it also
carries a legend *and* a distinct marker shape -- identity is never colour
alone.

Ground truth is always the same black dashed line, never a series colour.

Layout note: every figure gets its title block from `_header`, which reserves
space in **inches** and then places the text into it. Hard-coded figure-fraction
y positions do not survive a change in row count or figure height, which is how
titles end up sitting on top of axes.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

# --------------------------------------------------------------------------- #
# design tokens
# --------------------------------------------------------------------------- #

PALETTE = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#4a3aa7"}
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D"}
LABELS = {
    "A": "A · unconstrained",
    "B": "B · soft penalty",
    "C": "C · Hamiltonian net",
    "D": "D · projected",
}
ORDER = ["A", "B", "C", "D"]

TRUTH = "#0b0b0b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

SYSTEM_TITLE = {
    "pendulum": "Nonlinear pendulum  (1 DOF)",
    "henon_heiles": "Hénon–Heiles  (2 DOF, chaotic)",
    "damped_pendulum": "Damped pendulum  (energy NOT conserved)",
}

#: short, two-line forms for rotated row labels -- a rotated label is bounded by
#: the axes *height*, so the long titles above collide when used there.
SYSTEM_ROW = {
    "pendulum": "Pendulum\nd = 2",
    "henon_heiles": "Hénon–Heiles\nd = 4, chaotic",
    "damped_pendulum": "Damped pend.\nd = 2, dissipative",
}

#: a hard constraint can drive a metric to exactly 0, which a log axis cannot
#: show. Values at or below this floor are drawn at the floor and labelled with
#: their true value, so "exactly zero" stays visible instead of disappearing.
LOG_FLOOR = 1e-12


def use_style():
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


# --------------------------------------------------------------------------- #
# layout helpers
# --------------------------------------------------------------------------- #

_TITLE_IN = 0.30  #: baseline of the title, inches below the figure top
_SUBLINE_IN = 0.19  #: height of one subtitle line, inches
_GAP_IN = 0.20  #: gap between the header block and the axes


def _header(fig, title, sub="", bottom=0.0, title_size=12.5):
    """Reserve an inch-measured header, lay the axes out below it, then fill it.

    Returns the top of the axes region, in figure fractions.
    """
    fw, fh = fig.get_size_inches()
    lines = textwrap.wrap(sub, width=max(40, int(fw * 17))) if sub else []
    head_in = _TITLE_IN + 0.16 + _SUBLINE_IN * len(lines) + _GAP_IN
    top = 1.0 - head_in / fh

    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.text(0.008, 1.0 - _TITLE_IN / fh, title, ha="left", va="center",
             fontsize=title_size, fontweight="semibold", color=INK)
    for i, line in enumerate(lines):
        y = 1.0 - (_TITLE_IN + 0.18 + _SUBLINE_IN * i) / fh
        fig.text(0.008, y, line, ha="left", va="center", fontsize=8.5, color=MUTED)
    return top


def _decade_ticks(ax, axis="x"):
    """Decade-only labels on a log axis.

    Matplotlib labels minor log ticks when a range spans less than a decade,
    which is how '3x10^-3' ends up printed on top of '4x10^-3'. Every point in
    these charts carries its own value label, so the axis only has to convey
    scale.
    """
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10.0))
    a.set_minor_formatter(NullFormatter())


def _despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def _title(ax, text, sub=None):
    ax.set_title(text, color=INK, pad=16 if sub else 6)
    if sub:
        ax.text(0.0, 1.008, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, color=MUTED)


def _handles(keys=ORDER):
    return [
        plt.Line2D([], [], color=PALETTE[k], marker=MARKERS[k], label=LABELS[k], lw=2, ms=5.5)
        for k in keys
    ]


def _legend(ax, keys=ORDER, **kw):
    ax.legend(handles=_handles(keys), **kw)


def _logsafe(v):
    return LOG_FLOOR if (not np.isfinite(v) or v <= 0) else max(v, LOG_FLOOR)


def _save(fig, name, outdir="figures"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    p = Path(outdir) / name
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)
    return p


def _pick(summary, system, model, metric):
    r = summary[
        (summary.system == system) & (summary.model == model) & (summary.metric == metric)
    ]
    if r.empty:
        return np.nan, np.nan, np.nan
    r = r.iloc[0]
    return r["median"], r["lo"], r["hi"]


# --------------------------------------------------------------------------- #
# 1 — headline
# --------------------------------------------------------------------------- #

HEADLINE_PANELS = [
    ("vf_mse_in", "One-step error\n(in-distribution)", True, "lower is better"),
    ("vf_mse_out", "One-step error\n(unseen energy shell)", True, "lower is better"),
    ("valid_time_rk4", "Valid rollout time\n(time units)", False, "higher is better"),
    ("energy_drift_rk4", "Energy drift\n(relative, end of rollout)", True, "lower is better"),
]


def fig_headline(
    summary,
    systems,
    outdir="figures",
    name="fig1_headline.png",
    title="Four ways to impose a conservation law, scored on the same data",
    sub=("median over 5 seeds · bars are bootstrap 95% CI · identical data, budget and "
         "parameter count for every model · columns are labelled once, at the top"),
):
    nrow = len(systems)
    fig, axes = plt.subplots(nrow, 4, figsize=(11.6, 2.45 * nrow + 0.5), squeeze=False)

    for i, sysname in enumerate(systems):
        for j, (metric, coltitle, logx, hint) in enumerate(HEADLINE_PANELS):
            ax = axes[i][j]
            ys = np.arange(len(ORDER))[::-1]
            any_finite = False
            for y, k in zip(ys, ORDER):
                med, lo, hi = _pick(summary, sysname, k, metric)
                if not np.isfinite(med):
                    continue
                any_finite = True
                a, b, c = (_logsafe(med), _logsafe(lo), _logsafe(hi)) if logx else (med, lo, hi)
                ax.plot([b, c], [y, y], color=PALETTE[k], lw=2.4, solid_capstyle="round",
                        alpha=0.45)
                ax.plot([a], [y], marker=MARKERS[k], color=PALETTE[k], ms=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
                ax.annotate(f"{med:.3g}" if med > 0 else "0", (a, y),
                            textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=7.5, color=INK2, clip_on=False)
            ax.set_yticks(ys, ORDER)
            ax.set_ylim(-0.7, len(ORDER) - 0.25)
            if logx and any_finite:
                ax.set_xscale("log")
                _decade_ticks(ax)
            ax.margins(x=0.16)
            ax.grid(axis="y", visible=False)
            _despine(ax)
            # columns are named once, at the top: the grid alignment carries the
            # label down, and repeating it collides with the legend strip.
            if i == 0:
                _title(ax, coltitle, hint)
        axes[i][0].set_ylabel(SYSTEM_ROW[sysname], color=INK, fontsize=9.5,
                              labelpad=8, fontweight="semibold", linespacing=1.5)

    legend_in = 0.34
    bottom = legend_in / fig.get_size_inches()[1]
    _header(fig, title, sub, bottom=bottom, title_size=13)
    fig.legend(handles=_handles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.004), columnspacing=2.4)
    return _save(fig, name, outdir)


# --------------------------------------------------------------------------- #
# 2 — phase portraits
# --------------------------------------------------------------------------- #


def fig_phase_portraits(truth, preds, outdir="figures", name="fig2_phase_portraits.png",
                        system="pendulum", extra_note=""):
    """truth: (n_ic, n_t, 2); preds: {key: (n_ic, n_t, 2)}"""
    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.0), sharex=True, sharey=True)
    axes[0].set_title("ground truth", color=INK)
    for tr in truth:
        axes[0].plot(tr[:, 0], tr[:, 1], color=TRUTH, lw=1.1, alpha=0.85)
    for ax, k in zip(axes[1:], ORDER):
        for tr in truth:
            ax.plot(tr[:, 0], tr[:, 1], color=TRUTH, lw=0.9, alpha=0.28, ls=(0, (3, 2)))
        for tr in preds[k]:
            ax.plot(tr[:, 0], tr[:, 1], color=PALETTE[k], lw=1.2, alpha=0.9)
        ax.set_title(LABELS[k], color=INK)
    for ax in axes:
        ax.set_xlabel("q")
        _despine(ax)
    axes[0].set_ylabel("p")
    _header(
        fig,
        f"Long rollouts in phase space — {SYSTEM_TITLE[system]}",
        "dashed black = truth. " + (extra_note or
        "Closed orbits mean the flow is conserving something; matching the black curve means "
        "it is conserving the right thing."),
    )
    return _save(fig, name, outdir)


# --------------------------------------------------------------------------- #
# 3 — energy drift over time
# --------------------------------------------------------------------------- #


def fig_energy_curves(curves, outdir="figures"):
    """curves: {system: (t, {key: median |dH|/scale over ICs})}"""
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 3.6), squeeze=False)
    for ax, (sysname, (t, series)) in zip(axes[0], curves.items()):
        for k in ORDER:
            y = np.maximum(series[k], LOG_FLOOR)
            step = max(1, len(t) // 8)
            ax.plot(t, y, color=PALETTE[k], lw=1.9)
            ax.plot(t[::step], y[::step], ls="none", marker=MARKERS[k], color=PALETTE[k],
                    ms=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_yscale("log")
        _decade_ticks(ax, "y")
        ax.set_xlabel("time")
        ax.set_ylabel("|ΔE| / E scale")
        _title(ax, SYSTEM_TITLE[sysname])
        _despine(ax)
    _legend(axes[0][0], loc="lower right")
    _header(fig, "Does the rollout stay on its energy shell?",
            "median over 24 initial conditions, RK4 at dt = 0.05 · D is flat at machine "
            "precision because the constraint is structural")
    return _save(fig, "fig3_energy_drift.png", outdir)


# --------------------------------------------------------------------------- #
# 4/5 — sweeps
# --------------------------------------------------------------------------- #


def _sweep_panel(ax, summary, sysname, xcol, metric, logx, logy, xlabel, ylabel):
    for k in ORDER:
        sub = summary[
            (summary.system == sysname) & (summary.model == k) & (summary.metric == metric)
        ].sort_values(xcol)
        if sub.empty:
            continue
        x = sub[xcol].to_numpy(dtype=float)
        pos = x if logx else np.arange(len(x))
        clip = (lambda a: np.clip(a.to_numpy(dtype=float), LOG_FLOOR, None)) if logy else (
            lambda a: a.to_numpy(dtype=float))
        ax.fill_between(pos, clip(sub["lo"]), clip(sub["hi"]), color=PALETTE[k], alpha=0.14, lw=0)
        ax.plot(pos, clip(sub["median"]), color=PALETTE[k], lw=1.9, marker=MARKERS[k],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        if not logx:
            ax.set_xticks(pos, [f"{v:g}" for v in x])
    if logx:
        ax.set_xscale("log")
        _decade_ticks(ax)
    if logy:
        ax.set_yscale("log")
        _decade_ticks(ax, "y")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _despine(ax)


def fig_sweep(summary, systems, xcol, panels, title, sub, name, logx=False,
              xlabel=None, outdir="figures"):
    fig, axes = plt.subplots(len(panels), len(systems),
                             figsize=(5.5 * len(systems), 3.25 * len(panels)), squeeze=False)
    for j, sysname in enumerate(systems):
        for i, (metric, ylabel, logy) in enumerate(panels):
            ax = axes[i][j]
            _sweep_panel(ax, summary, sysname, xcol, metric, logx, logy,
                         xlabel or xcol.replace("_", " "), ylabel)
            if i == 0:
                _title(ax, SYSTEM_TITLE[sysname])
    _legend(axes[0][0], loc="best", ncol=2)
    _header(fig, title, sub)
    return _save(fig, name, outdir)


# --------------------------------------------------------------------------- #
# 6 — the lambda Pareto front
# --------------------------------------------------------------------------- #


def fig_lambda(lam_table, systems, ref_points, selected, outdir="figures"):
    fig, axes = plt.subplots(1, len(systems), figsize=(5.9 * len(systems), 4.2), squeeze=False)
    for ax, sysname in zip(axes[0], systems):
        sub = lam_table[lam_table.system == sysname]
        piv = sub.pivot_table(index="lam", columns="metric", values="median").sort_index()
        x, y = piv["residual_cos_in"].to_numpy(), piv["vf_mse_in"].to_numpy()
        ax.plot(x, y, color=PALETTE["B"], lw=1.6, marker="s", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        for lam, xx, yy in zip(piv.index, x, y):
            ax.annotate(f"λ={lam:g}", (xx, yy), textcoords="offset points", xytext=(7, 5),
                        fontsize=7.5, color=INK2)
        for k, marker_lbl in [("A", "no physics"), ("D", "hard constraint")]:
            px, py = ref_points[sysname][k]
            ax.plot([_logsafe(px)], [_logsafe(py)], marker=MARKERS[k], color=PALETTE[k], ms=9,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
            ax.annotate(f"{k} · {marker_lbl}", (_logsafe(px), _logsafe(py)),
                        textcoords="offset points", xytext=(8, -13), fontsize=8,
                        color=PALETTE[k], fontweight="semibold")
        sel = selected.get(sysname)
        if sel is not None and sel in piv.index:
            ax.plot([piv.loc[sel, "residual_cos_in"]], [piv.loc[sel, "vf_mse_in"]],
                    marker="o", ms=15, mfc="none", mec=INK, mew=1.4, zorder=5)
            ax.annotate("λ picked by\nvalidation loss",
                        (piv.loc[sel, "residual_cos_in"], piv.loc[sel, "vf_mse_in"]),
                        textcoords="offset points", xytext=(-22, -44), fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _decade_ticks(ax)
        _decade_ticks(ax, "y")
        # mathtext, not a literal glyph: Segoe UI has no NABLA and would render
        # a tofu box. Mathtext always falls back to DejaVu, which does.
        ax.set_xlabel(r"physics violation  $|\cos(\nabla H,\, f)|$   →  worse")
        ax.set_ylabel("one-step error (in-distribution)  →  worse")
        _title(ax, SYSTEM_TITLE[sysname])
        _despine(ax)
    _header(fig, "The soft penalty is a dial, and you have to pay to turn it",
            "Each point is one λ, median over 5 seeds. Down-left is better on both axes. "
            "The hard-constrained model D sits off the front entirely: zero violation at no "
            "accuracy cost.")
    return _save(fig, "fig6_lambda_pareto.png", outdir)


# --------------------------------------------------------------------------- #
# 7 — what invariant did the Hamiltonian model learn?
# --------------------------------------------------------------------------- #


def fig_invariant(grids, residuals, outdir="figures"):
    """Level sets, then the *residual* of the affine fit.

    Plotting H_learned against H_true produces a near-perfect straight line and
    hides the finding. The information is in what the straight line misses, so
    that is what gets plotted: everything shared between the two functions is
    subtracted away and only the disagreement is left.
    """
    n = len(residuals)
    fig, panels = plt.subplots(1, n + 1, figsize=(4.7 * (n + 1), 4.1))

    ax = panels[0]
    Q, P, Ht, Hl = grids
    ax.contour(Q, P, Ht, levels=10, colors=[TRUTH], linewidths=1.0, linestyles="dashed")
    ax.contour(Q, P, Hl, levels=10, colors=[PALETTE["C"]], linewidths=1.6)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    _title(ax, "Pendulum: level sets", "dashed = true energy · solid = learned invariant")
    _despine(ax)

    for i, (sysname, d) in enumerate(residuals.items()):
        ax = panels[i + 1]
        ax.axhline(0.0, color=TRUTH, lw=1.0, ls=(0, (3, 2)), zorder=1)
        for ht, res, lbl, col in [
            (d["h_in"], d["r_in"], "training shell", PALETTE["C"]),
            (d["h_out"], d["r_out"], "unseen shell", PALETTE["D"]),
        ]:
            ax.scatter(ht, res, s=5, color=col, alpha=0.3, lw=0, label=lbl)
        ax.set_xlabel("true energy  H(z)")
        ax.set_ylabel("$H_\\theta$ − (best affine fit of H)")
        _title(ax, SYSTEM_TITLE[sysname],
               f"R² = {d['r2']:.4f} on the training shell, {d['r2_out']:.4f} on the unseen one")
        ax.legend(loc="upper left", markerscale=2.6)
        _despine(ax)

    _header(fig, "Model C conserves something exactly. Is it the energy?",
            "The architecture guarantees its own invariant is conserved to machine precision. "
            "Whether that invariant is the physical energy is an empirical question. Subtract "
            "the best affine fit and what remains is structured, not noise — the learned "
            "invariant is a different function of state, and it diverges off the training shell.")
    return _save(fig, "fig7_learned_invariant.png", outdir)


# --------------------------------------------------------------------------- #
# 9 — integrator
# --------------------------------------------------------------------------- #


def fig_integrator(integ, systems, dt_fine, dt_coarse, outdir="figures"):
    fig, axes = plt.subplots(1, len(systems), figsize=(5.9 * len(systems), 3.9), squeeze=False)
    width = 0.36
    for ax, sysname in zip(axes[0], systems):
        sub = integ[integ.system == sysname]
        piv = sub.pivot_table(index="model", columns="metric", values="median")
        xs = np.arange(len(ORDER))
        for off, col, hatch in [
            (-width / 2, "energy_drift_rk4_coarse", ""),
            (+width / 2, "energy_drift_midpoint_coarse", "///"),
        ]:
            vals = [piv.loc[k, col] if (k in piv.index and col in piv.columns) else np.nan
                    for k in ORDER]
            vals = np.nan_to_num(np.asarray(vals, dtype=float), nan=LOG_FLOOR)
            ax.bar(xs + off, np.maximum(vals, LOG_FLOOR), width * 0.92,
                   color=[PALETTE[k] for k in ORDER], hatch=hatch, edgecolor=SURFACE,
                   linewidth=1.6)
        for i, k in enumerate(ORDER):
            v = piv.loc[k, "energy_drift_rk4"] if k in piv.index else np.nan
            ax.plot([xs[i] - width * 0.8, xs[i] + width * 0.8], [v, v], color=TRUTH, lw=1.4,
                    ls=(0, (2, 2)), zorder=4)
        ax.set_yscale("log")
        _decade_ticks(ax, "y")
        ax.set_xticks(xs, ORDER)
        ax.set_ylabel("energy drift (relative)")
        _title(ax, SYSTEM_TITLE[sysname],
               f"bars: coarse step dt={dt_coarse[sysname]:g} · dashed: fine step dt={dt_fine:g}")
        ax.grid(axis="x", visible=False)
        _despine(ax)
        ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=SURFACE, label="RK4"),
            plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor=SURFACE, hatch="///",
                          label="implicit midpoint (symplectic)"),
            plt.Line2D([], [], color=TRUTH, ls=(0, (2, 2)), lw=1.4,
                       label="fine step, either solver"),
        ], loc="upper left", fontsize=8)
    _header(fig, "Does the solver spend the architecture's guarantee?",
            "Same trained weights, three rollouts. At a fine step the choice of solver is "
            "irrelevant — the drift you see is the model's. It only starts to matter once the "
            "step is coarse, by which point every model is worse anyway.")
    return _save(fig, "fig9_integrator.png", outdir)

"""Publication figures, mirroring the base paper's Figs. 2, 13, 14 and 22.

Design rules this module follows, deliberately:

* **Categorical colour is assigned by identity, in fixed slot order, never
  cycled.** Slot 1 = linear COAR, slot 2 = NACA, slot 3 = multi-alpha COAR. A
  method keeps its colour in every figure, whether or not the others are drawn.
* **No dual-axis plots, ever.** ``Var(f_M)`` shares an x-axis with rho/R^2/MSE
  but lives in its own panel of a small-multiples grid, because a second y-scale
  invites exactly the misreading the design spec 2.7 is trying to prevent.
* **Text wears ink colours, not series colours**; marks carry identity.
* Every series is both legend-labelled and (where <= 4 series) direct-labelled,
  so identity never rests on colour alone. The aqua slot sits below 3:1 contrast
  on a white surface, so direct labels are a requirement here, not a flourish.
* Grid and spines are recessive; marks are thin.

Palette validated with the dataviz validator (light surface, all-pairs):
worst CVD ΔE 9.2, worst normal-vision ΔE 24.0, all checks pass.

Figures are light-surface only: they are for print.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "SERIES",
    "LINESTYLES",
    "INK",
    "setup_style",
    "fig_accuracy_vs_alpha",
    "fig_naf_vs_alpha",
    "fig_depth_localization",
    "fig_method_comparison",
    "fig_interaction_spectrum",
    "fig_protocol_trial",
    "save",
]

# Categorical slots -- identity, fixed order (dataviz reference palette, light)
SERIES: dict[str, str] = {
    "linear": "#2a78d6",  # slot 1  blue
    "naca": "#eb6834",  # slot 2  orange
    "multi_alpha": "#1baf7a",  # slot 3  aqua
    "mlp": "#eda100",  # slot 4  yellow  (facet away from slot 2 when possible)
    "gbt": "#e87ba4",  # slot 5  magenta
    "psd": "#4a3aa7",  # slot 7  violet
}
# Distinct dash patterns so two series that land on the same values still read
# as two series. On Setup A this is not a nicety: NACA and linear COAR agree to
# four decimals, and with both drawn solid the lower one is simply invisible --
# the figure would look like a missing series rather than like the finding.
LINESTYLES: dict[str, object] = {
    "linear": "-",
    "naca": (0, (5, 2)),
    "multi_alpha": (0, (1, 1.6)),
    "mlp": (0, (3, 1, 1, 1)),
    "gbt": (0, (4, 1, 1, 1, 1, 1)),
    "psd": (0, (2, 2)),
}
MARKERS: dict[str, str] = {
    "linear": "o",
    "naca": "s",
    "multi_alpha": "^",
    "mlp": "D",
    "gbt": "v",
    "psd": "P",
}
LABELS: dict[str, str] = {
    "linear": "Linear COAR",
    "naca": "NACA",
    "multi_alpha": "Multi-$\\alpha$ COAR",
    "mlp": "MLP",
    "gbt": "GBT",
    "psd": "PSD-FM",
}

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#8a8985",
    "surface": "#ffffff",
    "grid": "#e5e4e0",
}


def setup_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK["primary"],
            "axes.edgecolor": INK["muted"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.7,
            "text.color": INK["primary"],
            "xtick.color": INK["secondary"],
            "ytick.color": INK["secondary"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
        }
    )


def _mark_alpha_train(ax, alpha_train: float) -> None:
    ax.axvline(
        alpha_train,
        color=INK["muted"],
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=0,
    )
    ax.annotate(
        r"$\alpha_{train}$",
        xy=(alpha_train, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -10),
        textcoords="offset points",
        color=INK["secondary"],
        fontsize=7.5,
    )


_LABEL_SLOTS: dict = {}


def _direct_label(ax, x, y, text: str, color: str) -> None:
    """Direct label at the last point -- identity without relying on colour.

    Series that end at nearly the same value would print on top of each other.
    That is not hypothetical here: the whole Setup A result is that NACA and
    linear COAR are almost indistinguishable, so their curves *do* land on the
    same point. Overlapping labels would turn the finding into an illegible
    smudge, so co-located labels are stacked instead.
    """
    used = _LABEL_SLOTS.setdefault(id(ax), [])
    lo, hi = ax.get_ylim()
    span = max(abs(hi - lo), 1e-12)
    dy = 0
    for prev in used:
        if abs(prev - y) / span < 0.06:
            dy -= 9
    used.append(y)
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(5, dy),
        textcoords="offset points",
        color=INK["primary"],
        fontsize=7.5,
        va="center",
    )


def fig_accuracy_vs_alpha(
    df,
    alpha_train: float,
    *,
    series: tuple[str, ...] = ("linear", "naca"),
    columns: dict[str, dict[str, str]] | None = None,
    title: str = "Attribution accuracy across ablation rates",
):
    """Base-paper Fig. 13 analogue: rho, R^2, MSE and Var(f_M) as small multiples.

    Four panels rather than one plot with two y-axes. ``Var(f_M)`` is what
    separates a signal-to-noise collapse at small alpha from genuine
    misspecification, and it only reads correctly on its own scale.
    """
    import matplotlib.pyplot as plt

    setup_style()
    _LABEL_SLOTS.clear()
    columns = columns or {
        "linear": {"rho": "rho_mean", "r2": "r2_oc_mean", "mse": "mse_mean"},
        "naca": {"rho": "rho_naca", "r2": "r2_oc_naca", "mse": "mse_naca"},
    }
    panels = [
        ("rho", r"Pearson $\rho$", False),
        ("r2", r"$R^2$ (offset-corrected)", False),
        ("mse", "MSE", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9))
    x = df["alpha_test"].to_numpy()

    for ax, (key, ylabel, logy) in zip(axes[:3], panels):
        for name in series:
            col = columns.get(name, {}).get(key)
            if col is None or col not in df:
                continue
            y = df[col].to_numpy()
            ax.plot(
                x,
                y,
                color=SERIES[name],
                marker=MARKERS[name],
                linestyle=LINESTYLES.get(name, "-"),
                label=LABELS[name],
                markeredgecolor=INK["surface"],
                markeredgewidth=0.8,
                markerfacecolor="none" if name != "linear" else SERIES[name],
            )
            _direct_label(ax, x[-1], y[-1], LABELS[name], SERIES[name])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha_{test}$")
        ax.set_ylabel(ylabel)
        # headroom on the right so direct labels sit inside the axes
        ax.set_xmargin(0.30)
        _mark_alpha_train(ax, alpha_train)

    # Panel 4: the SNR diagnostic, on its own scale -- never a second y-axis.
    ax = axes[3]
    if "target_var_mean" in df:
        ax.plot(
            x,
            df["target_var_mean"].to_numpy(),
            color=INK["secondary"],
            marker="o",
            markeredgecolor=INK["surface"],
            markeredgewidth=0.8,
        )
    ax.set_xlabel(r"$\alpha_{test}$")
    ax.set_ylabel(r"$\mathrm{Var}_t\,(f_M(z,C_t))$")
    ax.set_title("signal available", fontsize=8, color=INK["secondary"], loc="left")
    _mark_alpha_train(ax, alpha_train)

    # One legend for the whole figure: every panel is already direct-labelled, so
    # a per-panel legend would state identity twice in the same frame.
    handles, labels = [], []
    for ax_ in axes:
        h_, l_ = ax_.get_legend_handles_labels()
        for h, lab in zip(h_, l_):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.005, 1.02),
            ncol=len(handles),
        )
    fig.suptitle(title, y=1.10, fontsize=11, color=INK["primary"], fontweight="bold")
    fig.tight_layout()
    return fig


def fig_naf_vs_alpha(df, alpha_train: float, *, title: str = "Non-additive fraction"):
    """Base-paper Fig. 14 analogue: NAF vs alpha, raw and offset-corrected.

    Both gauges are drawn because raw ``R^2`` is not gauge-invariant out of
    distribution (see the gauge-degeneracy analysis) and reporting only the
    raw curve could manufacture an H2-shaped result.
    """
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = df["alpha_test"].to_numpy()

    if "naf_mean" in df:
        ax.plot(
            x,
            df["naf_mean"].to_numpy(),
            color=SERIES["naca"],
            marker=MARKERS["naca"],
            linestyle=(0, (5, 2)),
            label="NAF (raw gauge)",
            markeredgecolor=INK["surface"],
            markeredgewidth=0.8,
        )
    y = df["naf_oc_mean"].to_numpy()
    ax.plot(
        x,
        y,
        color=SERIES["naca"],
        marker=MARKERS["naca"],
        label="NAF (offset-corrected)",
        markeredgecolor=INK["surface"],
        markeredgewidth=0.8,
    )
    ax.axhline(0.0, color=INK["muted"], linewidth=0.9, zorder=0)
    _mark_alpha_train(ax, alpha_train)

    # A near-zero NAF plotted on an auto-scaled axis looks dramatic. State the
    # magnitude on the figure so the axis scale cannot be misread as an effect:
    # the reader should see "0.05% of residual variance", not a big curve.
    at = df.loc[(df["alpha_test"] - alpha_train).abs().idxmin()]
    span = float(np.nanmax(np.abs(y)))
    _naf_note = (
        rf"at $\alpha_{{train}}$: NAF = {at['naf_oc_mean']:+.4f} "
        rf"({at['naf_oc_mean'] * 100:+.2f}% of the residual variance). "
        rf"This axis spans only $\pm${span * 100:.2f}% — the curve is structure, not size."
    )
    ax.set_xlabel(r"$\alpha_{test}$")
    ax.set_ylabel(r"NAF $=(R^2_{NACA}-R^2_{lin})/(1-R^2_{lin})$")
    ax.set_title(title, loc="left")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.text(0.0, -0.02, _naf_note, fontsize=7.5, color=INK["secondary"], va="top")
    return fig


def fig_depth_localization(dd, *, title: str = "Where non-additivity lives"):
    """Base-paper Fig. 22 analogue: NAF and pool enrichment by depth group."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    groups = dd["group"].tolist()
    ypos = np.arange(len(groups))

    axes[0].barh(
        ypos,
        dd["naf_oc_weighted"].to_numpy(),
        color=SERIES["naca"],
        height=0.62,
        edgecolor=INK["surface"],
        linewidth=1.2,
    )
    axes[0].set_yticks(ypos, groups)
    axes[0].set_xmargin(0.18)
    axes[0].set_xlabel("NAF from a pool confined to this group")
    axes[0].set_title("non-additivity reachable within each group", loc="left")
    axes[0].invert_yaxis()
    for i, v in enumerate(dd["naf_oc_weighted"].to_numpy()):
        axes[0].annotate(
            f"{v:+.3f}",
            xy=(v, i),
            xytext=(4 if v >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=7.5,
            color=INK["primary"],
        )

    enr = dd["pool_enrichment"].to_numpy()
    axes[1].barh(
        ypos,
        enr,
        color=SERIES["linear"],
        height=0.62,
        edgecolor=INK["surface"],
        linewidth=1.2,
    )
    # parity reference drawn above the bars -- it is the thing being compared to
    axes[1].axvline(1.0, color=INK["primary"], linewidth=1.0, zorder=3)
    axes[1].set_yticks(ypos, groups)
    axes[1].set_xmargin(0.18)
    axes[1].set_xlabel("pool enrichment  (share in pool / share of network)")
    axes[1].set_title("what screening selected", loc="left")
    axes[1].invert_yaxis()
    for i, v in enumerate(enr):
        axes[1].annotate(
            f"{v:.2f}×",
            xy=(v, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK["primary"],
        )

    fig.suptitle(title, y=1.03, fontsize=11, fontweight="bold", color=INK["primary"])
    fig.tight_layout()
    return fig


def fig_method_comparison(df, alpha_train: float, *, metric: str = "r2_oc_mean"):
    """Base-paper Fig. 2 analogue: every estimator on one axis, across alpha."""
    import matplotlib.pyplot as plt

    setup_style()
    _LABEL_SLOTS.clear()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = df["alpha_test"].to_numpy()
    mapping = [
        ("linear", f"single_alpha__{metric}"),
        ("multi_alpha", f"multi_alpha_interp__{metric}"),
        ("naca", "r2_oc_naca"),
    ]
    for name, col in mapping:
        if col not in df:
            continue
        y = df[col].to_numpy()
        ax.plot(
            x,
            y,
            color=SERIES[name],
            marker=MARKERS[name],
            linestyle=LINESTYLES.get(name, "-"),
            label=LABELS[name],
            markeredgecolor=INK["surface"],
            markeredgewidth=0.8,
        )
        _direct_label(ax, x[-1], y[-1], LABELS[name], SERIES[name])
    ax.set_xmargin(0.24)
    _mark_alpha_train(ax, alpha_train)
    ax.set_xlabel(r"$\alpha_{test}$")
    ax.set_ylabel(r"$R^2$ (offset-corrected)")
    ax.set_title("Estimator comparison", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def save(fig, path: str | Path, *, also_pdf: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def fig_interaction_spectrum(rows, share_by_order, *, title="Exact interaction spectrum"):
    """The exhaustive-surface result: variance explained by ANOVA order.

    Two panels because they answer different questions. Left: cumulative R^2 as
    orders are added -- how much of the surface a truncated model can reach.
    Right: the incremental gain per order -- where the structure actually is.
    A cumulative curve alone hides the second, because everything looks flat once
    the first order dominates.

    ``rows`` is the per-order table from wp2b; ``share_by_order`` maps order ->
    share of total |beta| mass.
    """
    import matplotlib.pyplot as plt

    setup_style()
    _LABEL_SLOTS.clear()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    k = np.array([r["max_order"] for r in rows])
    r2 = np.array([r["r2"] for r in rows])
    inc = np.array([r["increment"] for r in rows])

    ax = axes[0]
    ax.plot(k, r2, color=SERIES["naca"], marker=MARKERS["naca"],
            markeredgecolor=INK["surface"], markeredgewidth=0.8)
    ax.axhline(1.0, color=INK["muted"], linewidth=0.9, zorder=0)
    ax.set_xlabel("model truncated at interaction order")
    ax.set_ylabel(r"$R^2$ on the exhaustive surface")
    ax.set_title("what a truncated model reaches", loc="left", fontsize=9)
    for x, y in ((k[1], r2[1]), (k[2], r2[2])):
        ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(4, -10), textcoords="offset points",
                    fontsize=7.5, color=INK["primary"])

    ax = axes[1]
    ax.bar(k[1:], inc[1:], color=SERIES["linear"], edgecolor=INK["surface"], linewidth=1.2)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("interaction order")
    ax.set_ylabel(r"incremental $R^2$ from this order")
    ax.set_title("where the structure is", loc="left", fontsize=9)
    ax.set_xticks(k[1:])

    fig.suptitle(title, y=1.04, fontsize=11, fontweight="bold", color=INK["primary"])
    fig.tight_layout()
    _ = share_by_order
    return fig


def fig_protocol_trial(df, *, title="What fixed-cardinality sampling costs"):
    """Models fitted at one cardinality, scored against the exhaustive truth.

    The gap between the two lines is the price of the sampling protocol, measured
    against a known answer rather than argued about.
    """
    import matplotlib.pyplot as plt

    setup_style()
    _LABEL_SLOTS.clear()
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    k = df["k"].to_numpy()
    for col, name, lbl in (
        ("additive_full", "linear", "additive, scored on full surface"),
        ("pairwise_full", "naca", "pairwise, scored on full surface"),
    ):
        if col in df:
            ax.plot(k, df[col].to_numpy(), color=SERIES[name], marker=MARKERS[name],
                    linestyle=LINESTYLES[name], label=lbl,
                    markeredgecolor=INK["surface"], markeredgewidth=0.8)
    for col, name, lbl in (
        ("additive_at_k", "linear", "additive, scored on its own stratum"),
        ("pairwise_at_k", "naca", "pairwise, scored on its own stratum"),
    ):
        if col in df:
            ax.plot(k, df[col].to_numpy(), color=SERIES[name], marker=MARKERS[name],
                    linestyle=(0, (1, 1.6)), alpha=0.55, label=lbl,
                    markeredgecolor=INK["surface"], markeredgewidth=0.8)
    ax.axhline(0.0, color=INK["muted"], linewidth=0.9, zorder=0)
    ax.set_xlabel("cardinality the model was fitted on (units ablated)")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower center", fontsize=7.5)
    fig.tight_layout()
    return fig

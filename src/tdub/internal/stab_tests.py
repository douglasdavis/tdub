"""Internal module for stability tests."""

import numpy as np
import matplotlib.pyplot as plt


def make_delta_mu_plot(ax: plt.Axes, nom_down, nom_up, xvals, xerlo, xerhi, ylabs) -> plt.Axes:
    """Skeleton for making a delta mu plot."""
    yvals = np.arange(1, len(xvals) + 1)
    ax.fill_betweenx(
        [-50, 500],
        nom_down,
        nom_up,
        color="gray",
        alpha=0.5,
        label="Nominal Fit Uncertainty",
    )
    ax.set_xlabel(r"$\Delta\mu=\mu_{tW}^{\mathrm{nominal}}-\mu_{tW}^{\mathrm{test}}$")
    for xv, yv in zip(xvals, yvals):
        t = f"{xv:1.3f}"
        ax.text(xv, yv + 0.075, t, ha="center", va="bottom", size=10)
    ax.set_yticks(yvals)
    ax.set_yticklabels(ylabs)
    ax.set_ylim([0.0, len(yvals) + 1])
    ax.errorbar(
        xvals,
        yvals,
        xerr=[abs(xerlo), xerhi],
        label="Individual tests",
        fmt="ko",
        lw=2,
        elinewidth=2.25,
        capsize=3.5,
    )
    ax.grid(color="black", alpha=0.15)
    ax.legend(bbox_to_anchor=(-1, 0.97, 0, 0), loc="lower left")
    return ax

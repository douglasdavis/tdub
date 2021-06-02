"""Cramped plot."""

import os
import pathlib
import sys

from typing import Tuple

import matplotlib.pyplot as plt

from tdub.art import canvas_from_counts, legend_last_to_first, draw_atlas_label
from tdub.rex import meta_text
from tdub.rex import region_plot_raw_material


def cramped(rex_dir, stage="pre"):
    """Generate a crampted plot."""
    fig: plt.Figure
    ax: Tuple[Tuple[plt.Axes, ...], ...]
    heights = [3.25, 1]
    fig, ax = plt.subplots(
        2,
        3,
        figsize=(11.5, 5.5),
        gridspec_kw=dict(
            width_ratios=[1, 1, 1],
            height_ratios=heights,
            hspace=0.15,
            wspace=0.020,
        ),
    )

    counts, errors, datagram, total_mc, uncertainty = region_plot_raw_material(
        rex_dir,
        "reg1j1b",
        stage,
        "tW",
    )
    bin_edges = datagram.edges
    canvas_from_counts(
        counts,
        errors,
        bin_edges,
        uncertainty=uncertainty,
        total_mc=total_mc,
        mpl_triplet=(fig, ax[0][0], ax[1][0]),
        combine_minor=True,
    )

    counts, errors, datagram, total_mc, uncertainty = region_plot_raw_material(
        rex_dir,
        "reg2j1b",
        stage,
        "tW",
    )
    bin_edges = datagram.edges
    canvas_from_counts(
        counts,
        errors,
        bin_edges,
        uncertainty=uncertainty,
        total_mc=total_mc,
        mpl_triplet=(fig, ax[0][1], ax[1][1]),
        combine_minor=True,
    )

    counts, errors, datagram, total_mc, uncertainty = region_plot_raw_material(
        rex_dir,
        "reg2j2b",
        stage,
        "tW",
    )
    bin_edges = datagram.edges
    canvas_from_counts(
        counts,
        errors,
        bin_edges,
        uncertainty=uncertainty,
        total_mc=total_mc,
        mpl_triplet=(fig, ax[0][2], ax[1][2]),
        combine_minor=True,
    )

    legend_last_to_first(ax[0][2], ncol=1, loc="upper right")
    draw_atlas_label(
        ax[0][0],
        follow_shift=0.280,
        extra_lines=[meta_text("reg1j1b", stage)],
        follow="",
    )

    y1, y2 = ax[0][1].get_ylim()
    y2 *= 0.7
    ax[0][0].set_ylim([y1, y2])
    ax[0][1].set_ylim([y1, y2])
    ax[0][2].set_ylim([y1, y2])

    ax[0][0].set_xticklabels([])
    ax[0][1].set_xticklabels([])
    ax[0][2].set_xticklabels([])

    ax[0][1].set_yticklabels([])
    ax[0][2].set_yticklabels([])
    ax[1][1].set_yticklabels([])
    ax[1][2].set_yticklabels([])

    ax[0][0].set_ylabel("Events", ha="right", y=1.0)
    ax[1][2].set_xlabel("BDT Response", ha="right", x=1.0)
    ax[1][0].set_ylabel("Data/MC")

    # ax[1][0].set_xticks([0.4, 0.5, 0.6, 0.7])
    ax[1][0].set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax[1][1].set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
    # ax[1][1].set_xticks([0.25, 0.35, 0.45, 0.55, 0.65])
    ax[1][2].set_xticks([0.5, 0.6, 0.7])
    # ax[1][2].set_xticks([0.5, 0.55, 0.6, 0.65, 0.7])
    # ax[1][2].set_xticks([0.5, 0.55, 0.6, 0.65, 0.7, 0.75])

    ax[0][1].text(0.05, 0.925, "2j1b", transform=ax[0][1].transAxes, fontsize=14)
    ax[0][2].text(0.05, 0.925, "2j2b", transform=ax[0][2].transAxes, fontsize=14)

    ax[1][0].set_xlim([0.3, ax[1][0].get_xlim()[1]])
    ax[0][0].set_xlim([0.3, ax[1][0].get_xlim()[1]])

    ax[1][1].set_xlim([0.15, ax[1][1].get_xlim()[1]])
    ax[0][1].set_xlim([0.15, ax[1][1].get_xlim()[1]])

    fig.subplots_adjust(left=0.075)

    if not os.path.exists(rd / "matplotlib"):
        os.mkdir(rd / "matplotlib")

    fig.savefig(rd / "matplotlib" / f"allregions_{stage}.pdf")
    fig.savefig(rd / "matplotlib" / f"allregions_{stage}.png")


if __name__ == "__main__":
    rd = pathlib.Path(sys.argv[1])
    cramped(rd, stage="pre")
    cramped(rd, stage="post")

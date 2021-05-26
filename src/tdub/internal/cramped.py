from tdub.art import canvas_from_counts, legend_last_to_first, draw_atlas_label
from tdub.rex import meta_text
from tdub.rex import region_plot_raw_material

import matplotlib.pyplot as plt


def cramped():
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
        "tW",
        "reg1j1b",
        "pre",
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
        "tW",
        "reg2j1b",
        "pre",
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
        "tW",
        "reg2j2b",
        "pre",
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
        follow_shift=0.25,
        extra_lines=[meta_text("reg1j1b", "pre")],
        follow="Internal",
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

    ax[0][0].set_ylabel("Events", ha="right", x=1.0)
    ax[1][2].set_xlabel("BDT Response", ha="right", x=1.0)

    fig.savefig("cramped.pdf")


if __name__ == "__main__":
    cramped()

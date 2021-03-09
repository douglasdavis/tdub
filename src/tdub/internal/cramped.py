from tdub.art import canvas_from_counts, legend_last_to_first, draw_atlas_label
from tdub.rex import meta_text
from tdub.rex import region_plot_raw_material

import matplotlib.pyplot as plt

if __name__ == "__main__":

    heights = [3.25, 1]
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(13.5, 5.5),
        gridspec_kw=dict(
            width_ratios=[1, 1, 1],
            height_ratios=heights,
            hspace=0.025,
            wspace=0.010,
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
        mpl_triplet=(fig, axs[0][0], axs[1][0]),
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
        mpl_triplet=(fig, axs[0][1], axs[1][1]),
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
        mpl_triplet=(fig, axs[0][2], axs[1][2]),
    )

    legend_last_to_first(axs[0][2], ncol=2, loc="upper right")
    draw_atlas_label(
        axs[0][0],
        follow_shift=0.25,
        extra_lines=[meta_text("reg1j1b", "pre")],
        follow="Internal",
    )

    y1, y2 = axs[0][1].get_ylim()
    y2 *= 0.7
    axs[0][0].set_ylim([y1, y2])
    axs[0][1].set_ylim([y1, y2])
    axs[0][2].set_ylim([y1, y2])

    axs[0][0].set_xticklabels([])
    axs[0][1].set_xticklabels([])
    axs[0][2].set_xticklabels([])

    axs[0][1].set_yticklabels([])
    axs[0][2].set_yticklabels([])
    axs[1][1].set_yticklabels([])
    axs[1][2].set_yticklabels([])


    fig.savefig("cramped.pdf")

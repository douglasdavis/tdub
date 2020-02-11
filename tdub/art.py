"""
Art utilities
"""

# stdlib
from typing import Any, Dict, Optional, Tuple

# external
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# tdub
from tdub.utils import bin_centers


def setup_style():
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.figsize"] = (6, 5.25)
    matplotlib.rcParams["axes.labelsize"] = 14
    matplotlib.rcParams["font.size"] = 12
    matplotlib.rcParams["xtick.top"] = True
    matplotlib.rcParams["ytick.right"] = True
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.labelsize"] = 12
    matplotlib.rcParams["ytick.labelsize"] = 12
    matplotlib.rcParams["xtick.minor.visible"] = True
    matplotlib.rcParams["ytick.minor.visible"] = True
    matplotlib.rcParams["xtick.major.width"] = 0.8
    matplotlib.rcParams["xtick.minor.width"] = 0.8
    matplotlib.rcParams["xtick.major.size"] = 7.0
    matplotlib.rcParams["xtick.minor.size"] = 4.0
    matplotlib.rcParams["xtick.major.pad"] = 1.5
    matplotlib.rcParams["xtick.minor.pad"] = 1.4
    matplotlib.rcParams["ytick.major.width"] = 0.8
    matplotlib.rcParams["ytick.minor.width"] = 0.8
    matplotlib.rcParams["ytick.major.size"] = 7.0
    matplotlib.rcParams["ytick.minor.size"] = 4.0
    matplotlib.rcParams["ytick.major.pad"] = 1.5
    matplotlib.rcParams["ytick.minor.pad"] = 1.4
    matplotlib.rcParams["legend.frameon"] = False
    matplotlib.rcParams["legend.numpoints"] = 1
    matplotlib.rcParams["legend.fontsize"] = 11
    matplotlib.rcParams["legend.handlelength"] = 1.5
    matplotlib.rcParams["axes.formatter.limits"] = [-4, 4]
    matplotlib.rcParams["axes.formatter.use_mathtext"] = True


def canvas_from_counts(
    counts: Dict[str, np.ndarray],
    errors: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    stack_error_band: Optional[Any] = None,
    ratio_error_band: Optional[Any] = None,
    **subplots_kw,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """create a plot canvas given a dictionary of counts and bin edges

    The ``counts`` and ``errors`` dictionaries are expected to have
    the following keys:

    - ``"Data"``
    - ``"tW_DR"``
    - ``"ttbar"``
    - ``"Zjets"``
    - ``"Diboson"``
    - ``"MCNP"``

    Parameters
    ----------
    counts : dict(str, np.ndarray)
        a dictionary pairing samples to bin counts
    errors : dict(str, np.ndarray)
        a dictionray pairing samples to bin count errors
    bin_edges : np.ndarray
        the histogram bin edges
    stack_error_band : Any, optional
        todo
    ratio_error_band : Any, optional
        todo
    subplots_kw : dict
        remaining keyword arguments passed to :py:func:`matplotlib.pyplot.subplots`

    Returns
    -------
    fig : :obj:`matplotlib.figure.Figure`
        the matplotlib figure
    ax : :obj:`matplotlib.axes.Axes`
        the matplotlib axes for the histogram stack
    axr : :obj:`matplotlib.axes.Axes`
        the matplotlib axes for the ratio comparison

    """
    centers = bin_centers(bin_edges)
    start, stop = bin_edges[0], bin_edges[-1]
    mc_counts = np.zeros_like(centers, dtype=np.float32)
    mc_errs = np.zeros_like(centers, dtype=np.float32)
    for key in counts.keys():
        if key != "Data":
            mc_counts += counts[key]
            mc_errs += errors[key] ** 2
    mc_errs = np.sqrt(mc_errs)
    ratio = counts["Data"] / mc_counts
    ratio_err = counts["Data"] / (mc_counts ** 2) + np.power(
        counts["Data"] * mc_errs / (mc_counts ** 2), 2
    )
    fig, (ax, axr) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw=dict(height_ratios=[3.25, 1], hspace=0.025),
        **subplots_kw,
    )
    ax.hist(
        [centers for _ in range(5)],
        bins=bin_edges,
        weights=[
            counts["MCNP"],
            counts["Diboson"],
            counts["Zjets"],
            counts["ttbar"],
            counts["tW_DR"],
        ],
        histtype="stepfilled",
        stacked=True,
        label=["MCNP", "Diboson", "$Z$+jets", "$t\\bar{t}$", "$tW$"],
        color=["#9467bd", "#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"],
    )
    ax.errorbar(
        centers, counts["Data"], yerr=errors["Data"], label="Data", fmt="ko", zorder=999
    )
    axr.plot([start, stop], [1.0, 1.0], color="gray", linestyle="solid", marker=None)
    axr.errorbar(centers, ratio, yerr=ratio_err, fmt="ko", zorder=999)
    axr.set_ylim([0.75, 1.25])
    axr.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])

    return fig, ax, axr

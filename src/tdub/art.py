"""Art creation utilities."""

# stdlib
from typing import Any, Dict, Optional, Tuple, List
from pathlib import PosixPath
import logging
import os

# external
import numpy as np
import matplotlib.pyplot as plt
import uproot
from uproot.rootio import ROOTDirectory

# tdub
from tdub import setup_logging
import tdub._art
import tdub.hist


setup_logging()
log = logging.getLogger(__name__)


def setup_tdub_style():
    """Modify matplotlib's rcParams."""
    tdub._art.setup_style()


def adjust_figure(
    fig: plt.Figure,
    left: float = 0.125,
    bottom: float = 0.095,
    right: float = 0.965,
    top: float = 0.95,
) -> None:
    """Adjust a matplotlib Figure with nice defaults."""
    NotImplementedError("This is TODO")


def canvas_from_counts(
    counts: Dict[str, np.ndarray],
    errors: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    stack_error_band: Optional[Any] = None,
    ratio_error_band: Optional[Any] = None,
    **subplots_kw,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """create a plot canvas given a dictionary of counts and bin edges.

    The ``counts`` and ``errors`` dictionaries are expected to have
    the following keys:

    - ``"Data"``
    - ``"tW_DR"`` or ``"tW"``
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
    :py:obj:`matplotlib.figure.Figure`
        the matplotlib figure
    :py:obj:`matplotlib.axes.Axes`
        the matplotlib axes for the histogram stack
    :py:obj:`matplotlib.axes.Axes`
        the matplotlib axes for the ratio comparison
    """
    tW_name = "tW_DR"
    if tW_name not in counts.keys():
        tW_name = "tW"
    centers = tdub.hist.bin_centers(bin_edges)
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
            counts[tW_name],
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

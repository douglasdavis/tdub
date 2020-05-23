"""Art creation utilities."""

# stdlib
from typing import Dict, Tuple, Optional  # noqa
import logging  # noqa

# external
import matplotlib  # noqa

matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
from uproot_methods.classes.TGraphAsymmErrors import Methods as ROOT_TGraphAsymmErrors  # noqa
from uproot_methods.classes.TH1 import Methods as ROOT_TH1  # noqa
# tdub
from tdub import setup_logging  # noqa
import tdub._art  # noqa
import tdub.hist  # noqa


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
    NotImplementedError("TODO")


# WIP
def draw_ratio_errorband(
    errorband: ROOT_TGraphAsymmErrors,
    total_mc: ROOT_TH1,
    ax: plt.Axes,
    axr: plt.Axes,
    label="Total Unc.",
) -> None:
    """Draw ratio bands on axes.

    Parameters
    ----------
    errorband : uproot_methods.classes.TGraphAsymmErrors.Methods

    """
    lo = np.hstack([errorband.yerrorslow, errorband.yerrorslow[-1]])
    hi = np.hstack([errorband.yerrorshigh, errorband.yerrorshigh[-1]])
    mc = np.hstack([total_mc.values, total_mc.values[-1]])
    ax.fill_between(
        x=total_mc.edges,
        y1=(mc - lo),
        y2=(mc + hi),
        step="post",
        facecolor="none",
        hatch="////",
        edgecolor="cornflowerblue",
        linewidth=0.0,
        label=label,
        zorder=50,
    )
    axr.fill_between(
        x=total_mc.edges,
        y1=(1 - lo / mc),
        y2=(1 + hi / mc),
        step="post",
        facecolor=(0, 0, 0, 0.33),
        linewidth=0.0,
        label=label,
        zorder=50,
    )


def canvas_from_counts(
    counts: Dict[str, np.ndarray],
    errors: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    errorband: Optional[ROOT_TGraphAsymmErrors] = None,
    totalmc: Optional[ROOT_TH1] = None,
    logy: bool = False,
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
    errorband : uproot_methods.base.ROOTMethods, optional
        Errorband (TGraphAsym)
    totalmc : uproot_methods.base.ROOTMethods, optional
        Total MC histogram (TH1D)
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
    ratio_err = np.sqrt(counts["Data"] / (mc_counts ** 2) + np.power(
        counts["Data"] * mc_errs / (mc_counts ** 2), 2)
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
        color=["#9467bd",  "#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"],
    )
    ax.errorbar(
        centers, counts["Data"], yerr=errors["Data"], label="Data", fmt="ko", zorder=999
    )
    axr.plot([start, stop], [1.0, 1.0], color="gray", linestyle="solid", marker=None)
    axr.errorbar(centers, ratio, yerr=ratio_err, fmt="ko", zorder=999)
    axr.set_ylim([0.8, 1.2])
    axr.set_yticks([0.8, 0.9, 1.0, 1.1])

    if errorband is not None and totalmc is not None:
        draw_ratio_errorband(errorband, totalmc, ax, axr)

    axr.set_xlim([bin_edges[0], bin_edges[-1]])
    if logy:
        ax.set_ylim([1, ax.get_ylim()[1] * 100])
    else:
        ax.set_ylim([0, ax.get_ylim()[1] * 1.375])

    return fig, ax, axr

"""Art creation utilities."""

# stdlib
from typing import Any, Dict, Tuple, Optional, List, Union
import logging

# external
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from uproot_methods.classes.TGraphAsymmErrors import Methods as ROOT_TGraphAsymmErrors
from uproot_methods.classes.TH1 import Methods as ROOT_TH1

# tdub
from tdub import setup_logging
import tdub.hist


setup_logging()
log = logging.getLogger(__name__)


def adjust_figure(
    fig: plt.Figure,
    left: float = 0.125,
    bottom: float = 0.095,
    right: float = 0.965,
    top: float = 0.95,
) -> None:
    """Adjust a matplotlib Figure with nice defaults."""
    NotImplementedError("TODO")


def legend_last_to_first(ax: plt.Axes, **kwargs):
    """Move the last element of the legend to first

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to create a legend on.
    kwargs : dict
        Arguments passed to :py:obj:`matplotlib.axes.Axes.legend`.

    """
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, handles.pop())
    labels.insert(0, labels.pop())
    ax.legend(handles, labels, **kwargs)


def draw_atlas_label(
    ax: plt.Axes,
    follow: str = "Internal",
    cme_and_lumi: bool = True,
    extra_lines: Optional[List[str]] = None,
    cme: Union[int, float] = 13,
    lumi: float = 139,
    x: float = 0.050,
    y: float = 0.905,
    follow_shift: float = 0.15,
    s1: int = 14,
    s2: int = 12,
) -> None:
    """Draw the ATLAS label text, with extra lines if desired.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the text on.
    follow : str
        Text to follow the ATLAS label (usually 'Internal').
    extra_lines : list(str), optional
        Set of extra lines to draw below ATLAS label.
    cme : int or float
        Center-of-mass energy.
    lumi : int or float
        Integrated luminosity of the data.
    x : float
        `x`-location of the text.
    y : float
        `y`-location of the text.
    follow_shift : float
        `x`-shift of the text following the ATLAS label.
    s1 : int
        Size of the main label.
    s2 : int
        Size of the extra text

    """
    ax.text(
        x,
        y,
        "ATLAS",
        fontstyle="italic",
        fontweight="bold",
        transform=ax.transAxes,
        size=s1,
    )
    if follow:
        ax.text(x + follow_shift, y, follow, transform=ax.transAxes, size=s1)
    if cme_and_lumi:
        exlines = [f"$\\sqrt{{s}}$ = {cme} TeV, $L = {lumi}$ fb$^{{-1}}$"]
    else:
        exlines = []
    if extra_lines is not None:
        exlines += extra_lines
    for i, exline in enumerate(exlines):
        ax.text(x, y - (i + 1) * 0.06, exline, transform=ax.transAxes, size=s2)


def draw_uncertainty_bands(
    uncertainty: ROOT_TGraphAsymmErrors,
    total_mc: ROOT_TH1,
    ax: plt.Axes,
    axr: plt.Axes,
    label: str = "Uncertainty",
    edgecolor: Any = "mediumblue",
    zero_threshold: float = 0.25,
) -> None:
    """Draw uncertainty bands on both axes in stack plot with a ratio.

    Parameters
    ----------
    uncertainty : uproot_methods.classes.TGraphAsymmErrors.Methods
        ROOT TGraphAsymmErrors from uproot with full systematic uncertainty.
    total_mc : uproot_methods.classes.TH1.Methods
        ROOT TH1 from uproot providing the full Monte Carlo prediction.
    ax : matplotlib.axes.Axes
        Main axis (where histogram stack is painted)
    axr : matplotlib.axes.Axes
        Ratio axis
    label : str
        Legend label for the uncertainty.
    zero_threshold : float
        When total MC events are below threshold, zero contents and error.

    """
    lo = np.hstack([uncertainty.yerrorslow, uncertainty.yerrorslow[-1]])
    hi = np.hstack([uncertainty.yerrorshigh, uncertainty.yerrorshigh[-1]])
    mc = np.hstack([total_mc.values, total_mc.values[-1]])
    ratio_y1 = 1 - (lo / mc)
    ratio_y2 = 1 + (hi / mc)
    set_to_zero = mc < zero_threshold
    lo[set_to_zero] = 0.0
    hi[set_to_zero] = 0.0
    mc[set_to_zero] = 0.0
    ratio_y1[set_to_zero] = 0.0
    ratio_y2[set_to_zero] = 0.0
    ax.fill_between(
        x=total_mc.edges,
        y1=(mc - lo),
        y2=(mc + hi),
        step="post",
        facecolor="none",
        hatch="////",
        edgecolor=edgecolor,
        linewidth=0.0,
        label=label,
        zorder=50,
    )
    axr.fill_between(
        x=total_mc.edges,
        y1=ratio_y1,
        y2=ratio_y2,
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
    uncertainty: Optional[ROOT_TGraphAsymmErrors] = None,
    total_mc: Optional[ROOT_TH1] = None,
    logy: bool = False,
    **subplots_kw,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a plot canvas given a dictionary of counts and bin edges.

    The ``counts`` and ``errors`` dictionaries are expected to have
    the following keys:

    - `"Data"`
    - `"tW_DR"` or `"tW"`
    - `"ttbar"`
    - `"Zjets"`
    - `"Diboson"`
    - `"MCNP"`

    Parameters
    ----------
    counts : dict(str, np.ndarray)
        a dictionary pairing samples to bin counts.
    errors : dict(str, np.ndarray)
        a dictionray pairing samples to bin count errors.
    bin_edges : array_like
        the histogram bin edges.
    uncertainty : uproot_methods.base.classes.TGraphAsymmErrors.Methods, optional
        Uncertainty (TGraphAsym).
    total_mc : uproot_methods.base.classes.TH1.Methods, optional
        Total MC histogram (TH1D).
    subplots_kw : dict
        remaining keyword arguments passed to :py:func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    :py:obj:`matplotlib.figure.Figure`
        Matplotlib figure.
    :py:obj:`matplotlib.axes.Axes`
        Matplotlib axes for the histogram stack.
    :py:obj:`matplotlib.axes.Axes`
        Matplotlib axes for the ratio comparison.

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
    ratio_err = np.sqrt(
        counts["Data"] / (mc_counts ** 2)
        + np.power(counts["Data"] * mc_errs / (mc_counts ** 2), 2)
    )
    fig, (ax, axr) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw=dict(height_ratios=[3.25, 1], hspace=0.025),
        **subplots_kw,
    )
    ax.errorbar(
        centers, counts["Data"], yerr=errors["Data"], label="Data", fmt="ko", zorder=999
    )

    # colors = ["#9467bd", "#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"]
    colors = ["#9467bd", "#2ca02c", "#ff7f0e", "#9d0000", "#1f77b4"]
    labels = ["Non-prompt", "Diboson", "$Z$+jets", "$t\\bar{t}$", "$tW$"]

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
        label=labels,
        color=colors,
    )
    axr.plot([start, stop], [1.0, 1.0], color="gray", linestyle="solid", marker=None)
    axr.errorbar(centers, ratio, yerr=ratio_err, fmt="ko", zorder=999)
    axr.set_ylim([0.8, 1.2])
    axr.set_yticks([0.8, 0.9, 1.0, 1.1])

    if uncertainty is not None and total_mc is not None:
        draw_uncertainty_bands(uncertainty, total_mc, ax, axr)

    axr.set_xlim([bin_edges[0], bin_edges[-1]])
    if logy:
        ax.set_yscale("log")
        ax.set_ylim([5, ax.get_ylim()[1] * 100])
    else:
        ax.set_ylim([0, ax.get_ylim()[1] * 1.375])

    return fig, ax, axr


def draw_impact_barh(ax, df, hi_color="skyblue", lo_color="peru") -> Tuple[plt.Axes, plt.Axes]:
    """Draw the impact plot."""
    ys = np.array(df.ys)
    ax.barh(
        ys,
        df.pre_down.abs(),
        left=df.pre_down_lefts,
        fill=False,
        edgecolor=lo_color,
        zorder=5,
        label=r"Prefit $\theta=\hat{\theta}-\Delta\theta$",
    )
    ax.barh(
        ys,
        df.pre_up.abs(),
        left=df.pre_up_lefts,
        fill=False,
        edgecolor=hi_color,
        zorder=5,
        label=r"Prefit $\theta=\hat{\theta}+\Delta\theta$",
    )
    ax.barh(
        ys,
        df.post_down.abs(),
        left=df.post_down_lefts,
        fill=True,
        color=lo_color,
        zorder=6,
        label=r"Postfit $\theta=\hat{\theta}-\Delta\theta$",
    )
    ax.barh(
        ys,
        df.post_up.abs(),
        left=df.post_up_lefts,
        fill=True,
        color=hi_color,
        zorder=6,
        label=r"Postfit $\theta=\hat{\theta}+\Delta\theta$",
    )
    xlims = np.amax([np.abs(df.pre_down), np.abs(df.pre_up)]) * 1.25
    ax.set_xlim([-xlims, xlims])
    ax.set_yticks(ys)
    ax2 = ax.twiny()
    ax2.errorbar(
        df.central,
        ys,
        xerr=[np.abs(df.sig_lo), df.sig_hi],
        fmt="ko",
        zorder=999,
        label="Nuisance Parameter Pull",
    )
    ax2.set_xlim([-1.8, 1.8])
    ax2.plot([-1, -1], [-0.5, ys[-1] + 0.5], ls="--", color="black")
    ax2.plot([1, 1], [-0.5, ys[-1] + 0.5], ls="--", color="black")
    ax2.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("top")
    return ax, ax2


def setup_tdub_style() -> None:
    """Modify matplotlib's rcParams."""
    matplotlib.rcParams["font.sans-serif"] = [
        "Helvetica",
        "helvetica",
        "Nimbus Sans L",
        "FreeSans",
    ]
    matplotlib.rcParams["axes.formatter.limits"] = [-4, 4]
    matplotlib.rcParams["axes.formatter.use_mathtext"] = True
    matplotlib.rcParams["axes.labelsize"] = 15
    matplotlib.rcParams["figure.figsize"] = (5.8, 5.2)
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["figure.subplot.left"] = 0.125
    matplotlib.rcParams["figure.subplot.bottom"] = 0.1075
    matplotlib.rcParams["figure.subplot.right"] = 0.965
    matplotlib.rcParams["figure.subplot.top"] = 0.95
    matplotlib.rcParams["font.size"] = 13
    matplotlib.rcParams["legend.frameon"] = False
    matplotlib.rcParams["legend.numpoints"] = 1
    matplotlib.rcParams["legend.fontsize"] = 11
    matplotlib.rcParams["legend.handlelength"] = 1.5
    matplotlib.rcParams["lines.linewidth"] = 1
    matplotlib.rcParams["xtick.top"] = True
    matplotlib.rcParams["ytick.right"] = True
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.labelsize"] = 13
    matplotlib.rcParams["ytick.labelsize"] = 13
    matplotlib.rcParams["xtick.minor.visible"] = True
    matplotlib.rcParams["ytick.minor.visible"] = True
    matplotlib.rcParams["xtick.major.width"] = 0.8
    matplotlib.rcParams["xtick.minor.width"] = 0.8
    matplotlib.rcParams["xtick.major.size"] = 7.0
    matplotlib.rcParams["xtick.minor.size"] = 4.0
    matplotlib.rcParams["xtick.major.pad"] = 3.5
    matplotlib.rcParams["xtick.minor.pad"] = 3.4
    matplotlib.rcParams["ytick.major.width"] = 0.8
    matplotlib.rcParams["ytick.minor.width"] = 0.8
    matplotlib.rcParams["ytick.major.size"] = 7.0
    matplotlib.rcParams["ytick.minor.size"] = 4.0
    matplotlib.rcParams["ytick.major.pad"] = 3.5
    matplotlib.rcParams["ytick.minor.pad"] = 3.4

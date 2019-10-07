from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pygram11

from ._art import setup_style
from .utils import bin_centers


def draw_rocs(
    frs: List[FoldedResult],
    ax: Optional[matplotlib.axes.Axes] = None,
    labels: Optional[List[str]] = None,
    draw_guess: bool = False,
    draw_grid: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """draw ROC curves from a set of folded training results

    Parameters
    ----------
    frs : list(FoldedResult)
       the set of folded training results to plot
    ax : :py:obj:`matplotlib.axes.Axes`, optional
       an existing matplotlib axis to plot on
    labels : list(str)
       a label for each training, defaults to use the region
       associated with each folded result
    draw_guess : bool
       draw a straight line from (0, 0) to (1, 1) to represent a 50/50
       (guess) ROC curve.
    draw_grid : bool
       draw a grid on the axis

    Returns
    -------
    :py:obj:`matplotlib.figure.Figure`
       the figure associated with the axis
    :py:obj:`matplotlib.axes.Axes`
       the axis object which has the plot

    Examples
    --------

    >>> from tdub.apply import FoldedResult
    >>> from tdub.rawart import draw_rocs
    >>> fr_1j1b = FoldedResult("/path/to/train_1j1b")
    >>> fr_2j1b = FoldedResult("/path/to/train_2j1b")
    >>> fr_2j2b = FoldedResult("/path/to/train_2j2b")
    >>> fig, ax = draw_rocs([fr_1j1b, fr_2j1b, fr_2j2b])

    """
    if labels is None:
        labels = [str(fr.region) for fr in frs]

    if ax is None:
        setup_style()
        fig, ax = plt.subplots()

    for label, fr in zip(labels, frs):
        x = fr.summary["roc"]["mean_fpr"]
        y = fr.summary["roc"]["mean_tpr"]
        auc = fr.summary["roc"]["auc"]
        ax.plot(x, y, label=f"{label}, AUC: {auc:0.2f}", lw=2, alpha=0.9)

    if draw_guess:
        ax.plot([0, 1.0], [0, 1.0], lw=1, alpha=0.4, ls="--", color="k")

    if draw_grid:
        ax.grid(alpha=0.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="best")
    return ax.figure, ax


def draw_stack(
    *,
    data_df: pandas.DataFrame,
    mc_dfs: Iterable[pandas.DataFrame],
    distribution: str,
    bins: int,
    range: Tuple[float, float],
    weight_name: str = "weight_nominal",
    colors: Optional[Iterable[str]] = None,
    mc_labels: Optional[Iterable[str]] = None,
    lumi: float = 139.0,
    legend_ncol: int = 2,
    y_scalefac: float = 1.35,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes]:
    """draw a standard histogram stack for some distribution and selection

    create the stacked distribution of a particular variable after
    applying a selection to all data and MC.

    Parameters
    ----------
    data_df : pandas.DataFrame
       the dataframe for data
    mc_dfs : list(pandas.DataFrame)
       the list of MC dataframes
    distribution: str
       the variable to histogram
    bins : int
       the number of bins
    range : tuple(float, float)
       the range to histogram the distribution
    weight_name : str
       the name of the weight column
    colors : list(str), optional
       the colors for the Monte Carlo histograms
    mc_labels : list(str)
       the list of labels for the legend
    lumi : float
       the luminosity for the data (to scale the MC)
    legend_ncol : int
       number of columns for the legend
    y_scalefac : float
       factor to scale the default maximum y-axis range by

    Returns
    -------
    :py:obj:`matplotlib.figure.Figure`
       the figure associated with the axes
    :py:obj:`matplotlib.axes.Axes`
       the main axis object which has the plot
    :py:obj:`matplotlib.axes.Axes`
       the axis object which has the ratio plot

    """
    edges = np.linspace(range[0], range[1], bins + 1)
    centers = bin_centers(edges)

    data_count, __ = pygram11.histogram(
        data_df[distribution].to_numpy(), bins=bins, range=range, flow=True
    )
    data_err = np.sqrt(data_count)
    mc_dists = [df[distribution].to_numpy() for df in mc_dfs]
    mc_ws = [df[weight_name].to_numpy() * lumi for df in mc_dfs]
    mc_hists = [
        pygram11.histogram(mcd, weights=mcw, bins=bins, range=range, flow=True)
        for mcd, mcw in zip(mc_dists, mc_ws)
    ]
    mc_counts = [mcc[0] for mcc in mc_hists]
    mc_errs = [mcc[1] for mcc in mc_hists]
    mc_total = np.sum(mc_counts, axis=0)
    ratio = data_count / mc_total
    mc_total_err = np.sqrt(np.sum([mce ** 2 for mce in mc_errs], axis=0))
    ratio_err = data_count / (mc_total ** 2) + np.power(
        data_count * mc_total_err / (mc_total ** 2), 2
    )

    if colors is None:
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
        colors.reverse()
    if mc_labels is None:
        mc_labels = ["$tW$", "$t\\bar{t}$", "Diboson", "$Z+$jets", "MCNP"]
        mc_labels.reverse()

    setup_style()
    fig, (ax, axr) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(6, 5.25),
        gridspec_kw=dict(height_ratios=[3.25, 1], hspace=0.025),
    )

    ax.hist(
        [centers for _ in mc_labels],
        weights=mc_counts,
        bins=edges,
        histtype="stepfilled",
        label=mc_labels,
        color=colors,
        stacked=True,
    )
    ax.errorbar(centers, data_count, yerr=data_err, fmt="ko", label="Data", zorder=999)

    ax.set_ylim([0, ax.get_ylim()[1] * y_scalefac])

    ax.legend(loc="upper right")
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, handles.pop())
    labels.insert(0, labels.pop())
    ax.legend(handles, labels, loc="upper right", ncol=legend_ncol)

    axr.errorbar(centers, ratio, yerr=ratio_err, fmt="ko", zorder=999)
    axr.plot([range[0], range[1]], [1, 1], color="gray", linestyle="solid", marker=None)
    axr.set_ylim([0.8, 1.2])
    axr.set_yticks([0.9, 1.0, 1.1])
    axr.autoscale(enable=True, axis="x", tight=True)

    return ax.figure, ax, axr

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt

from ._art import setup_style


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

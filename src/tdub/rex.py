"""Utilities for parsing TRExFitter."""

# stdlib
import logging
import os
from pathlib import PosixPath
from typing import Dict, Iterable, List, Tuple, Union

# external
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from uproot.rootio import ROOTDirectory
from uproot_methods.classes.TGraphAsymmErrors import Methods as ROOT_TGraphAsymmErrors
from uproot_methods.classes.TH1 import Methods as ROOT_TH1
import yaml

# tdub
from .art import (
    canvas_from_counts,
    setup_tdub_style,
    draw_atlas_label,
    legend_last_to_first,
)
import tdub.config

setup_tdub_style()

log = logging.getLogger(__name__)


def available_regions(wkspace: Union[str, os.PathLike]) -> List[str]:
    """Get a list of available regions from a workspace.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace

    Returns
    -------
    list(str)
        Regions discovered in the workspace.
    """
    root_files = (PosixPath(wkspace) / "Histograms").glob("*_preFit.root")
    return [rf.name[:-12] for rf in root_files]


def data_histogram(
    wkspace: Union[str, os.PathLike], region: str, fitname: str = "tW"
) -> ROOT_TH1:
    """Get the histogram for the Data in a region from a workspace.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace
    region : str
        TRExFitter region name.
    fitname : str
        Name of the Fit

    Returns
    -------
    uproot_methods.base.ROOTMethods
        ROOT histogram for the Data sample.
    """
    root_path = PosixPath(wkspace) / "Histograms" / f"{fitname}_{region}_histos.root"
    return uproot.open(root_path).get(f"{region}_Data")


def chisq(
    wkspace: Union[str, os.PathLike], region: str, stage: str = "pre"
) -> Tuple[float, int, float]:
    r"""Get prefit :math:`\chi^2` information from TRExFitter region.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace
    region : str
        TRExFitter region name.
    stage : str
        Drawing fit stage, ('pre' or 'post').

    Returns
    -------
    float
        :math:`\chi^2` value for the region.
    int
        Number of degrees of freedom.
    float
        :math:`\chi^2` probability for the region.
    """
    if stage not in ("pre", "post"):
        raise ValueError("stage can only be 'pre' or 'post'")
    txt_path = PosixPath(wkspace) / "Histograms" / f"{region}_{stage}Fit_Chi2.txt"
    table = yaml.full_load(txt_path.read_text())
    return table["chi2"], table["ndof"], table["probability"]


def chisq_text(wkspace: Union[str, os.PathLike], region: str, stage: str = "pre") -> None:
    r"""Generate nicely formatted text for :math:`\chi^2` information.

    Deploys the :py:func:`tdub.rex.chisq` for grab the info.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace
    region : str
        TRExFitter region name.
    stage : str
        Drawing fit stage, ('pre' or 'post').

    Returns
    -------
    str
        Formatted string showing the :math:`\chi^2` information.

    """
    chi2, ndof, prob = chisq(wkspace, region, stage=stage)
    return (
        f"$\\chi^2/\\mathrm{{ndf}} = {chi2:3.2f} / {ndof}$, "
        f"$\\chi^2_{{\\mathrm{{prob}}}} = {prob:3.2f}$"
    )


def prefit_histogram(root_file: ROOTDirectory, sample: str, region: str) -> ROOT_TH1:
    """Get a prefit histogram from a file.

    Parameters
    ----------
    root_file : root.rootio.ROOTDirectory
        File containing the desired prefit histogram.
    sample : str
        Physics sample name.
    region : str
        TRExFitter region name.

    Returns
    -------
    uproot_methods.classes.TH1.Methods
        ROOT histogram.
    """
    histname = f"{region}_{sample}"
    try:
        h = root_file.get(histname)
        return h
    except KeyError:
        log.fatal("%s histogram not found in %s" % (histname, root_file))
        exit(1)


def prefit_histograms(
    wkspace: Union[str, os.PathLike],
    samples: Iterable[str],
    region: str,
    fitname: str = "tW",
) -> Dict[str, ROOT_TH1]:
    """Retrieve sample prefit histograms for a region.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace
    samples : Iterable(str)
        Physics samples of the desired histograms
    region : str
        Region to get histograms for
    fitname : str
        Name of the Fit

    Returns
    -------
    dict(str, uproot_methods.classes.TH1.Methods)
        Prefit ROOT histograms
    """

    root_path = PosixPath(wkspace) / "Histograms" / f"{fitname}_{region}_histos.root"
    root_file = uproot.open(root_path)
    histograms = {}
    for samp in samples:
        h = prefit_histogram(root_file, samp, region)
        if h is None:
            log.warn("Histogram for sample %s in region: %s not found" % (samp, region))
        histograms[samp] = h
    return histograms


def prefit_total_and_uncertainty(
    wkspace: Union[str, os.PathLike], region: str
) -> Tuple[ROOT_TGraphAsymmErrors, ROOT_TH1]:
    """Get the prefit total MC prediction and uncertainty band for a region.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace.
    region : str
        Region to get error band for.

    Returns
    -------
    uproot_methods.classes.TH1.Methods
        The total MC expectation histogram.
    uproot_methods.classes.TGraphAsymmErrors.Methods
        The error TGraph.
    """
    root_path = PosixPath(wkspace) / "Histograms" / f"{region}_preFit.root"
    root_file = uproot.open(root_path)
    err = root_file.get("g_totErr")
    tot = root_file.get("h_tot")
    return tot, err


def postfit_available(wkspace: Union[str, os.PathLike]) -> bool:
    """Check if TRExFitter workspace contains postFit information.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace

    Returns
    -------
    bool
        True of postFit discovered
    """
    histdir = PosixPath(wkspace) / "Histograms"
    for f in histdir.iterdir():
        if "postFit" in f.name:
            return True
    return False


def postfit_histogram(root_file: ROOTDirectory, sample: str) -> ROOT_TH1:
    """Get a postfit histogram from a file.

    Parameters
    ----------
    root_file : root.rootio.ROOTDirectory
        File containing the desired postfit histogram.
    sample : str
        Physics sample name.

    Returns
    -------
    uproot_methods.classes.TH1.Methods
        ROOT histogram.
    """
    histname = f"h_{sample}_postFit"
    try:
        h = root_file.get(histname)
        return h
    except KeyError:
        log.fatal("%s histogram not found in %s" % (histname, root_file))
        exit(1)


def postfit_histograms(
    wkspace: Union[str, os.PathLike], samples: Iterable[str], region: str
) -> Dict[str, ROOT_TH1]:
    """Retrieve sample postfit histograms for a region.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace
    region : str
        Region to get histograms for
    samples : Iterable(str)
        Physics samples of the desired histograms

    Returns
    -------
    dict(str, uproot_methods.classes.TH1.Methods)
        Postfit ROOT histograms
    """
    root_path = PosixPath(wkspace) / "Histograms" / f"{region}_postFit.root"
    root_file = uproot.open(root_path)
    histograms = {}
    for samp in samples:
        if samp == "Data":
            continue
        h = postfit_histogram(root_file, samp)
        if h is None:
            log.warn("Histogram for sample %s in region %s not found" % (samp, region))
        histograms[samp] = h
    return histograms


def postfit_total_and_uncertainty(
    wkspace: Union[str, os.PathLike], region: str
) -> Tuple[ROOT_TGraphAsymmErrors, ROOT_TH1]:
    """Get the postfit total MC prediction and uncertainty band for a region.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace.
    region : str
        Region to get error band for.

    Returns
    -------
    uproot_methods.classes.TH1.Methods
        The total MC expectation histogram.
    uproot_methods.classes.TGraphAsymmErrors.Methods
        The error TGraph.
    """
    root_path = PosixPath(wkspace) / "Histograms" / f"{region}_postFit.root"
    root_file = uproot.open(root_path)
    err = root_file.get("g_totErr_postFit")
    tot = root_file.get("h_tot_postFit")
    return tot, err


# WIP
def meta_text(region: str, stage: str) -> str:
    if stage == "pre":
        stage = "Pre-fit"
    elif stage == "post":
        stage = "Post-fit"
    else:
        raise ValueError("stage can be 'pre' or 'post'")
    if "1j1b" in region:
        region = "1j1b"
    elif "2j1b" in region:
        region = "2j1b"
    elif "2j2b" in region:
        region = "2j2b"
    else:
        raise ValueError("region must contain '1j1b', '2j1b', or '2j2b'")
    return f"$tW$ Dilepton, {region}, {stage}"


# WIP
def meta_axis_label(region: str) -> str:
    if tdub.config.PLOTTING_META_TABLE is None:
        raise ValueError("tdub.config.PLOTTING_META_TABLE must be defined")
    if "VRP" in region:
        region = region[12:]
    main_label = tdub.config.PLOTTING_META_TABLE["titles"][region]["mpl"]
    unit_label = tdub.config.PLOTTING_META_TABLE["titles"][region]["unit"]
    if not unit_label:
        return main_label
    else:
        return f"{main_label} [{unit_label}]"


# WIP
def stack_canvas(
    wkspace: Union[str, os.PathLike], region: str, stage: str = "pre", fitname: str = "tW",
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create a pre- or post-fit plot canvas for a TRExFitter region.

    Parameters
    ---------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace.
    region : str
        Region to get error band for.
    stage : str
        Drawing fit stage, ('pre' or 'post').
    fitname : str
        Name of the Fit

    Returns
    -------
    :py:obj:`matplotlib.figure.Figure`
        Figure for housing the plot.
    :py:obj:`matplotlib.axes.Axes`
        Main axes for the histogram stack.
    :py:obj:`matplotlib.axes.Axes`
        Ratio axes to show Data/MC.
    """
    samples = ("tW", "ttbar", "Zjets", "Diboson", "MCNP")
    if stage == "pre":
        histograms = prefit_histograms(wkspace, samples, region, fitname=fitname)
        total_mc, uncertainty = prefit_total_and_uncertainty(wkspace, region)
    elif stage == "post":
        histograms = postfit_histograms(wkspace, samples, region)
        total_mc, uncertainty = postfit_total_and_uncertainty(wkspace, region)
    else:
        raise ValueError("stage must be 'pre' or 'post'")
    histograms["Data"] = data_histogram(wkspace, region)
    bin_edges = histograms["Data"].edges
    counts = {k: v.values for k, v in histograms.items()}
    errors = {k: np.sqrt(v.variances) for k, v in histograms.items()}

    logy = False
    for pat in tdub.config.PLOTTING_LOGY:
        if pat.search(region) is not None:
            logy = True

    fig, ax0, ax1 = canvas_from_counts(
        counts, errors, bin_edges, uncertainty=uncertainty, total_mc=total_mc, logy=logy,
    )

    # stack axes cosmetics
    ax0.set_ylabel("Events", horizontalalignment="right", y=1.0)
    draw_atlas_label(ax0, extra_lines=[meta_text(region, stage)])
    legend_last_to_first(ax0, ncol=1, loc="upper right")

    # ratio axes cosmetics
    ax1.set_xlabel(meta_axis_label(region), horizontalalignment="right", x=1.0)
    ax1.set_ylabel("Data/MC")
    if stage == "post":
        ax1.set_ylim([0.9, 1.1])
        ax1.set_yticks([0.9, 0.95, 1.0, 1.05])
    ax1.text(
        0.02, 0.8, chisq_text(wkspace, region, stage), transform=ax1.transAxes, size=10
    )
    ax1.legend(loc="lower left", fontsize=10)

    # return objects
    return fig, ax0, ax1


def plot_all_regions(
    wkspace: Union[str, os.PathLike],
    outdir: Union[str, os.PathLike],
    stage: str = "both",
    fitname: str = "tW",
) -> None:
    """Plot all regions discovered in a workspace.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace
    outdir : str or os.PathLike
        Path to save resulting files to
    stage : str
        Fitting stage (`"pre"`, `"post"`, or `"both"`)
    fitname : str
        Name of the Fit
    """
    PosixPath(outdir).mkdir(parents=True, exist_ok=True)
    regions = available_regions(wkspace)

    def plot_stage(stage):
        for region in regions:
            fig, ax0, ax1 = stack_canvas(wkspace, region, stage=stage)
            fig.savefig(f"{outdir}/{region}_{stage}Fit.pdf")
            plt.close(fig)
            del fig, ax0, ax1

    if stage == "both":
        plot_stage("pre")
        plot_stage("post")
    elif stage == "pre":
        plot_stage("pre")
    elif stage == "post":
        plot_stage("post")
    else:
        raise ValueError("stage can be 'both', 'pre', or 'post'")

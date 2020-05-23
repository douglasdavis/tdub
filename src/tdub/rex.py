"""Utilities for parsing TRExFitter."""

# stdlib
import logging
import os
from pathlib import PosixPath
from typing import Dict, Iterable, List, Tuple, Union

# external
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import uproot  # noqa
from uproot.rootio import ROOTDirectory  # noqa
from uproot_methods.base import ROOTMethods  # noqa
from uproot_methods.classes.TGraphAsymmErrors import Methods as ROOT_TGraphAsymmErrors  # noqa
from uproot_methods.classes.TH1 import Methods as ROOT_TH1  # noqa
import yaml  # noqa

# tdub
from .art import canvas_from_counts  # noqa
from .art import setup_tdub_style  # noqa

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


def prefit_histogram(rfile: ROOTDirectory, sample: str, region: str) -> ROOT_TH1:
    """Get a prefit histogram from a file.

    Parameters
    ----------
    rfile : root.rootio.ROOTDirectory
        File containing the desired prefit histogram.
    sample : str
        Physics sample name.
    region : str
        TRExFitter region name.

    Returns
    -------
    uproot_methods.base.ROOTMethods
        ROOT histogram (None if not found).
    """
    histname = f"{region}_{sample}"
    try:
        h = rfile.get(histname)
        return h
    except KeyError:
        log.fatal("%s histogram not found in %s" % (histname, rfile))
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
    dict(str, uproot_methods.base.ROOTMethods)
        Prefit ROOT histograms
    """

    root_path = PosixPath(wkspace) / "Histograms" / f"{fitname}_{region}_histos.root"
    rfile = uproot.open(root_path)
    histograms = {}
    for samp in samples:
        h = prefit_histogram(rfile, samp, region)
        if h is None:
            log.warn("Histogram for sample %s in region: %s not found" % (samp, region))
        histograms[samp] = h
    return histograms


def prefit_errorband(
    wkspace: Union[str, os.PathLike], region: str
) -> Tuple[ROOT_TGraphAsymmErrors, ROOT_TH1]:
    """Get the prefit uncertainty band for a region.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace.
    region : str
        Region to get error band for.

    Returns
    -------
    uproot_methods.base.ROOTMethods
        The error TGraph.
    uproot_methods.base.ROOTMethods
        The total MC expectation histogram.
    """
    root_path = PosixPath(wkspace) / "Histograms" / f"{region}_preFit.root"
    rfile = uproot.open(root_path)
    err = rfile.get("g_totErr")
    tot = rfile.get("h_tot")
    return err, tot


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


def postfit_histogram(rfile: ROOTDirectory, sample: str) -> ROOT_TH1:
    """Get a postfit histogram from a file.

    Parameters
    ----------
    rfile : root.rootio.ROOTDirectory
        File containing the desired postfit histogram.
    sample : str
        Physics sample name.

    Returns
    -------
    uproot_methods.base.ROOTMethods
        ROOT histogram (None if not found).
    """
    histname = f"h_{sample}_postFit"
    try:
        h = rfile.get(histname)
        return h
    except KeyError:
        log.fatal("%s histogram not found in %s" % (histname, rfile))
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
    dict(str, uproot_methods.base.ROOTMethods)
        Postfit ROOT histograms
    """
    root_path = PosixPath(wkspace) / "Histograms" / f"{region}_postFit.root"
    rfile = uproot.open(root_path)
    histograms = {}
    for samp in samples:
        if samp == "Data":
            continue
        h = postfit_histogram(rfile, samp)
        if h is None:
            log.warn("Histogram for sample %s in region %s not found" % (samp, region))
        histograms[samp] = h
    return histograms


def postfit_errorband(
    wkspace: Union[str, os.PathLike], region: str
) -> Tuple[ROOT_TGraphAsymmErrors, ROOT_TH1]:
    """Get the postfit uncertainty band for a region.

    Parameters
    ----------
    wkspace : str or os.PathLike
        Path of the TRExFitter workspace.
    region : str
        Region to get error band for.

    Returns
    -------
    uproot_methods.base.ROOTMethods
        The error TGraph.
    uproot_methods.base.ROOTMethods
        The total MC expectation histogram.
    """
    root_path = PosixPath(wkspace) / "Histograms" / f"{region}_postFit.root"
    rfile = uproot.open(root_path)
    err = rfile.get("g_totErr_postFit")
    tot = rfile.get("h_tot_postFit")
    return err, tot


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
    matplotlib.figure.Figure
        Figure for housing the plot.
    matplotlib.axes.Axes
        Main axes for the histogram stack.
    matplotlib.axes.Axes
        Ratio axes to show Data/MC.
    """
    samples = ("tW", "ttbar", "Zjets", "Diboson", "MCNP")
    if stage == "pre":
        histograms = prefit_histograms(wkspace, samples, region, fitname=fitname)
        errband, totmc = prefit_errorband(wkspace, region)
    elif stage == "post":
        histograms = postfit_histograms(wkspace, samples, region)
        errband, totmc = postfit_errorband(wkspace, region)
    else:
        raise ValueError("stage must be 'pre' or 'post'")
    histograms["Data"] = data_histogram(wkspace, region)
    bin_edges = histograms["Data"].edges
    count_df = {k: v.values for k, v in histograms.items()}
    error_df = {k: np.sqrt(v.variances) for k, v in histograms.items()}
    chi2, ndof, prob = chisq(wkspace, region, stage=stage)
    return canvas_from_counts(
        count_df, error_df, bin_edges, errorband=errband, totalmc=totmc
    )

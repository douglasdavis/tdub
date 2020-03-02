"""
Art utilities
"""

# stdlib
from typing import Any, Dict, Optional, Tuple, List
from pathlib import PosixPath
import logging

# external
import numpy as np
import matplotlib.pyplot as plt
import uproot
from uproot.rootio import ROOTDirectory

# tdub
from tdub import setup_logging
import tdub._art
import tdub.utils
from tdub.utils import PathLike

setup_style = tdub._art.setup_style

setup_logging()
log = logging.getLogger(__name__)


class TRExHistogram:
    """Defines a histogram built from a TRExFitter output file

    Attributes
    ----------
    region : str
        the region as defined in the TRExFitter configuration
    sample : str
        which physics sample
    var : str
        the variable that was histogrammed
    signature : str
        the actual name of the TH1 object in hte ROOT file
    postfit : bool
        whether or not the histogram is a postfit histogram

    Parameters
    ----------
    hfile : uproot.rootio.ROOTDirectory
        the uproot file
    region : str
        the region string as defined in TRExFitter
    sample : str
        which physics sample
    postfit : bool
        to flag this as a TRExFitter post-fit histogram
    """

    def __init__(
        self, hfile: ROOTDirectory, region: str, sample: str, postfit: bool = False
    ) -> None:
        self.region = region
        self.sample = sample
        self.postfit = postfit

        if self.region.startswith("VRP"):
            self.var = self.region.split("VRP_")[-1]
        else:
            self.var = "bdtres"

        pfp = "" if not self.postfit else "h_"
        pfs = "" if not self.postfit else "_postFit"
        regionsig = f"{self.region}_" if not self.postfit else ""
        self.signature = f"{pfp}{regionsig}{self.sample}{pfs}"
        try:
            self.uproothist = hfile.get(self.signature)
            self.content = self.uproothist.values
            self.content[self.content < 0] = 1.0e-6
        except KeyError:
            self.uproothist = None
            self.content = None

    def __bool__(self):
        return self.uproothist is not None

    def __call__(self):
        return self.uproothist

    @property
    def sumw2(self) -> np.ndarray:
        """numpy.ndarray: the sum of weights squared in each bin"""
        return self.uproothist.variances

    @property
    def error(self) -> np.ndarray:
        """numpy.ndarray: the uncertainty in each bin (sqrt of sumw2)"""
        return np.sqrt(self.sumw2)

    @property
    def bins(self) -> np.ndarray:
        """numpy.ndarray: the bin edges"""
        return self.uproothist.edges

    @property
    def bin_centers(self) -> np.ndarray:
        """numpy.ndarray: the bin centers"""
        return tdub.utils.bin_centers(self.bins)

    @property
    def bin_width(self) -> np.ndarray:
        """numpy.ndarray: the bin widths"""
        return round(self.bins[-1] - self.bins[-2], 2)

    def has_uniform_bins(self) -> bool:
        """determines if the histogram has uniform bin widths

        Returns
        -------
        bool
            whether or not bin widthds are uniform.
        """
        diffs = np.ediff1d(self.bins)
        return np.allclose(diffs, diffs[0])


class TRExRegionSources:
    """The sources for TRExFitter objects associated with a region

    Attributes
    ----------
    region : str
        the string representation region
    histo_root_file : uproot.rootio.ROOTDirectory
        the file housing all prefit histograms
    prefit_root_file : uproot.rootio.ROOTDirectory
        the file housing the prefit error band
    prefit_yaml_file : pathlib.PosixPath
        the file housing the prefix chi^2 information
    postfit_root_file : uproot.rootio.ROOTDirectory
        the file housing the postfit histograms and error band
    postfit_yaml_file : pathlib.PosixPath
        the file housing the prefix chi^2 information

    Parameters
    ----------
    fitdir : str or os.PathLike
        the TRExFitter fit directory to parse
    region : str
        the specific region to parse
    """

    def __init__(self, fitdir: PathLike, region: str) -> None:
        log.info("start parse")
        fitdir = PosixPath(fitdir).resolve()
        fitname = fitdir.stem
        histdir = fitdir / "Histograms"
        self.region = region
        self.histo_root_file = uproot.open(histdir / f"{fitname}_{region}_histos.root")
        self.prefit_root_file = uproot.open(histdir / f"{region}_preFit.root")
        self.prefit_yaml_file = histdir / f"{region}_preFitChi2.yml"
        self.postfit_root_file = histdir / f"{region}_postFit.root"
        self.postfit_yaml_file = histdir / f"{region}_postFitChi2.yml"
        if not self.postfit_root_file.exists():
            self.postfit_root_file = None
        else:
            self.postfit_root_file = uproot.open(self.postfit_root_file)
        if not self.postfit_yaml_file.exists():
            self.postfit_yml = None

        if self.region.startswith("VRP"):
            self.var = self.region.split("VRP_")[-1]
        else:
            self.var = "bdtres"

    def has_postfit(self) -> bool:
        return self.postfit_file is not None

    def prefit_hist(self, sample: str) -> TRExHistogram:
        return TRExHistogram(self.histo_root_file, self.region, sample)

    def postfit_hist(self, sample: str) -> TRExHistogram:
        if self.has_postfit():
            return TRExHistogram(self.postfit_root_file, self.region, sample, postfit=True)
        return None

    def prefit_ratio_band(self):
        return self.prefit_root_file.get("g_totErr")

    def postfit_ratio_band(self):
        return self.postfit_root_file.get("g_totErr_postFit")


def regions_from_fitdir(fitdir: PathLike) -> List[str]:
    """get a list of regions from a TRExFitter directory

    Parameters
    ----------
    fitdir : str or os.PathLike
        the TRExFitter directory

    Returns
    -------
    list(str)
        list of region strings
    """
    fitdir = PosixPath(fitdir).resolve()
    histdir = fitdir / "Histograms"
    fitname = fitdir.stem
    regions = []
    for f in histdir.glob("*_histos.root"):
        region = f.name.split(f"{fitname}_")[-1].split("_histos.root")[0]
        regions.append(region)
    return regions


def counts_from_trex_sources(sources: TRExRegionSources) -> Dict[str, np.ndarray]:
    counts = {}
    samples = [
        "Data",
        "tW",
        "ttbar",
        "Zjets",
        "Diboson",
        "MCNP",
    ]
    for sample in samples:
        histo = sources.hist_for_sample(sample)
        counts[sample] = histo.content
    return counts


def construct_from_sources(sources: TRExRegionSources) -> None:
    counts = counts_from_trex_sources(sources)
    error_band = None
    pass


def foo(fitdir):
    regions = regions_from_fitdir(fitdir)
    sources = [TRExRegionSources(fitdir, region) for region in regions]


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
    centers = tdub.utils.bin_centers(bin_edges)
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

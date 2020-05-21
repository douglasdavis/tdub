"""Module for art from TRExFitter."""

# stdlib
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import PosixPath

from typing import List, Any, Optional, Tuple, Set

# externals
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
import yaml

# tdub
import tdub.hist
from tdub.art import setup_tdub_style
from tdub._art import draw_atlas_label, var_to_axis_meta


setup_tdub_style()
log = logging.getLogger(__name__)


@dataclass
class Sample:
    """Defines a physics sample.

    Attributes
    ----------
    name : str
       name of the sample
    signature : str
       name of the sample on disk
    color : str
       matplotlib color
    tex : str
       LaTeX label
    """

    name: str = "none"
    signature: str = "none"
    color: str = "black"
    tex: str = "none"


@dataclass
class NuisPar:
    """Defines a nuisance parameter from TRExFitter.

    Attributes
    ----------
    name : str
       name on disk
    mean : float
       mean value
    minus : float
       minus part of the error bar
    plus : float
       plus part of the error bar
    category : str
       systematic category
    title : str
       a title for the plots
    """

    name: str = "none"
    mean: float = 0
    minus: float = 0
    plus: float = 0
    category: str = "none"
    title: str = "none"


@dataclass
class Template:
    """Defines a template for generation by TREx.

    Attributes
    ----------
    var : str
       variable (branch) from the tree
    region : List[str]
       list of regions where the template was calculated
    xmin : float
       axis range minimum
    xmax: float
       axis range maximum
    nbins : int
       number of bins
    use_region_binning : bool
       for wt-stat usage
    axis_title : str
       axis title for TRExFitter
    mpl_title : str
       axis title for matplotlib
    is_aux : bool
       for wt-stat usagse
    unit : str
       unit as a string (e.g. GeV)
    """

    var: str = ""
    regions: List[str] = field(default_factory=list)
    xmin: float = 0
    xmax: float = 0
    nbins: int = 0
    use_region_binning: bool = False
    axis_title: str = ""
    mpl_title: str = ""
    is_aux: bool = False
    unit: str = ""


@dataclass
class TRExHistogram:
    """Defines a histogram built from a TRExFitter output file.

    Attributes
    ----------
    hfile : str
       the name of the TRExFitter histogram file holding the data
    region : str
       the region as defined in the TRExFitter configuration
    sample : Sample
       which physics sample
    unit : str
       a unit for plots (e.g. "GeV")
    mpl_title : str
       an axis title for matplotlib usage
    signature : str
       the actual name of the TH1 object in hte ROOT file
    postfit : bool
       whether or not the histogram is a postfit histogram
    """

    hfile: str
    region: str
    sample: Sample
    unit: str = ""
    mpl_title: str = ""
    signature: str = ""
    postfit: bool = False

    def __post_init__(self):
        pfp = "" if not self.postfit else "h_"
        pfs = "" if not self.postfit else "_postFit"
        regionsig = f"{self.region}_" if not self.postfit else ""
        self.signature = f"{pfp}{regionsig}{self.sample.signature}{pfs}"
        try:
            self.uproothist = uproot.open(self.hfile).get(self.signature)
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
        return tdub.hist.bin_centers(self.bins)

    @property
    def bin_width(self) -> np.ndarray:
        """numpy.ndarray: the bin widths"""
        return round(self.bins[-1] - self.bins[-2], 2)

    def has_uniform_bins(self) -> bool:
        """determines if the histogram has uniform bin widths.

        Returns
        -------
        bool
           whether or not bin widthds are uniform.
        """
        diffs = np.ediff1d(self.bins)
        return np.allclose(diffs, diffs[0])


def _get_plot_samples() -> List[Sample]:
    plot_samples = [
        Sample(name="tW", signature="tW", color="#1f77b4", tex="$tW$"),
        Sample(name="ttbar", signature="ttbar", color="#d62728", tex="$t\\bar{t}$"),
        Sample(name="Diboson", signature="Diboson", color="#2ca02c", tex="Diboson"),
        Sample(name="Zjets", signature="Zjets", color="#ff7f0e", tex="$Z+$jets"),
        Sample(name="MCNP", signature="MCNP", color="#9467bd", tex="MCNP"),
    ]
    plot_samples.reverse()
    return plot_samples


_region_meta = var_to_axis_meta()


def draw_ratio_with_line(
    ax: plt.Axes,
    data: TRExHistogram,
    mc_sum: np.ndarray,
    mc_err: np.ndarray,
    yline: float = 1.0,
    autoxscale: bool = True,
) -> None:
    """draw the ratio with a horizontal line on the axis."""
    x1 = data.bins[0]
    x2 = data.bins[-1]
    err = np.sqrt(
        data.content / (mc_sum ** 2) + np.power(data.content * mc_err / (mc_sum ** 2), 2)
    )
    ax.plot([x1, x2], [yline, yline], color="gray", linestyle="solid", marker=None)
    ax.errorbar(data.bin_centers, data.content / mc_sum, yerr=err, fmt="ko", zorder=999)
    ax.set_ylabel("Data / MC")
    if autoxscale:
        ax.autoscale(enable=True, axis="x", tight=True)


def set_labels(ax: plt.Axes, histogram: TRExHistogram) -> None:
    """define the axis labels."""
    if histogram.has_uniform_bins():
        ylabel_suffix = f" / {histogram.bin_width} {histogram.unit}"
    else:
        ylabel_suffix = f" / bin"
    ax.set_ylabel(f"Events{ylabel_suffix}", horizontalalignment="right", y=1.0)


def shrink_pdf(file_path: str) -> None:
    """shrink pdf file using ghostscript."""
    command = (
        "gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 "
        "-dNOPAUSE -dQUIET -dBATCH "
        "-sOutputFile=temp.pdf {in_file}"
    )
    log.debug(f"shrinking {file_path} via gs")
    in_file = PosixPath(file_path).absolute()
    in_name = os.fspath(in_file)
    proc = subprocess.Popen(command.format(in_file=in_file), shell=True)
    proc.wait()
    os.remove(in_file)
    os.rename("temp.pdf", in_name)


def stackem(
    args: Any,
    region: str,
    data: Sample,
    histograms: List[Sample],
    template_var: str,
    band: Optional[Any] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[matplotlib.figure.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Create a stack plot.

    Parameters
    ----------
    args : argparse.Namespace
       command line arguments
    region : str
       region from TRExFitter
    data : Sample
       data sample
    histograms : list(Sample)
       list of MC samples to stack
    template_var : str
       name of the template variable
    band : Optional[uproot_methods.classes.TGraphAsymmErrors]
       error band
    figsize : tuple(float, float), optional
       matplotlib figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
       matplotlib fiture
    axes : tuple(matplotlib.axes.Axes, matplotlib.axes.Axes)
       matplotlib axes (main, ratio)
    """
    # fmt: off
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                  gridspec_kw=dict(height_ratios=[3.25, 1], hspace=0.025))
    expected_sum = np.sum([h.content for h in histograms], axis=0)
    expected_err = np.sqrt(np.sum([h.sumw2 for h in histograms], axis=0))
    ax.hist([h.bin_centers for h in histograms], weights=[h.content for h in histograms],
            bins=histograms[0].bins, histtype="stepfilled", stacked=True,
            color=[h.sample.color for h in histograms], label=[h.sample.tex for h in histograms])
    ax.errorbar(data.bin_centers, data.content, yerr=data.error, fmt="ko", label="Data", zorder=999)
    set_labels(ax, histograms[0])

    if _region_meta[template_var].logy:
        ax.set_yscale("log")
        ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] * 1000])
    else:
        ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] * 1.5])

    if band is not None:
        yerrlo = np.hstack([band.yerrorslow, band.yerrorslow[-1]])
        yerrhi = np.hstack([band.yerrorshigh, band.yerrorshigh[-1]])
        expected_sum4band = np.hstack([expected_sum, expected_sum[-1]])

        if args.band_style == "hatch":
            ax.fill_between(x=data.bins, y1=(expected_sum4band - yerrlo), y2=(expected_sum4band + yerrhi),
                            step="post", facecolor="none", hatch="////", edgecolor="cornflowerblue",
                            linewidth=0.0, zorder=50, label="Syst. Unc.")
            axr.fill_between(x=data.bins, y1=1 - yerrlo / expected_sum4band, y2=1 + yerrhi / expected_sum4band,
                             step="post", facecolor="none", hatch="////", edgecolor="cornflowerblue",
                             linewidth=0.0, zorder=50, label="Syst. Unc.")
        elif args.band_style == "shade":
            axr.fill_between(x=data.bins, y1=1 - yerrlo / expected_sum4band, y2=1 + yerrhi / expected_sum4band,
                             step="post", facecolor=(0, 0, 0, 0.33), linewidth=0.0, zorder=50, label="Syst. Unc.")

    # fmt: on
    draw_ratio_with_line(axr, data, expected_sum, expected_err)
    axr.set_ylim([0.8, 1.2])
    axr.set_yticks([0.9, 1.0, 1.1])
    axis_title = "{}{}".format(data.mpl_title, "" if data.unit == "" else f" [{data.unit}]")
    axr.set_xlabel(axis_title, horizontalalignment="right", x=1.0)

    if band is not None and args.band_style == "shade":
        axr.legend(loc="upper center", frameon=True, fontsize=8)

    ax.legend(loc="upper right")
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, handles.pop())
    labels.insert(0, labels.pop())
    ax.legend(handles, labels, loc="upper right", ncol=args.legend_ncol)

    raw_region = region.split("reg")[-1].split("_")[0]
    extra_line1 = f"$\\sqrt{{s}}$ = 13 TeV, $L = {args.lumi}$ fb$^{{-1}}$"
    extra_line2 = f"$pp\\rightarrow tW \\rightarrow e^{{\\pm}}\\mu^{{\\mp}} ({raw_region})$"
    extra_line3 = "Pre-fit"
    if histograms[0].postfit:
        extra_line3 = "Post-fit"
    draw_atlas_label(ax, extra_lines=[extra_line1, extra_line2, extra_line3])

    fig.subplots_adjust(left=0.115, bottom=0.115, right=0.965, top=0.95)
    return fig, (ax, axr)


def prefit_histograms(
    args: Any, fit_name: str, region: str, samples: List[Sample]
) -> Tuple[TRExHistogram, List[TRExHistogram], Any]:
    """Prepare prefit histogram objects.

    Parameters
    ---------
    args : argparse.Namespace
       command line arguments
    fit_name : str
       TRExFitter fit name
    region : str
       TRExFitter region
    samples : list(Sample)
       list of MC samples

    Returns
    -------
    data : TRExHistogram
       data histogram
    histograms : List[TRExHistogram]
       MC histograms
    band : uproot_methods.classes.TGraphAsymmErrors
       uncertainty band
    """
    hfile = f"{args.workspace}/Histograms/{fit_name}_{region}_histos.root"
    bfile = f"{args.workspace}/Histograms/{fit_name}_{region}_preFit.root"
    band = uproot.open(bfile).get("g_totErr")
    histograms = [TRExHistogram(hfile, region, sample) for sample in samples]
    data = TRExHistogram(hfile, region, Sample("Data", "Data", "", "Data"))
    return data, histograms, band


def postfit_histograms(
    args: Any, fit_name: str, region: str, samples: List[Sample]
) -> Tuple[TRExHistogram, List[TRExHistogram], Any]:
    """Prepare postfit histogram objects.

    Parameters
    ---------
    args : argparse.Namespace
       command line arguments
    fit_name : str
       TRExFitter fit name
    region : str
       TRExFitter region
    samples : List[Sample]
       list of MC samples

    Returns
    -------
    data : TRExHistogram
       data histogram
    histograms : List[TRExHistogram]
       MC histograms
    band : uproot_methods.classes.TGraphAsymmErrors
       uncertainty band
    """
    hfile = f"{args.workspace}/Histograms/{region}_postFit.root"
    bfile = f"{args.workspace}/Histograms/{region}_postFit.root"
    band = uproot.open(bfile).get("g_totErr_postFit")
    histograms = [TRExHistogram(hfile, region, sample, postfit=True) for sample in samples]
    return histograms, band


def split_region_str(region: str) -> Tuple[str, str]:
    splits = region.split("_")
    if len(splits) == 1:
        return (region, "bdt_response")
    else:
        return (splits[0], "_".join(splits[1:]))


def run_stacks(args: Any) -> None:
    """Given command line arguments generate stack plots.

    Parameters
    ----------
    args : argparse.Namespace
    """
    samples = _get_plot_samples()

    if args.out_dir is None:
        outd = f"{args.workspace}/MPL"
    else:
        outd = args.out_dir
    if outd != ".":
        PosixPath(outd).mkdir(parents=True, exist_ok=True)

    fit_name = PosixPath(args.workspace).stem
    hfiledir = PosixPath(f"{fit_name}/Histograms")
    regions = []
    if args.skip_regions is not None:
        skipregex = re.compile(args.skip_regions)
    else:
        skipregex = None
    for hfile in hfiledir.iterdir():
        if "_histos.root" in hfile.name:
            region = hfile.name.split("_histos.root")[0].split(f"{fit_name}_")[-1]
            if skipregex:
                if re.search(skipregex, region):
                    continue
            regions.append(region)

    for region in regions:
        raw_region, template_variable = split_region_str(region)
        data, histograms, band = prefit_histograms(args, fit_name, region, samples)
        data.unit = _region_meta[template_variable].unit
        data.mpl_title = _region_meta[template_variable].title
        fig, (ax, axr) = stackem(
            args, region, data, histograms, template_variable, band=band
        )
        out_name = f"{outd}/preFit_{region}.pdf"
        fig.savefig(out_name)
        plt.close(fig)
        log.info(f"Done with {region} prefit")

        if args.do_postfit:
            histograms, band = postfit_histograms(args, fit_name, region, samples)
            fig, (ax, axr) = stackem(
                args, region, data, histograms, template_variable, band=band
            )
            axr.set_ylim([0.925, 1.075])
            axr.set_yticks([0.95, 1.0, 1.05])
            out_name = f"{outd}/postFit_{region}.pdf"
            fig.savefig(out_name)
            plt.close(fig)
            log.info(f"Done with {region} postfit")


def get_blank_systematics(config_file: str) -> Tuple[List[NuisPar], Set[str]]:
    """Get list of NPs and categories from TRExFitter config file.

    Parameters
    ----------
    config_file : str
        name of config file

    Returns
    -------
    nps : list(NuisPar)
        list of nuisance parameters
    categories: set(str)
        all categories that were found
    """
    trex_config = PosixPath(config_file)
    nps = []
    with trex_config.open("r") as f:
        config_blocks = f.read().split("\n\n")
        for block in config_blocks:
            if "Systematic:" in block:
                if block[0] == "%":
                    continue
                block = block.replace("\n  ", "\n")
                s = yaml.load(block, Loader=yaml.FullLoader)
                np = NuisPar(name=s["Systematic"], category=s["Category"], title=s["Title"])
                nps.append(np)
    categories = set()
    nps = {np.name: np for np in nps}
    for npn, np in nps.items():
        categories.add(np.category)
    return nps, categories


def draw_pulls(args: Any, nps: List[NuisPar]) -> Tuple[plt.Figure, plt.Axes]:
    """Draw pulls from command line arguments and nuisance parameters.

    Parameters
    ----------
    args : argparse.Namespace
       command line arguments
    nps : list
       list of nuisance parameters to draw

    Returns
    -------
    fig : matplotlib.figure.Figure
       matplotlib fiture
    ax : matplotlib.axes.Axes
       matplotlib axis
    """
    Y_OFFSET_TEXT = 0.095
    Y_OFFSET_TEXT_MEAN = 0.165
    X_OFFSET_TEXT = 0.035

    nnps = len(nps)
    xval = np.array([np.mean for np in nps])
    yval = np.array([(i + 1) for i in range(len(xval))])
    xerr_lo = np.array([np.minus for np in nps])
    xerr_hi = np.array([np.plus for np in nps])
    ylabels = [np.title.replace("ttbar", "$t\\bar{t}$").replace("tW", "$tW$") for np in nps]

    fig, ax = plt.subplots(figsize=(7, 1.0 + len(yval) * 0.38))
    fig.subplots_adjust(left=0.5, right=0.95)
    ax.fill_betweenx([-50, 500], -2, 2, color="yellow")
    ax.fill_betweenx([-50, 500], -1, 1, color="limegreen")
    ax.set_yticks(yval)
    ax.set_yticklabels(ylabels)
    ax.errorbar(
        xval,
        yval,
        xerr=[abs(xerr_lo), xerr_hi],
        fmt="ko",
        capsize=3.5,
        lw=2,
        elinewidth=2.25,
    )
    ax.set_xlim([-2.2, 2.2])
    ax.set_ylim([0.0, len(yval) + 1])
    ax.grid(color="black", alpha=0.15)

    # fmt: off
    if not args.no_text:
        for mean, iyval, minus, plus in zip(xval, yval, xerr_lo, xerr_hi):
            ax.text(mean, iyval + Y_OFFSET_TEXT_MEAN, "${}$".format(round(mean, 3)),
                    color="black", size=10, horizontalalignment="center")
            ax.text(mean + minus - 6.5 * X_OFFSET_TEXT, iyval + Y_OFFSET_TEXT, "${}$".format(round(minus, 3)),
                    color="red", size=10, horizontalalignment="center")
            ax.text(mean + plus + X_OFFSET_TEXT, iyval + Y_OFFSET_TEXT, "${}$".format(round(plus, 3)),
                    color="blue", size=10)
    # fmt: on

    fig.subplots_adjust(left=0.45)
    if nnps < 10:
        fig.subplots_adjust(bottom=(0.225 + 0.25 / nnps))
    ax.set_xlabel(r"$\left(\hat\theta - \theta_0\right) / \Delta \theta$")
    return fig, ax


def run_pulls(args: Any) -> None:
    """Given command line arguments generate pull plots.

    Parameters
    ----------
    args : argparse.Namespace
    """
    systematics, categories = get_blank_systematics(args.config)
    fit_name = PosixPath(args.workspace).stem
    fit_result = PosixPath(f"{args.workspace}/Fits/{fit_name}.txt")
    np_by_cat = {c: [] for c in categories}
    with fit_result.open("r") as f:
        lines = f.read().split("CORRELATION_MATRIX")[0].strip()
        for line in lines.split("\n")[2:-1]:
            if line.startswith("gamma"):
                continue
            elements = line.split()
            systematics[elements[0]].mean = float(elements[1])
            systematics[elements[0]].plus = float(elements[2])
            systematics[elements[0]].minus = float(elements[3])
            np_by_cat[systematics[elements[0]].category].append(systematics[elements[0]])

    if args.out_dir is None:
        outd = f"{args.workspace}/MPL"
    else:
        outd = args.out_dir
    if outd != ".":
        PosixPath(outd).mkdir(parents=True, exist_ok=True)

    for category, nps in np_by_cat.items():
        fig, ax = draw_pulls(args, nps)
        out_name = f"{outd}/pulls_{category}.pdf"
        fig.savefig(out_name)
        log.info(f"Done with {category}")

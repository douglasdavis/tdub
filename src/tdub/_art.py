"""Backend art utilities."""

# std
from dataclasses import dataclass
from typing import Optional, List

# external
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def setup_style():
    matplotlib.rcParams["font.sans-serif"] = [
        "Helvetica",
        "helvetica",
        "Nimbus Sans L",
        "FreeSans",
    ]
    matplotlib.rcParams["axes.formatter.limits"] = [-4, 4]
    matplotlib.rcParams["axes.formatter.use_mathtext"] = True
    matplotlib.rcParams["axes.labelsize"] = 15
    matplotlib.rcParams["figure.figsize"] = (6, 5.5)
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["figure.subplot.left"] = 0.125
    matplotlib.rcParams["figure.subplot.bottom"] = 0.095
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


def draw_atlas_label(
    ax: plt.Axes,
    internal: bool = True,
    extra_lines: Optional[List[str]] = None,
    x: float = 0.050,
    y: float = 0.905,
    s1: int = 14,
    s2: int = 12,
) -> None:
    """draw the ATLAS label on the plot, with extra lines if desired."""
    ax.text(
        x,
        y,
        "ATLAS",
        fontstyle="italic",
        fontweight="bold",
        transform=ax.transAxes,
        size=s1,
    )
    if internal:
        ax.text(x + 0.15, y, r"Internal", transform=ax.transAxes, size=s1)
    if extra_lines is not None:
        for i, exline in enumerate(extra_lines):
            ax.text(x, y - (i + 1) * 0.06, exline, transform=ax.transAxes, size=s2)


@dataclass
class AxisMeta:
    title: str = ""
    unit: str = ""
    logy: bool = False


# fmt: off
def var_to_axis_meta():
    return {
        "bdt_response": AxisMeta("Classifier Response", ""),
        "bdt_response_DR": AxisMeta("Classifier Response (DR)", ""),
        "bdt_response_DS": AxisMeta("Classifier Response (DS)", ""),
        "bdtres_DR": AxisMeta("Classifier Response (DR)", ""),
        "bdtres_DS": AxisMeta("Classifier Response (DS)", ""),
        "bdt_DR_nonsoft": AxisMeta("Classifier Response (DR)", ""),
        "bdt_DR_wnsoft": AxisMeta("Classifier Response (DR)", ""),
        "bdt_DS_nonsoft": AxisMeta("Classifier Response (DS)", ""),
        "bdt_DS_wnsoft": AxisMeta("Classifier Response (DS)", ""),
        "cent_lep1lep2": AxisMeta("Centrality($\\ell_1\\ell_2$)", ""),
        "mT_lep2met": AxisMeta("$m_{\\mathrm{T}}(\\ell_2E_\\mathrm{T}^{\\mathrm{miss}})$", "GeV"),
        "mT_jet1met": AxisMeta("$m_{\\mathrm{T}}(j_1 E_\\mathrm{T}^{\\mathrm{miss}})$", "GeV"),
        "nsoftjets": AxisMeta("$N_j^{\\mathrm{soft}}$", "", True),
        "nsoftbjets": AxisMeta("$N_b^{\\mathrm{soft}}$", "", True),
        "deltapT_lep1_jet1": AxisMeta("$\\Delta p_{\\mathrm{T}}(\\ell_1, j_1)$", "GeV"),
        "deltapT_lep1_lep2": AxisMeta("$\\Delta p_{\\mathrm{T}}(\\ell_1, \\ell_2)$", "GeV"),
        "mass_lep2jet1": AxisMeta("$m_{\\ell_2 j_1}$", "GeV"),
        "mass_lep2jet1met": AxisMeta("$m_{\\ell_2 j_1 E_{\\mathrm{T}}^{\\mathrm{miss}}}$", "GeV"),
        "mass_lep1jet2": AxisMeta("$m_{\\ell_1 j_2}$", "GeV"),
        "mass_lep1jet1": AxisMeta("$m_{\\ell_1 j_1}$", "GeV"),
        "mass_lep2jet2": AxisMeta("$m_{\\ell_2 j_2}$", "GeV"),
        "psuedoContTagBin_jet2": AxisMeta("$b$-tag bin ($j_2$)", ""),
        "psuedoContTagBin_jet1": AxisMeta("$b$-tag bin ($j_1$)", ""),
        "pTsys_jet1met": AxisMeta("$p_{\\mathrm{T}}^{\\mathrm{sys}}(j_1 E_{\\mathrm{T}}^{\\mathrm{miss}})$", "GeV"),
        "pTsys_lep1lep2jet1": AxisMeta("$p_{\\mathrm{T}}^{\\mathrm{sys}}(\\ell_1\\ell_2 j_1)$", "GeV"),
        "pTsys_lep1lep2jet1jet2met": AxisMeta("$p_{\\mathrm{T}}^{\\mathrm{sys}}(\\ell_1\\ell_2 j_1 j_2 E_{\\mathrm{T}}^{\\mathrm{miss}})$", "GeV"),
        "pTsys_lep1lep2jet1met": AxisMeta("$p_{\\mathrm{T}}^{\\mathrm{sys}}(\\ell_1\\ell_2 j_1 E_{\\mathrm{T}}^{\\mathrm{miss}})$", "GeV"),
        "pTsys_lep1lep2": AxisMeta("$p_{\\mathrm{T}}^{\\mathrm{sys}}(\\ell_1\\ell_2)$", "GeV"),
        "pTsys_jet1jet2": AxisMeta("$p_{\\mathrm{T}}^{\\mathrm{sys}}(j_1 j_2)$", "GeV"),
        "pTsys_lep1lep2met": AxisMeta("$p_{\\mathrm{T}}^{\\mathrm{sys}}(\\ell_1\\ell_2 E_{\\mathrm{T}}^{\\mathrm{miss}})$", "GeV"),
        "deltaR_lep2_jet1": AxisMeta("$\\Delta R(\\ell_2, j_1)$", ""),
        "deltaR_lep1_jet1": AxisMeta("$\\Delta R(\\ell_1, j_1)$", ""),
        "deltaR_lep1_lep2": AxisMeta("$\\Delta R(\\ell_1, \\ell_2)$", ""),
        "deltaR_jet1_jet2": AxisMeta("$\\Delta R(j_1, j_2)$", ""),
        "deltaR_lep1lep2_jet1jet2met": AxisMeta("$\\Delta R(\\ell_1\\ell_2, j_1j_2 E_{\\mathrm{T}}^{\\mathrm{miss}})$", ""),
        "pT_lep1": AxisMeta("$p_{\\mathrm{T}}(\\ell_1)$", "GeV"),
        "pT_lep2": AxisMeta("$p_{\\mathrm{T}}(\\ell_2)$", "GeV"),
        "pT_jet1": AxisMeta("$p_{\\mathrm{T}}(j_1)$", "GeV"),
        "pT_jet2": AxisMeta("$p_{\\mathrm{T}}(j_2)$", "GeV"),
        "pT_jetS1": AxisMeta("$p_{\\mathrm{T}}(j_{S1})$", "GeV"),
        "eta_lep1": AxisMeta("$\\eta(\\ell_1)$", ""),
        "eta_lep2": AxisMeta("$\\eta(\\ell_2)$", ""),
        "eta_jet1": AxisMeta("$\\eta(j_1)$", ""),
        "eta_jet2": AxisMeta("$\\eta(j_2)$", ""),
        "met": AxisMeta("$E_{\\mathrm{T}}^{\\mathrm{miss}}$", "GeV"),
        "HT_jet1jet2": AxisMeta("H_{\\mathrm{T}}(j_1 j_2)", "GeV")
    }
# fmt: on

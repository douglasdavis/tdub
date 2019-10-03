from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import PosixPath
import re
import uproot

import tdub.regions


__all__ = [
    "SampleInfo",
    "categorize_branches",
    "quick_files",
    "bin_centers",
    "get_branches",
    "conservative_branches",
]


_detail_sample_info_re = re.compile(
    r"""(?P<phy_process>\w+)_
        (?P<dsid>[0-9]{6})_
        (?P<sim_type>(FS|AFII))_
        (?P<campaign>MC16(a|d|e))_
        (?P<tree>\w+)
        (\.\w+|$)""",
    re.X,
)


@dataclass
class SampleInfo:
    """Describes a sample's attritubes given it's name

    Parameters
    ----------
    input_file : str
       the file stem containing the necessary groups to parse

    Attributes
    ----------
    phy_process : str
       physics process (e.g. ttbar or tW_DR or Zjets)
    dsid : int
       the dataset ID
    sim_type : str
       the simulation type, "FS" or "AFII"
    campaign : str
       the campaign, MC16{a,d,e}
    tree : str
       the original tree (e.g. "nominal" or "EG_SCALE_ALL__1up")

    Examples
    --------
    >>> from tdub.utils import SampleInfo
    >>> sampinfo = SampleInfo("ttbar_410472_AFII_MC16d_nominal.root")
    >>> sampinfo.phy_process
    ttbar
    >>> sampinfo.dsid
    410472
    >>> sampinfo.sim_type
    AFII
    >>> sampinfo.campaign
    MC16d
    >>> sampinfo.tree
    nominal

    """

    phy_process: str
    dsid: int
    sim_type: str
    campaign: str
    tree: str

    def __init__(self, input_file: str) -> SampleInfo:
        if "Data_Data" in input_file:
            self.phy_process = "Data"
            self.dsid = 0
            self.sim_type = "Data"
            self.campaign = "Data"
            self.tree = "nominal"
        else:
            m = _detail_sample_info_re.match(input_file)
            if not m:
                raise ValueError(f"{input_file} cannot be parsed by SampleInfo regex")
            self.phy_process = m.group("phy_process")
            if self.phy_process.startswith("MCNP"):
                self.phy_process = "MCNP"
            self.dsid = int(m.group("dsid"))
            self.sim_type = m.group("sim_type")
            self.campaign = m.group("campaign")
            self.tree = m.group("tree")


def categorize_branches(branches: Iterable[str]) -> Dict[str, List[str]]:
    """categorize branches into a separate lists

    The categories:

    - ``meta`` for meta information (final state information)
    - ``kin`` for kinematic features (used for classifiers)
    - ``weights`` for any branch that starts or ends with ``weight``

    Parameters
    ----------
    branches : Iterable(str)
       whole set of branches (columns from dataframes)

    Returns
    -------
    dict(str, list(str))
       dictionary of ``{category : list-of-branches}``

    Examples
    --------
    >>> from tdub.utils import categorize_branches
    >>> branches = ["pT_lep1", "pT_lep2", "weight_nominal", "weight_sys_jvt", "reg2j2b"]
    >>> cated = categorize_branches(branches)
    >>> cated["weights"]
    ['weight_sys_jvt', 'weight_nominal']
    >>> cated["meta"]
    ['reg2j2b']
    >>> cated["kin"]
    ['pT_lep1', 'pT_lep2']

    """
    metas = {
        "reg1j1b",
        "reg2j1b",
        "reg2j2b",
        "reg1j0b",
        "reg2j0b",
        "OS",
        "SS",
        "elmu",
        "elel",
        "mumu",
        "runNumber",
        "randomRunNumber",
        "eventNumber",
    }
    bset = set(branches)
    weight_re = re.compile(r"(^weight_\w+)|(\w+_weight$)")
    weights = set(filter(weight_re.match, bset))
    metas = metas & set(bset)
    kins = (set(bset) ^ weights) ^ metas
    return {
        "weights": sorted(weights, key=str.lower),
        "kin": sorted(kins, key=str.lower),
        "meta": sorted(metas, key=str.lower),
    }


def quick_files(datapath: str) -> Dict[str, List[str]]:
    """get a dictionary of ``{sample_str : file_list}`` for quick file access.

    The lists of files are sorted alphabetically. These types of
    samples are currently tested:

    - ``ttbar`` (nominal 410472)
    - ``tW_DR`` (nominal 410648, 410649)
    - ``tW_DS`` (nominal 410656, 410657)
    - ``Diboson``
    - ``Zjets``
    - ``MCNP``
    - ``Data``

    Parameters
    ----------
    datapath : str
        path where all of the ROOT files live

    Returns
    -------
    dict(str, list(str))
        dictionary for quick file access

    Examples
    --------
    >>> from pprint import pprint
    >>> from tdub.utils import quick_files
    >>> qf = quick_files("/path/to/some_files") ## has 410472 ttbar samples
    >>> pprint(qf["ttbar"])
    ['/path/to/some/files/ttbar_410472_FS_MC16a_nominal.root',
     '/path/to/some/files/ttbar_410472_FS_MC16d_nominal.root',
     '/path/to/some/files/ttbar_410472_FS_MC16e_nominal.root']

    """
    path = str(PosixPath(datapath).resolve())
    ttbar_files = sorted(glob(f"{path}/ttbar_410472_FS*nominal.root"))
    tW_DR_files = sorted(glob(f"{path}/tW_DR_41064*FS*nominal.root"))
    tW_DS_files = sorted(glob(f"{path}/tW_DS_41065*FS*nominal.root"))
    Diboson_files = sorted(glob(f"{path}/Diboson_*FS*nominal.root"))
    Zjets_files = sorted(glob(f"{path}/Zjets_*FS*nominal.root"))
    MCNP_files = sorted(glob(f"{path}/MCNP_*FS*nominal.root"))
    Data_files = sorted(glob(f"{path}/Data_data*.root"))
    return {
        "ttbar": ttbar_files,
        "tW_DR": tW_DR_files,
        "tW_DS": tW_DS_files,
        "Diboson": Diboson_files,
        "Zjets": Zjets_files,
        "MCNP": MCNP_files,
        "Data": Data_files,
    }


def bin_centers(bin_edges: numpy.ndarray) -> numpy.ndarray:
    """get bin centers given bin edges

    Parameters
    ----------
    bin_edges : numpy.ndarray
       edges defining binning

    Returns
    -------
    numpy.ndarray
       the centers associated with the edges

    Examples
    --------

    >>> import numpy as np
    >>> from tdub.utils import bin_centers
    >>> bin_edges = np.linspace(25, 225, 11)
    >>> centers = bin_centers(bin_edges)
    >>> bin_edges
    array([ 25.,  45.,  65.,  85., 105., 125., 145., 165., 185., 205., 225.])
    >>> centers
    array([ 35.,  55.,  75.,  95., 115., 135., 155., 175., 195., 215.])

    """
    return (bin_edges[1:] + bin_edges[:-1]) * 0.5


def get_branches(
    file_name: str,
    tree: str = "WtLoop_nominal",
    ignore_weights: bool = False,
    sort: bool = False,
) -> List[str]:
    """get list of branches in a ROOT TTree

    Parameters
    ----------
    file_name : str
       the ROOT file name
    tree : str
       the ROOT tree name
    ignore_weights : bool
       ignore all branches which start with ``weight_``.
    sort : bool
       sort the resulting branch list before returning

    Returns
    -------
    list(str)
       list of branches

    Examples
    --------

    A file with two kinematic variables and two weights

    >>> from tdub.utils import get_branches
    >>> get_branches("/path/to/file.root", ignore_weights=True)
    ["pT_lep1", "pT_lep2"]
    >>> get_branches("/path/to/file.root")
    ["pT_lep1", "pT_lep2", "weight_nominal", "weight_tptrw"]

    """
    t = uproot.open(file_name).get(tree)
    bs = [b.decode("utf-8") for b in t.allkeys()]
    if not ignore_weights:
        if sort:
            return sorted(bs)
        return bs

    weight_re = re.compile(r"(^weight_\w+)")
    weights = set(filter(weight_re.match, bs))
    if sort:
        return sorted(set(bs) ^ weights, key=str.lower)
    return list(set(bs) ^ weights)


def conservative_branches(
    file_name: str, tree: str = "WtLoop_nominal", sort: bool = False
) -> List[str]:
    """get branches in a ROOT file that form a conservative minimum

    we define "conservative minimum" as the branches necessary for
    using our BDT infrastructure, so this conservative minimum
    includes all of the features used by the BDTs as well as the
    variables necessary for region selection.

    Parameters
    ----------
    file_name : str
       the ROOT file name
    tree : str
       the ROOT tree name
    sort : bool
       sort the resulting branch list before returning

    Returns
    -------
    list(str)
       list of branches

    Examples
    --------

    Grab branches for a file that are relevant for applying BDT models
    and do something useful

    >>> from tdub.utils import conservative_branches
    >>> from tdub.frames import raw_dataframe
    >>> from tdub.apply import FoldedResult, to_dataframe
    >>> cb = conservative_branches("/path/to/file.root")
    >>> df = raw_dataframe("/path/to/file.root", branches=cb)
    >>> fr_2j2b = FoldedResult("/path/to/trained/fold2j2b", "2j2b")
    >>> fr_2j1b = FoldedResult("/path/to/trained/fold2j1b", "2j1b")
    >>> fr_2j2b.to_dataframe(df, query=True)
    >>> fr_2j1b.to_dataframe(df, query=True)

    """
    t = uproot.open(file_name).get(tree)
    bs = set([b.decode("utf-8") for b in t.allkeys()])

    good_branches = set(
        {"reg1j1b", "reg2j1b", "reg2j2b", "OS"}
        | set(tdub.regions.FEATURESET_1j1b)
        | set(tdub.regions.FEATURESET_2j1b)
        | set(tdub.regions.FEATURESET_2j2b)
    )
    good_branches = bs & good_branches

    if sort:
        return sorted(good_branches)
    return list(good_branches)

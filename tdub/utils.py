"""
Module for general utilities
"""

from __future__ import annotations

# stdlib
import numbers
import os
import re
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import PosixPath

# external
import numpy as np
import uproot


__all__ = [
    "Region",
    "SampleInfo",
    "categorize_branches",
    "quick_files",
    "bin_centers",
    "edges_and_centers",
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


class Region(Enum):
    """A simple enum class for easily using region information

    Attributes
    ----------
    r1j1b
       Our ``1j1b`` region
    r1j1b
       Our ``2j1b`` region
    r2j1b = 1
       Our ``2j2b`` region

    Examples
    --------

    Using this enum for grabing the ``2j2b`` region from a set of
    files:

    >>> from tdub.utils import Region
    >>> from tdub.frames import specific_dataframe
    >>> sdf = specific_dataframe(files, Region.r2j2b)

    """

    r1j1b = 0
    r2j1b = 1
    r2j2b = 2

    @staticmethod
    def from_str(s: str) -> Region:
        """get enum value for the given string

        This function supports three ways to define a region; prefixed
        with "r", prefixed with "reg", or no prefix at all. For
        example, ``Region.r2j2b`` can be retrieved like so:

        - ``Region.from_str("r2j2b")``
        - ``Region.from_str("reg2j2b")``
        - ``Region.from_str("2j2b")``

        Parameters
        ----------
        s : str
           string representation of the desired region

        Returns
        -------
        Region
           the enum version

        Examples
        --------

        >>> from tdub.utils import Region
        >>> Region.from_str("1j1b")
        <Region.r1j1b: 0>

        """
        if s.startswith("reg"):
            rsuff = s.split("reg")[-1]
            return Region.from_str(rsuff)
        elif s.startswith("r"):
            return Region[s]
        else:
            if s == "2j2b":
                return Region.r2j2b
            elif s == "2j1b":
                return Region.r2j1b
            elif s == "1j1b":
                return Region.r1j1b
            else:
                raise ValueError(f"{s} doesn't correspond to a Region")

    def __str__(self) -> str:
        return self.name[1:]


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


def categorize_branches(
    source: Union[Union[str, os.PathLike], Iterable[str]],
    tree: Optional[str] = "WtLoop_nominal",
) -> Dict[str, List[str]]:
    """categorize branches into a separate lists

    The categories:

    - ``kinematics`` for kinematic features (used for classifiers)
    - ``weights`` for any branch that starts or ends with ``weight``
    - ``meta`` for meta information (final state information)

    Parameters
    ----------
    source : os.PathLike or str or Iterable(str)
       if iterable of strings, use that as list of branches, if
       os.PathLike or str then get branches from ROOT file the
       ``tree`` argument.
    tree : str, optional
       the tree name in the file if ``source`` is os.PathLike; this is
       ignored if ``source`` is an iterable of strings.

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
    >>> cated["kinematics"]
    ['pT_lep1', 'pT_lep2']

    Using the file name

    >>> cbed = categorize_branches("/path/to/file.root")
    >>> root_file = PosixPath("/path/to/file.root")
    >>> cbed = categorized_branches(root_file)

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
        "charge_lep1",
        "charge_lep2",
        "pdgId_lep1",
        "pdgId_lep2",
        "runNumber",
        "randomRunNumber",
        "eventNumber",
    }

    if isinstance(source, str) or isinstance(source, os.PathLike):
        bset = set(get_branches(source, tree=tree))
    else:
        bset = set(source)
    weight_re = re.compile(r"(^weight_\w+)|(\w+_weight$)")
    weights = set(filter(weight_re.match, bset))
    metas = metas & set(bset)
    kinematics = (set(bset) ^ weights) ^ metas
    return {
        "kinematics": sorted(kinematics, key=str.lower),
        "weights": sorted(weights, key=str.lower),
        "meta": sorted(metas, key=str.lower),
    }


def quick_files(datapath: Union[str, os.PathLike]) -> Dict[str, List[str]]:
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
    datapath : str or os.PathLike
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
    Data_files = sorted(glob(f"{path}/*Data_Data_nominal.root"))
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


def edges_and_centers(
    bins: Union[int, Iterable], range: Optional[Tuple[float, float]] = None
) -> numpy.array:
    """create arrays for edges and bin centers

    Parameters
    ----------
    bins : int or sequence of scalers
       the number of bins or sequence representing bin edges
    range : tuple(float, float), optional
       the minimum and maximum defining the bin range (used if bins is integral)

    Returns
    -------
    :py:obj:`numpy.ndarray`
       the bin edges
    :py:obj:`numpy.ndarray`
       the bin centers

    Examples
    --------

    from bin multiplicity and a range

    >>> from tdub.utils import edges_and_centers
    >>> edges, centers = edges_and_centers(bins=20, range=(25, 225))

    from pre-existing edges

    >>> edges, centers = edges_and_centers(np.linspace(0, 10, 21))

    """
    if isinstance(bins, numbers.Integral):
        if range is None:
            raise ValueError("for integral bins we require non-None range")
        edges = np.linspace(range[0], range[1], bins + 1)
    else:
        edges = np.asarray(bins)
        if not np.all(edges[1:] >= edges[:-1]):
            raise ValueError("bins edges must monotonically increase")
    centers = bin_centers(edges)
    return edges, centers


def get_branches(
    file_name: Union[str, os.PathLike],
    tree: str = "WtLoop_nominal",
    ignore_weights: bool = False,
    sort: bool = False,
) -> List[str]:
    """get list of branches in a ROOT TTree

    Parameters
    ----------
    file_name : str or os.PathLike
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
    file_name: Union[str, os.PathLike], tree: str = "WtLoop_nominal"
) -> List[str]:
    """get branches in a ROOT file that form a conservative minimum

    we define "conservative minimum" as the branches necessary for
    using our BDT infrastructure, so this conservative minimum
    includes all of the features used by the BDTs as well as the
    variables necessary for region selection.

    Parameters
    ----------
    file_name : str or os.PathLike
       the ROOT file name
    tree : str
       the ROOT tree name

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
        | set(FEATURESET_1j1b)
        | set(FEATURESET_2j1b)
        | set(FEATURESET_2j2b)
    )
    good_branches = bs & good_branches

    return sorted(good_branches)


def get_selection(region: str) -> str:
    """get the selection given a region

    see :py:func:`tdub.utils.Region.from_str` for the compatible
    strings.

    Parameters
    ----------
    region : str
       the region as a string

    Returns
    -------
    str
       the selection string

    Examples
    --------

    >>> from tdub.utils import get_selection
    >>> get_selection("r2j1b")
    '(reg2j1b == True) & (OS == True)'
    >>> get_selection("reg1j1b")
    '(reg1j1b == True) & (OS == True)'
    >>> get_selection("2j2b")
    '(reg2j2b == True) & (OS == True)'

    """
    return SELECTIONS[Region.from_str(region)]


def get_features(region: str) -> List[str]:
    """get the feature list for a region

    see :py:func:`tdub.utils.Region.from_str` for the compatible
    strings.

    Parameters
    ----------
    region : str
       the region as a string

    Returns
    -------
    list(str)
       the list of features for that region

    Examples
    --------

    >>> from pprint import pprint
    >>> from tdub.utils import get_features
    >>> pprint(get_features("reg2j1b"))
    ['mass_lep1jet1',
     'mass_lep1jet2',
     'mass_lep2jet1',
     'mass_lep2jet2',
     'pT_jet2',
     'pTsys_lep1lep2jet1jet2met',
     'psuedoContTagBin_jet1',
     'psuedoContTagBin_jet2']

    """
    return FEATURESETS[Region.from_str(region)]


SELECTION_1j1b = "(reg1j1b == True) & (OS == True)"
"""
str: The pandas flavor selection string for the 1j1b region
"""

SELECTION_2j1b = "(reg2j1b == True) & (OS == True)"
"""
str: The pandas flavor selection string for the 2j1b region
"""

SELECTION_2j2b = "(reg2j2b == True) & (OS == True)"
"""
str: The pandas flavor selection string for the 2j2b region
"""


SELECTIONS = {
    Region.r1j1b: SELECTION_1j1b,
    Region.r2j1b: SELECTION_2j1b,
    Region.r2j2b: SELECTION_2j2b,
}
"""
dict(Region, str): key-value pairs for regions to their selection string
"""


FEATURESET_1j1b = sorted(
    [
        "pTsys_lep1lep2jet1met",
        "mass_lep2jet1",
        "mass_lep1jet1",
        "pTsys_lep1lep2",
        "deltaR_lep2_jet1",
        "nloosejets",
        "deltaR_lep1_lep2",
        "deltapT_lep1_jet1",
        "mT_lep2met",
        "nloosebjets",
        "cent_lep1lep2",
        "pTsys_lep1lep2jet1",
    ],
    key=str.lower,
)
"""
list(str): list of features we use for classifiers in the 1j1b region
"""

FEATURESET_2j1b = sorted(
    [
        "mass_lep1jet2",
        "psuedoContTagBin_jet1",
        "mass_lep1jet1",
        "mass_lep2jet1",
        "mass_lep2jet2",
        "pTsys_lep1lep2jet1jet2met",
        "psuedoContTagBin_jet2",
        "pT_jet2",
    ],
    key=str.lower,
)
"""
list(str): list of features we use for classifiers in the 2j1b region
"""


FEATURESET_2j2b = sorted(
    [
        "mass_lep1jet2",
        "mass_lep1jet1",
        "deltaR_lep1_jet1",
        "mass_lep2jet1",
        "pTsys_lep1lep2met",
        "pT_jet2",
        "mass_lep2jet2",
    ],
    key=str.lower,
)
"""
list(str): list of features we use for classifiers in the 2j2b region
"""


FEATURESETS = {
    Region.r1j1b: FEATURESET_1j1b,
    Region.r2j1b: FEATURESET_2j1b,
    Region.r2j2b: FEATURESET_2j2b,
}
"""
dict(Region, list(str)): key-value pairs for regions to their feature set
"""


AVOID_IN_CLF = sorted(
    [
        "bdt_response",
        "eta_met",
        "eta_jetL1",
        "eta_jetS1",
        "sumet",
        "mass_jet1",
        "mass_jet2",
        "mass_jetF",
        "mass_jetL1",
        "mass_jetS1",
        "E_jetL1",
        "E_jetS1",
        "E_jet1",
        "E_jet2",
        "pT_lep3",
        "pT_jetL1",
        "nbjets",
        "njets",
    ],
    key=str.lower,
)
"""
list(str): list of features to avoid in classifiers
"""

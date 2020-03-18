"""
Module for general utilities
"""

# stdlib
import copy
import logging
import numbers
import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import PosixPath

from typing import Union, Iterable, Optional, Dict, List, Tuple, Set

# external
import numpy as np
import scipy.special
import uproot
import pandas as pd
import formulate

# tdub
import tdub.constants

FileLike = Union[str, os.PathLike]
PathLike = Union[str, os.PathLike]
FileOrFiles = Union[List[FileLike], FileLike]
PathOrPaths = Union[List[PathLike], PathLike]

log = logging.getLogger(__name__)


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

    >>> from tdub.utils import Region, get_selection
    >>> from tdub.frames import iterative_selection
    >>> df = iterative_selection(files, get_selection(Region.r2j2b))

    """

    r1j1b = 0
    r2j1b = 1
    r2j2b = 2

    @staticmethod
    def from_str(s: str) -> "Region":
        """Get enum value for the given string

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

    _sample_info_extract_re = re.compile(
        r"""(?P<phy_process>\w+)_
        (?P<dsid>[0-9]{6})_
        (?P<sim_type>(FS|AFII))_
        (?P<campaign>MC16(a|d|e))_
        (?P<tree>\w+)
        (\.\w+|$)""",
        re.X,
    )

    def __init__(self, input_file: str) -> None:
        if "Data_Data" in input_file:
            self.phy_process = "Data"
            self.dsid = 0
            self.sim_type = "Data"
            self.campaign = "Data"
            self.tree = "nominal"
        else:
            m = self._sample_info_extract_re.match(input_file)
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
    source: Union[FileLike, Iterable[str]], tree: str = "WtLoop_nominal",
) -> Dict[str, List[str]]:
    """Categorize branches into a separate lists

    The categories:

    - ``kinematics`` for kinematic features (used for classifiers)
    - ``weights`` for any branch that starts or ends with ``weight``
    - ``meta`` for meta information (final state information)

    Parameters
    ----------
    source : os.PathLike or str or Iterable(str)
       if iterable of strings, use that as list of branches, if
       os.PathLike or str then the source is interpreted as a ROOT
       file and we get the branches by passing the file and ``tree``
       argument to :py:func:`get_branches`.
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
        "isMC16a",
        "isMC16d",
        "isMC16e",
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


def quick_files(datapath: FileLike, campaign: Optional[str] = None) -> Dict[str, List[str]]:
    """Get a dictionary of ``{sample_str : file_list}`` for quick file access.

    The lists of files are sorted alphabetically. These types of
    samples are currently tested:

    - ``ttbar`` (410472 full sim)
    - ``ttbar_AFII`` (410472 fast sim)
    - ``ttbar_PS`` (410558 fast sim)
    - ``ttbar_hdamp`` (410482 fast sim)
    - ``tW_DR`` (410648, 410649 full sim)
    - ``tW_DR_AFII`` (410648, 410648 fast sim)
    - ``tW_DR_PS`` (411038, 411039 fast sim)
    - ``tW_DS`` (410656, 410657 full sim)
    - ``Diboson``
    - ``Zjets``
    - ``MCNP``
    - ``Data``

    Parameters
    ----------
    datapath : str or os.PathLike
       path where all of the ROOT files live
    campaign : str, optional
       enforce a single campaign ("MC16a", "MC16d", or "MC16e")

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
    >>> qf = quick_files("/path/to/some/files", campaign="MC16d")
    >>> pprint(qf["tW_DR"])
    ['/path/to/some/files/tW_DR_410648_FS_MC16d_nominal.root',
     '/path/to/some/files/tW_DR_410649_FS_MC16d_nominal.root']
    >>> qf = quick_files("/path/to/some/files", campaign="MC16a")
    >>> pprint(qf["Data"])
    ['/path/to/some/files/Data15_data15_Data_Data_nominal.root',
     '/path/to/some/files/Data16_data16_Data_Data_nominal.root']

    """
    if campaign is None:
        camp = ""
    else:
        if campaign not in ("MC16a", "MC16d", "MC16e"):
            raise ValueError(f"{campaign} but be either 'MC16a', 'MC16d', or 'MC16e'")
        camp = f"_{campaign}"
    path = str(PosixPath(datapath).resolve())
    ttbar_files = sorted(glob(f"{path}/ttbar_410472_FS{camp}*nominal.root"))
    ttbar_AFII_files = sorted(glob(f"{path}/ttbar_410472_AFII{camp}*nominal.root"))
    ttbar_PS_files = sorted(glob(f"{path}/ttbar_410558*AFII{camp}*nominal.root"))
    ttbar_hdamp_files = sorted(glob(f"{path}/ttbar_410482_AFII{camp}*nominal.root"))
    tW_DR_AFII_files = sorted(glob(f"{path}/tW_DR_41064*AFII{camp}*nominal.root"))
    tW_DR_files = sorted(
        glob(f"{path}/tW_DR_410648_FS{camp}*nominal.root")
        + glob(f"{path}/tW_DR_410649_FS{camp}*nominal.root")
    )
    tW_DR_PS_files = sorted(glob(f"{path}/tW_DR_41103*AFII{camp}*nominal.root"))
    tW_DS_files = sorted(glob(f"{path}/tW_DS_41065*FS{camp}*nominal.root"))
    Diboson_files = sorted(glob(f"{path}/Diboson_*FS{camp}*nominal.root"))
    Zjets_files = sorted(glob(f"{path}/Zjets_*FS{camp}*nominal.root"))
    MCNP_files = sorted(glob(f"{path}/MCNP_*FS{camp}*nominal.root"))

    if campaign is None:
        Data_files = sorted(glob(f"{path}/*Data_Data_nominal.root"))
    elif campaign == "MC16a":
        Data_files = sorted(
            glob(f"{path}/Data15_data15*root") + glob(f"{path}/Data16_data16*root")
        )
    elif campaign == "MC16d":
        Data_files = sorted(glob(f"{path}/Data17_data17*root"))
    elif campaign == "MC16e":
        Data_files = sorted(glob(f"{path}/Data18_data18*root"))

    file_lists = {
        "ttbar": ttbar_files,
        "ttbar_AFII": ttbar_AFII_files,
        "ttbar_PS": ttbar_PS_files,
        "ttbar_hdamp": ttbar_hdamp_files,
        "tW_DR": tW_DR_files,
        "tW_DR_AFII": tW_DR_AFII_files,
        "tW_DR_PS": tW_DR_PS_files,
        "tW_DS": tW_DS_files,
        "Diboson": Diboson_files,
        "Zjets": Zjets_files,
        "MCNP": MCNP_files,
        "Data": Data_files,
    }
    for k, v in file_lists.items():
        if len(v) == 0:
            log.debug(f"we didn't find any files for {k}")
    return file_lists


def files_for_tree(
    datapath: FileLike, sample_prefix: str, tree_name: str, campaign: Optional[str] = None,
) -> List[str]:
    """Get a list of files for the sample and desired tree

    Parameters
    ----------
    datapath : str or os.PathLike
       path where all of the ROOT files live
    sample_prefix : str
       the prefix for the sample we want (``"ttbar"`` or ``"tW_DR"`` or ``"tW_DS"``)
    tree_name : str
       the name of the ATLAS systematic tree (e.g. ``"nominal"`` or ``"EG_RESOLUTION_ALL__1up"``)
    campaign : str, optional
       enforce a single campaign ("MC16a", "MC16d", or "MC16e")

    Returns
    -------
    list(str)
       the list of desired files (if they exist)

    Examples
    --------

    >>> from tdub.utils import files_for_tree
    >>> files_for_tree("/data/path", "ttbar", "JET_CategoryReduction_JET_JER_EffectiveNP_4__1up")
    ['/data/path/ttbar_410472_FS_MC16a_JET_CategoryReduction_JET_JER_EffectiveNP_4__1up.root',
     '/data/path/ttbar_410472_FS_MC16d_JET_CategoryReduction_JET_JER_EffectiveNP_4__1up.root',
     '/data/path/ttbar_410472_FS_MC16e_JET_CategoryReduction_JET_JER_EffectiveNP_4__1up.root']

    """
    if campaign is None:
        camp = ""
    else:
        if campaign not in ("MC16a", "MC16d", "MC16e"):
            raise ValueError(f"{campaign} but be either 'MC16a', 'MC16d', or 'MC16e'")
        camp = f"_{campaign}"

    path = str(PosixPath(datapath).resolve())
    if sample_prefix == "ttbar":
        return sorted(glob(f"{path}/ttbar_410472_FS{camp}*{tree_name}.root"))
    elif sample_prefix == "tW_DR":
        return sorted(glob(f"{path}/tW_DR_41064*FS{camp}*{tree_name}.root"))
    elif sample_prefix == "tW_DS":
        return sorted(glob(f"{path}/tW_DS_41065*FS{camp}*{tree_name}.root"))
    else:
        raise ValueError(
            f"bad sample_prefix '{sample_prefix}', must be one of: ['tW_DR', 'tW_DS', 'ttbar']"
        )


def bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """Get bin centers given bin edges

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
) -> np.array:
    """Create arrays for edges and bin centers

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
    file_name: FileOrFiles,
    tree: str = "WtLoop_nominal",
    ignore_weights: bool = False,
    sort: bool = False,
) -> List[str]:
    """Get list of branches in a ROOT TTree

    Parameters
    ----------
    file_name : str, list(str), os.PathLike, list(os.PathLike)
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
    if isinstance(file_name, list):
        t = uproot.open(file_name[0]).get(tree)
    else:
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
    file_name: FileOrFiles, tree: str = "WtLoop_nominal"
) -> List[str]:
    """Get branches in a ROOT file that form a conservative minimum

    we define "conservative minimum" as the branches necessary for
    using our BDT infrastructure, so this conservative minimum
    includes all of the features used by the BDTs as well as the
    variables necessary for region selection.

    Parameters
    ----------
    file_name : str or os.PathLike or list(str) or list(os.PathLike)
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
    branches_in_file = get_branches(file_name)
    good_branches = set(
        {"reg1j1b", "reg2j1b", "reg2j2b", "OS"}
        | set(tdub.constants.FEATURESET_1j1b)
        | set(tdub.constants.FEATURESET_2j1b)
        | set(tdub.constants.FEATURESET_2j2b)
    )
    good_branches = set(branches_in_file) & good_branches
    return sorted(good_branches, key=str.lower)


def get_selection(region: Union[str, Region]) -> str:
    """Get the selection given a region

    See the tdub.constants module for the defition of the
    selections. See :py:func:`tdub.utils.Region.from_str` for the
    compatible strings.

    Parameters
    ----------
    region : str or tdub.utils.Region
       the region as a string or enum entry

    Returns
    -------
    str
       the selection string

    Examples
    --------

    >>> from tdub.utils import get_selection, Region
    >>> get_selection(Region.r2j1b)
    '(reg2j1b == True) & (OS == True)'
    >>> get_selection("reg1j1b")
    '(reg1j1b == True) & (OS == True)'
    >>> get_selection("2j2b")
    '(reg2j2b == True) & (OS == True)'

    """
    options = {
        Region.r1j1b: tdub.constants.SELECTION_1j1b,
        Region.r2j1b: tdub.constants.SELECTION_2j1b,
        Region.r2j2b: tdub.constants.SELECTION_2j2b,
    }
    if isinstance(region, str):
        return options[Region.from_str(region)]
    return options[region]


def get_avoids(region: Union[str, Region]) -> List[str]:
    """Get the features to avoid for the given region.

    See the tdub.constants module for defintion of the variables to
    avoid. See :py:func:`tdub.utils.Region.from_str` for the
    compatible strings.

    Parameters
    ----------
    region : str or tdub.utils.Region
       the region as a string or enum entry

    Returns
    -------
    list(str)
       the list of features to avoid for the region

    Examples
    --------

    >>> from tdub.utils import get_avoids, Region
    >>> get_avoids(Region.r2j1b)
    ['HT_jet1jet2', 'deltaR_lep1lep2_jet1jet2met', 'mass_lep2jet1', 'pT_jet2']
    >>> get_avoids("2j2b")
    ['deltaR_jet1_jet2']

    """
    options = {
        Region.r1j1b: tdub.constants.AVOID_IN_CLF_1j1b,
        Region.r2j1b: tdub.constants.AVOID_IN_CLF_2j1b,
        Region.r2j2b: tdub.constants.AVOID_IN_CLF_2j2b,
    }
    if isinstance(region, str):
        return options[Region.from_str(region)]
    return options[region]


def get_features(region: Union[str, Region]) -> List[str]:
    """Get the feature list for a region

    See the tdub.constants module for the defition of the feature
    lists. See :py:func:`tdub.utils.Region.from_str` for the
    compatible strings.

    Parameters
    ----------
    region : str or tdub.utils.Region
       the region as a string or enum entry. If ``"ALL"`` returns a
       set of unique features from all regions

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
    # first allow retrieval of all features
    if region == "ALL":
        return sorted(
            set(tdub.constants.FEATURESET_1j1b)
            | set(tdub.constants.FEATURESET_2j1b)
            | set(tdub.constants.FEATURESET_2j2b),
            key=str.lower,
        )

    # if not "ALL" grab from a dict constructed from constants
    options = {
        Region.r1j1b: tdub.constants.FEATURESET_1j1b,
        Region.r2j1b: tdub.constants.FEATURESET_2j1b,
        Region.r2j2b: tdub.constants.FEATURESET_2j2b,
    }
    if isinstance(region, str):
        return options[Region.from_str(region)]
    return options[region]


def augment_features(region: Union[str, Region], to_add: List[str]) -> None:
    """Add some features to the existing lists

    See the tdub.constants module for the defition of the feature
    lists. See :py:func:`tdub.utils.Region.from_str` for the
    compatible strings.

    Parameters
    ----------
    region : str or tdub.utils.Region
       the region as a string or enum entry
    to_add : list(str)
       the new features to add

    Examples
    --------

    >>> from tdub.utils import augment_features, get_features
    >>> "another" in get_features("2j2b")
    False
    >>> "feature" in get_features("2j2b")
    False
    >>> augment_features("2j2b", ["another", "feature"])
    >>> "another" in get_features("2j2b")
    True
    >>> "feature" in get_features("2j2b")
    True

    """
    if isinstance(region, str):
        region = Region.from_str(region)
    if region == Region.r1j1b:
        tdub.constants.FEATURESET_1j1b += to_add
    elif region == Region.r2j1b:
        tdub.constants.FEATURESET_2j1b += to_add
    elif region == Region.r2j2b:
        tdub.constants.FEATURESET_2j2b += to_add
    else:
        raise ValueError("Bad region value")


def override_features(table: Dict[str, List[str]]) -> None:
    """Override feature constants ``tdub.constants.FEATURESET_{1j1b, 2j1b, 2j2b}``

    Given a dictionary of the form

    .. code-block:: python

        overrides = {
            "r1j1b": ["new1", "new2", "new3"],
            "r2j1b": ["new1", "new2", "new3", "new4"],
            "r2j2b": ["new1", "new2"],
        }

    we override the module constants

    - :py:data:`tdub.constants.FEATURESET_1j1b`
    - :py:data:`tdub.constants.FEATURESET_2j1b`
    - :py:data:`tdub.constants.FEATURESET_2j2b`

    Note
    ----

    Not all regions need to be defined; only those you wish to
    override.

    Parameters
    ----------
    table : dict(str, list(str))
       region to feature list table

    Examples
    --------

    Using the dictionary above as an example

    >>> from tdub.utils import override_features, get_features
    >>> import tdub.constants
    >>> tdub.constants.FEATURESET_1j1b
    ["old1", "old2"]
    >>> get_features("1j1b")
    ["old1", "old2"]
    >>> override_features(overrides)
    >>> get_features("1j1b")
    ["new1", "new2", "new3"]
    >>> tdub.constants.FEATURESET_1j1b
    ["new1", "new2", "new3"]

    """
    if "r1j1b" in table:
        log.info("Overriding tdub.constants.FEATURESET_1j1b")
        tdub.constants.FEATURESET_1j1b = copy.deepcopy(table["r1j1b"])
    if "r2j1b" in table:
        log.info("Overriding tdub.constants.FEATURESET_2j1b")
        tdub.constants.FEATURESET_2j1b = copy.deepcopy(table["r2j1b"])
    if "r2j2b" in table:
        log.info("Overriding tdub.constants.FEATURESET_2j2b")
        tdub.constants.FEATURESET_2j2b = copy.deepcopy(table["r2j2b"])


def kolmogorov_prob(z: float) -> float:
    """Calculates the Kolmogorov distribution function

    See ROOT's implementation in TMath_ (TMath::KolmogorovProb).

    .. _TMath: https://root.cern.ch/doc/master/namespaceTMath.html

    Parameters
    ----------
    z : float
       the value to test

    Returns
    -------
    float
       the probability that the test statistic exceeds :math:`z`
       (assuming the null hypothesis).

    Examples
    --------

    >>> from tdub.utils import kolmogorov_prob
    >>> kolmogorov_prob(1.13)
    0.15549781841748692

    """
    w = 2.50662827
    # c1 - -pi**2/8, c2 = 9*c1, c3 = 25*c1
    c1 = -1.2337005501361697
    c2 = -11.103304951225528
    c3 = -30.842513753404244
    u = abs(z)
    if u < 0.2:
        return 1.0
    elif u < 0.755:
        v = 1.0 / (u * u)
        return float(1 - w * (math.exp(c1 * v) + math.exp(c2 * v) + math.exp(c3 * v)) / u)
    elif u < 6.8116:
        fj = np.array([-2, -8, -18, -32], dtype=np.float64)
        r = np.zeros((4,), dtype=np.float64)
        v = u * u
        maxj = max(1.0, round(3.0 / u))
        for j in range(int(maxj)):
            r[j] = math.exp(fj[j] * v)
        return float(2 * (r[0] - r[1] + r[2] - r[3]))
    else:
        return float(0.0)


def ks_twosample_binned(
    hist1: np.ndarray, hist2: np.ndarray, err1: np.ndarray, err2: np.ndarray
) -> Tuple[float, float]:
    """Calculate KS statistic and p-value for two binned distributions

    See ROOT's implementation in TH1_ (TH1::KolmogorovTest).

    .. _TH1: https://root.cern.ch/doc/master/classTH1.html

    Parameters
    ----------
    hist1 : numpy.ndarray
       the histogram counts for the first distribution
    hist2 : numpy.ndarray
       the histogram counts for the second distribution
    err1 : numpy.ndarray
       the error on the histogram counts for the first distribution
    err2 : numpy.ndarray
       the error on the histogram counts for the second distribution

    Returns
    -------
    (float, float)
       first: the test-statistic; second: the probability of the test
       (much less than 1 means distributions are incompatible)

    Examples
    --------

    >>> import pygram11
    >>> from tdub.utils import ks_twosample_binned
    >>> data1, data2, w1, w2 = some_function_to_get_data()
    >>> h1, err1 = pygram11.histogram(data1, weights=w1, bins=40, range=(-3, 3))
    >>> h2, err2 = pygram11.histogram(data2, weights=w2, bins=40, range=(-3, 3))
    >>> kst, ksp = ks_twosample_binned(h1, h2, err1, err2)

    """
    sum1 = np.sum(hist1)
    sum2 = np.sum(hist2)
    w1 = np.sum(err1 * err1)
    w2 = np.sum(err2 * err2)
    esum1 = sum1 * sum1 / w1
    esum2 = sum2 * sum2 / w2
    s1 = 1 / sum1
    s2 = 1 / sum2
    rsum1 = s1 * hist1
    rsum2 = s2 * hist2
    dfmax = float(np.max(np.abs(rsum1 - rsum2)))
    z = float(dfmax * math.sqrt(esum1 * esum2 / (esum1 + esum2)))
    return dfmax, kolmogorov_prob(z)


def chisquared_cdf_c(chi2: float, ndf: float) -> float:
    r"""Calculate :math:`\chi^2` probability from the value and degrees of freedom

    See ROOT's ``TMath::Prob`` & ``ROOT::Math::chisquared_cdf_c``. Quoting
    the ROOT documentation:

    Computation of the probability for a certain :math:`\chi^2` and
    number of degrees of freedom (ndf). Calculations are based on the
    incomplete gamma function :math:`P(a,x)`, where
    :math:`a=\mathrm{ndf}/2` and :math:`x=\chi^2/2`.

    :math:`P(a,x)` represents the probability that the observed
    :math:`\chi^2` for a correct model should be less than the value
    :math:`\chi^2`. The returned probability corresponds to
    :math:`1-P(a,x)`, which denotes the probability that an observed
    :math:`\chi^2` exceeds the value :math:`\chi^2` by chance, even
    for a correct model.

    Parameters
    ----------
    chi2 : float
       the :math:`\chi^2` value
    ndf : float
       the degrees of freedom

    Returns
    -------
    float
       the :math:`\chi^2` probability

    """
    return scipy.special.gammaincc(0.5 * ndf, 0.5 * chi2)


def chisquared_test(
    h1: np.ndarray, err1: np.ndarray, h2: np.ndarray, err2: np.ndarray,
) -> Tuple[float, float, float]:
    r"""Perform :math:`\chi^2` test on two histograms

    Parameters
    ----------
    h1 : :py:obj:`numpy.ndarray`
       the first histogram bin contents
    h2 : :py:obj:`numpy.ndarray`
       the second histogram bin contents
    err1 : :py:obj:`numpy.ndarray`
       the first histogram bin errors
    err2 : :py:obj:`numpy.ndarray`
       the second histogram bin errors

    Returns
    -------
    (float, int, float)
       the :math:`\chi^2` test value, the degrees of freedom, and the probability
    """
    # remove 0 bin heights
    badh1 = h1 * h1 == 0
    badh2 = h2 * h2 == 0
    good_bins = np.invert(np.logical_or(badh1, badh2))
    histo1 = h1[good_bins]
    histo2 = h2[good_bins]
    error1 = err1[good_bins]
    error2 = err2[good_bins]

    # degrees of freedom
    nbins = histo1.shape[0]
    ndf = nbins - 1

    # sums
    sum1 = np.sum(histo1)
    sum2 = np.sum(histo2)
    sumw1 = np.sum(error1 * error1)
    sumw2 = np.sum(error2 * error2)

    # chi2 calculation
    error1sq = error1 * error1
    error2sq = error2 * error2
    sigma = sum1 * sum2 * error2sq + sum2 * sum2 * error1sq
    delta = sum2 * histo1 - sum1 * histo2
    chi2 = np.sum(delta * delta / sigma)

    return (chi2, ndf, chisquared_cdf_c(chi2, ndf))


def extended_selection(region: Union[Region, str], extra: str) -> str:
    """Construct an extended selection string for a region

    Parameters
    ----------
    region : str or tdub.utils.Region
        the region as a string or enum entry
    extra : str
        the extra selection string

    Returns
    -------
    str
        the complete new selection string

    Examples
    --------

    >>> from tdub.utils import extended_selection
    >>> extended_selection("2j2b", "met < 120")
    '((reg2j2b == True) & (OS == True)) & (met < 120)'

    """
    raw = get_selection(region)
    return f"({raw}) & ({extra})"


def minimal_branches(selection: str) -> Set[str]:
    """Get the minimal set of branches necessary to perform a selection

    Parameters
    ----------
    selection : str
        the selection string

    Returns
    -------
    set(str)
        the set of necessary branches/variables

    Examples
    --------

    >>> from tdub.utils import minimal_selection_branches
    >>> selection = "(reg1j1b == True) & (OS == True) & (mass_lep1lep2 > 100)"
    >>> minimal_branches(selection)
    {'OS', 'mass_lep1lep2', 'reg1j1b'}

    """
    return formulate.from_numexpr(selection).variables

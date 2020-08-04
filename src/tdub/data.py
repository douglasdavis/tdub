"""Module for working with datasets."""

# stdlib
from dataclasses import dataclass
from enum import Enum
from glob import glob
from pathlib import PosixPath
import logging
import os
import re
from typing import Union, Set, Dict, Iterable, List, Optional

# external
import formulate
import uproot4
from uproot4.reading import ReadOnlyDirectory
from uproot4.models.TTree import Model_TTree_v20

# tdub
import tdub.config

log = logging.getLogger(__name__)

DataSource = Union[
    str,
    Iterable[str],
    os.PathLike,
    Iterable[os.PathLike],
    ReadOnlyDirectory,
    Model_TTree_v20,
]


class Region(Enum):
    """A simple enum class for easily using region information.

    Attributes
    ----------
    r1j1b
        Label for our `1j1b` region.
    r2j1b
        Label for our `2j1b` region.
    r2j2b = 1
        Label for our `2j2b` region.

    Examples
    --------
    Using this enum for grabing the ``2j2b`` region from a set of
    files:

    >>> from tdub.data import Region, selection_for
    >>> from tdub.frames import iterative_selection
    >>> df = iterative_selection(files, selection_for(Region.r2j2b))

    """

    r1j1b = 0
    r2j1b = 1
    r2j2b = 2
    rUnkn = 9

    @staticmethod
    def from_str(s: str) -> "Region":
        """Get enum value for the given string.

        This function supports three ways to define a region; prefixed
        with "r", prefixed with "reg", or no prefix at all. For
        example, ``Region.r2j2b`` can be retrieved like so:

        - ``Region.from_str("r2j2b")``
        - ``Region.from_str("reg2j2b")``
        - ``Region.from_str("2j2b")``

        Parameters
        ----------
        s : str
            String representation of the desired region

        Returns
        -------
        Region
            Enum version

        Examples
        --------
        >>> from tdub.data import Region
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


_sample_info_extract_re = re.compile(
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
    """Describes a sample's attritubes given it's name.

    Parameters
    ----------
    input_file : str
        File stem containing the necessary groups to parse.

    Attributes
    ----------
    phy_process : str
        Physics process (e.g. ttbar or tW_DR or Zjets)
    dsid : int
        Dataset ID
    sim_type : str
        Simulation type, "FS" or "AFII"
    campaign : str
        Campaign, MC16{a,d,e}
    tree : str
        Original tree (e.g. "nominal" or "EG_SCALE_ALL__1up")

    Examples
    --------
    >>> from tdub.data import SampleInfo
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

    def __init__(self, input_file: str) -> None:
        if "Data_Data" in input_file:
            self.phy_process = "Data"
            self.dsid = 0
            self.sim_type = "Data"
            self.campaign = "Data"
            self.tree = "nominal"
        else:
            m = _sample_info_extract_re.match(input_file)
            if not m:
                raise ValueError(f"{input_file} cannot be parsed by SampleInfo regex")
            self.phy_process = m.group("phy_process")
            if self.phy_process.startswith("MCNP"):
                self.phy_process = "MCNP"
            self.dsid = int(m.group("dsid"))
            self.sim_type = m.group("sim_type")
            self.campaign = m.group("campaign")
            self.tree = m.group("tree")


def avoids_for(region: Union[str, Region]) -> List[str]:
    """Get the features to avoid for the given region.

    See the :py:mod:`tdub.config` module for definition of the
    variables to avoid (and how to modify them).

    Parameters
    ----------
    region : str or tdub.data.Region
        Region to get the associated avoided branches.

    Returns
    -------
    list(str)
        Features to avoid for the region.

    Examples
    --------
    >>> from tdub.data import avoids_for, Region
    >>> avoids_for(Region.r2j1b)
    ['HT_jet1jet2', 'deltaR_lep1lep2_jet1jet2met', 'mass_lep2jet1', 'pT_jet2']
    >>> avoids_for("2j2b")
    ['deltaR_jet1_jet2']

    """
    if isinstance(region, str):
        region = Region.from_str(region)
    if region == Region.r1j1b:
        return tdub.config.AVOID_IN_CLF_1j1b
    elif region == Region.r2j1b:
        return tdub.config.AVOID_IN_CLF_2j1b
    elif region == Region.r2j2b:
        return tdub.config.AVOID_IN_CLF_2j2b
    else:
        raise ValueError(f"Incompatible region: {region}")


def branches_from(
    source: DataSource, tree: str = "WtLoop_nominal", ignore_weights: bool = False,
) -> List[str]:
    """Get a list of branches from a data source.

    If the `source` is a list of files, the first file is the only
    file that is parsed.

    Parameters
    ----------
    source : str, list(str), os.PathLike, list(os.PathLike), or uproot File/Tree
        What to parse to get the branch information.
    tree : str
        Name of the tree to get branches from
    ignore_weights : bool
        Flag to ignore all branches starting with `weight_`.

    Returns
    -------
    list(str)
        Branches from the source.

    Examples
    --------
    >>> from tdub.data import branches_from
    >>> branches_from("/path/to/file.root", ignore_weights=True)
    ["pT_lep1", "pT_lep2"]
    >>> branches_from("/path/to/file.root")
    ["pT_lep1", "pT_lep2", "weight_nominal", "weight_tptrw"]

    """
    print(type(source))
    if isinstance(source, (str, os.PathLike)):
        t = uproot4.open(source).get(tree)
    elif isinstance(source, list):
        t = uproot4.open(source[0]).get(tree)
    elif isinstance(source, uproot4.reading.ReadOnlyDirectory):
        t = source.get(tree)
    elif isinstance(source, uproot4.models.TTree.Model_TTree_v20):
        t = source
    branches = t.keys()

    # return here if weights are not ignored
    if not ignore_weights:
        return branches

    # check for weight branches
    weight_re = re.compile(r"(weight_\w+)")
    weights = set(filter(weight_re.match, branches))

    return list(sorted(set(branches) ^ weights, key=str.lower))


def categorize_branches(source: List[str]) -> Dict[str, List[str]]:
    """Categorize branches into a separated lists.

    The categories:

    - `kinematics`: for kinematic features (used for classifiers)
    - `weights`: for any branch that starts or ends with ``weight``
    - `meta`: for meta information (final state information)

    Parameters
    ----------
    source : list(str)
        Complete list of branches to be categorized.

    Returns
    -------
    dict(str, list(str))
        Dictionary connecting categories to their associated list of
        branchess.

    Examples
    --------
    >>> from tdub.data import categorize_branches, branches_from
    >>> branches = ["pT_lep1", "pT_lep2", "weight_nominal", "weight_sys_jvt", "reg2j2b"]
    >>> cated = categorize_branches(branches)
    >>> cated["weights"]
    ['weight_sys_jvt', 'weight_nominal']
    >>> cated["meta"]
    ['reg2j2b']
    >>> cated["kinematics"]
    ['pT_lep1', 'pT_lep2']

    Using a ROOT file:

    >>> root_file = PosixPath("/path/to/file.root")
    >>> cated = categorize_branches(branches_from(root_file))

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


def features_for(region: Union[str, Region]) -> List[str]:
    """Get the feature list for a region.

    See the :py:mod:`tdub.config` module for the definitions of the
    feature lists (and how to modify them).

    Parameters
    ----------
    region : str or tdub.data.Region
        Region as a string or enum entry. Using ``"ALL"`` returns a
        list of unique features from all regions.

    Returns
    -------
    list(str)
        Features for that region (or all regions).

    Examples
    --------
    >>> from pprint import pprint
    >>> from tdub.data import features_for
    >>> pprint(features_for("reg2j1b"))
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
            set(tdub.config.FEATURESET_1j1b)
            | set(tdub.config.FEATURESET_2j1b)
            | set(tdub.config.FEATURESET_2j2b),
            key=str.lower,
        )

    # if not "ALL" grab from a dict constructed from config
    if isinstance(region, str):
        region = Region.from_str(region)
    if region == Region.r1j1b:
        return tdub.config.FEATURESET_1j1b
    if region == Region.r2j1b:
        return tdub.config.FEATURESET_2j1b
    if region == Region.r2j2b:
        return tdub.config.FEATURESET_2j2b
    else:
        raise ValueError(f"Incompatible region: {region}")


def files_for_tree(
    datapath: Union[str, os.PathLike],
    sample_prefix: str,
    tree_name: str,
    campaign: Optional[str] = None,
) -> List[str]:
    """Get a list of files for the sample and desired systematic tree.

    Parameters
    ----------
    datapath : str or os.PathLike
        Path where ROOT files are expected to live.
    sample_prefix : str
        Prefix for the sample we want (`"ttbar"` or `"tW_DR"` or `"tW_DS"`).
    tree_name : str
        Name of the ATLAS systematic tree (e.g. `"nominal"` or `"EG_RESOLUTION_ALL__1up"`).
    campaign : str, optional
        Enforce a single campaign ("MC16a", "MC16d", or "MC16e").

    Returns
    -------
    list(str)
        Desired files (if they exist)

    Examples
    --------
    >>> from tdub.data import files_for_tree
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


def quick_files(
    datapath: Union[str, os.PathLike], campaign: Optional[str] = None
) -> Dict[str, List[str]]:
    """Get a dictionary connecting sample processes to file lists.

    The lists of files are sorted alphabetically. These types of
    samples are currently tested:

    - `ttbar` (410472 full sim)
    - `ttbar_AFII` (410472 fast sim)
    - `ttbar_PS` (410558 fast sim)
    - `ttbar_PS713` (411234 fast sim)
    - `ttbar_hdamp` (410482 fast sim)
    - `ttbar_inc` (410470 full sim)
    - `ttbar_inc_AFII` (410470 fast sim)
    - `tW_DR` (410648, 410649 full sim)
    - `tW_DR_AFII` (410648, 410648 fast sim)
    - `tW_DR_PS` (411038, 411039 fast sim)
    - `tW_DR_inc` (410646, 410647 full sim)
    - `tW_DR_inc_AFII` (410646, 410647 fast sim)
    - `tW_DS` (410656, 410657 full sim)
    - `tW_DS_inc` (410654, 410655 ful sim)
    - `Diboson`
    - `Zjets`
    - `MCNP`
    - `Data`

    Parameters
    ----------
    datapath : str or os.PathLike
        Path where all of the ROOT files live.
    campaign : str, optional
        Enforce a single campaign ("MC16a", "MC16d", or "MC16e").

    Returns
    -------
    dict(str, list(str))
        The dictionary of processes and their associated files.

    Examples
    --------
    >>> from pprint import pprint
    >>> from tdub.data import quick_files
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

    # ttbar
    ttbar_files = sorted(glob(f"{path}/ttbar_410472_FS{camp}*nominal.root"))
    ttbar_AFII_files = sorted(glob(f"{path}/ttbar_410472_AFII{camp}*nominal.root"))
    ttbar_PS_files = sorted(glob(f"{path}/ttbar_410558*AFII{camp}*nominal.root"))
    ttbar_PS713_files = sorted(glob(f"{path}/ttbar_411234*AFII{camp}*nominal.root"))
    ttbar_hdamp_files = sorted(glob(f"{path}/ttbar_410482_AFII{camp}*nominal.root"))
    ttbar_inc_files = sorted(glob(f"{path}/ttbar_410470_FS{camp}*nominal.root"))
    ttbar_inc_AFII_files = sorted(glob(f"{path}/ttbar_410470_AFII{camp}*nominal.root"))

    # tW
    tW_DR_files = sorted(
        glob(f"{path}/tW_DR_410648_FS{camp}*nominal.root")
        + glob(f"{path}/tW_DR_410649_FS{camp}*nominal.root")
    )
    tW_DR_AFII_files = sorted(
        glob(f"{path}/tW_DR_410648_AFII{camp}*nominal.root")
        + glob(f"{path}/tW_DR_410649_AFII{camp}*nominal.root")
    )
    tW_DR_inc_files = sorted(
        glob(f"{path}/tW_DR_410646_FS{camp}*nominal.root")
        + glob(f"{path}/tW_DR_410647_FS{camp}*nominal.root")
    )
    tW_DR_inc_AFII_files = sorted(
        glob(f"{path}/tW_DR_410646_AFII{camp}*nominal.root")
        + glob(f"{path}/tW_DR_410647_AFII{camp}*nominal.root")
    )
    tW_DR_PS_files = sorted(
        glob(f"{path}/tW_DR_411038_AFII{camp}*nominal.root")
        + glob(f"{path}/tW_DR_411039_AFII{camp}*nominal.root")
    )
    tW_DS_files = sorted(
        glob(f"{path}/tW_DS_410656_FS{camp}*nominal.root")
        + glob(f"{path}/tW_DS_410657_FS{camp}*nominal.root")
    )
    tW_DS_inc_files = sorted(
        glob(f"{path}/tW_DS_410654_FS{camp}*nominal.root")
        + glob(f"{path}/tW_DS_410655_FS{camp}*nominal.root")
    )

    # Minor backgrounds
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
        "ttbar_PS713": ttbar_PS713_files,
        "ttbar_hdamp": ttbar_hdamp_files,
        "ttbar_inc": ttbar_inc_files,
        "ttbar_inc_AFII": ttbar_inc_AFII_files,
        "tW_DR": tW_DR_files,
        "tW_DR_AFII": tW_DR_AFII_files,
        "tW_DR_PS": tW_DR_PS_files,
        "tW_DR_inc": tW_DR_inc_files,
        "tW_DR_inc_AFII": tW_DR_inc_AFII_files,
        "tW_DS": tW_DS_files,
        "tW_DS_inc": tW_DS_inc_files,
        "Diboson": Diboson_files,
        "Zjets": Zjets_files,
        "MCNP": MCNP_files,
        "Data": Data_files,
    }
    for k, v in file_lists.items():
        if len(v) == 0:
            log.debug(f"we didn't find any files for {k}")
    return file_lists


def selection_as_numexpr(selection: str) -> str:
    """Get the numexpr selection string from an arbitrary selection.

    Parameters
    -----------
    selection : str
        Selection string in ROOT or numexpr

    Returns
    -------
    str
        Selection in numexpr format.

    Examples
    --------
    >>> selection = "reg1j1b == true && OS == true && mass_lep1jet1 < 155"
    >>> from tdub.data import selection_as_numexpr
    >>> selection_as_numexpr(selection)
    '(reg1j1b == True) & (OS == True) & (mass_lep1jet1 < 155)'

    """
    return formulate.from_auto(selection).to_numexpr()


def selection_as_root(selection: str) -> str:
    """Get the ROOT selection string from an arbitrary selection

    Parameters
    -----------
    selection : str
        The selection string in ROOT or numexpr

    Returns
    -------
    str
        The same selection in ROOT format.

    Examples
    --------
    >>> selection = "(reg1j1b == True) & (OS == True) & (mass_lep1jet1 < 155)"
    >>> from tdub.data import selection_as_root
    >>> selection_as_root(selection)
    '(reg1j1b == true) && (OS == true) && (mass_lep1jet1 < 155)'

    """
    return formulate.from_auto(selection).to_root()


def selection_branches(selection: str) -> Set[str]:
    """Construct the minimal set of branches required for a selection.

    Parameters
    -----------
    selection : str
        Selection string in ROOT or numexpr

    Returns
    -------
    set(str)
        Necessary branches/variables

    Examples
    --------
    >>> from tdub.data import minimal_selection_branches
    >>> selection = "(reg1j1b == True) & (OS == True) & (mass_lep1lep2 > 100)"
    >>> minimal_branches(selection)
    {'OS', 'mass_lep1lep2', 'reg1j1b'}
    >>> selection = "reg2j1b == true && OS == true && (mass_lep1jet1 < 155)"
    >>> minimal_branches(selection)
    {'OS', 'mass_lep1jet1', 'reg2j1b'}

    """
    return formulate.from_auto(selection).variables


def selection_for(region: Union[str, Region], additional: Optional[str] = None) -> str:
    """Get the selection for a given region.

    We have three regions with a default selection (`1j1b`, `2j1b`,
    and `2j2b`), these are the possible argument options (in str or
    Enum form). See the :py:mod:`tdub.config` module for the
    definitions of the selections (and how to modify them).

    Parameters
    ----------
    region : str or Region
        Region to get the selection for
    additional : str, optional
        Additional selection (in ROOT or numexpr form). This will
        connect the region specific selection using `and`.

    Returns
    -------
    str
        Selection string in numexpr format.

    Examples
    --------
    >>> from tdub.data import Region, selection_for
    >>> selection_for(Region.r2j1b)
    '(reg2j1b == True) & (OS == True)'
    >>> selection_for("reg1j1b")
    '(reg1j1b == True) & (OS == True)'
    >>> selection_for("2j2b")
    '(reg2j2b == True) & (OS == True)'
    >>> selection_for("2j2b", additional="minimaxmbl < 155")
    '((reg2j2b == True) & (OS == True)) & (minimaxmbl < 155)'
    >>> selection_for("2j1b", additional="mass_lep1jetb < 155 && mass_lep2jetb < 155")
    '((reg1j1b == True) & (OS == True)) & ((mass_lep1jetb < 155) & (mass_lep2jetb < 155))'

    """
    if isinstance(region, str):
        region = Region.from_str(region)

    if region == Region.r1j1b:
        selection = "(reg1j1b == True) & (OS == True)"
    elif region == Region.r2j1b:
        selection = "(reg2j1b == True) & (OS == True)"
    elif region == Region.r2j2b:
        selection = "(reg2j2b == True) & (OS == True)"
    else:
        raise ValueError("Incompatible region used")

    if additional is not None:
        additional = selection_as_numexpr(additional)
        selection = f"({selection}) & ({additional})"

    return selection

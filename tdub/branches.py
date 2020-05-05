"""Module for using/handling branches in ROOT trees"""

# stdlib
import os
import re
from typing import Union, Set, Dict, Iterable, List

# external
import formulate
import uproot

# tdub
import tdub.constants
from tdub.utils import Region, FileLike, FileOrFiles


def categorize_branches(
    source: Union[FileLike, Iterable[str]], tree: str = "WtLoop_nominal",
) -> Dict[str, List[str]]:
    """Categorize branches into a separate lists.

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
    >>> from tdub.branches import categorize_branches
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


def get_branches(
    file_name: FileOrFiles,
    tree: str = "WtLoop_nominal",
    ignore_weights: bool = False,
    sort: bool = False,
) -> List[str]:
    """Get list of branches in a ROOT TTree.

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

    >>> from tdub.branches import get_branches
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


def get_selection(region: Union[str, Region]) -> str:
    """Get the selection given a region.

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
    >>> from.branches import get_selection
    >>> from tdub.utils import Region
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


def extended_selection(region: Union[Region, str], extra: str) -> str:
    """Construct an extended selection string for a region.

    `extra` can be a ROOT or numexpr string, but a numexpr selection
    is always returned. You can convert it back to ROOT with
    :py:func:`root_selection`.

    Parameters
    ----------
    region : str or tdub.utils.Region
        the region as a string or enum entry
    extra : str
        the extra selection string (in ROOT or numexpr)

    Returns
    -------
    str
        the complete new selection string (numexpr format)

    Examples
    --------
    >>> from tdub.branches import extended_selection
    >>> extended_selection("2j2b", "met < 120")
    '((reg2j2b == True) & (OS == True)) & (met < 120)'

    """
    extra_sel = formulate.from_auto(extra).to_numexpr()
    raw = get_selection(region)
    return f"({raw}) & ({extra_sel})"


def numexpr_selection(selection: str) -> str:
    """Get the numexpr selection string from an arbitrary selection

    Parameters
    -----------
    selection : str
        The selection string in ROOT or numexpr

    Returns
    -------
    str
        The same selection in numexpr format.

    Examples
    --------
    >>> selection = "reg1j1b == true && OS == true && mass_lep1jet1 < 155"
    >>> from tdub.branches import numexpr_selection
    >>> numexpr_selection(selection)
    '(reg1j1b == True) & (OS == True) & (mass_lep1jet1 < 155)'
    """
    return formulate.from_auto(selection).to_numexpr()


def root_selection(selection: str) -> str:
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
    >>> from tdub.branches import root_selection
    >>> root_selection(selection)
    '(reg1j1b == true) && (OS == true) && (mass_lep1jet1 < 155)'
    """
    return formulate.from_auto(selection).to_root()


def minimal_branches(selection: str) -> Set[str]:
    """Get the minimal set of branches for a selection.

    The selection can either be ROOT or numexpr format.

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
    >>> from tdub.branches import minimal_selection_branches
    >>> selection = "(reg1j1b == True) & (OS == True) & (mass_lep1lep2 > 100)"
    >>> minimal_branches(selection)
    {'OS', 'mass_lep1lep2', 'reg1j1b'}
    >>> selection = "reg2j1b == true && OS == true && (mass_lep1jet1 < 155)"
    >>> minimal_branches(selection)
    {'OS', 'mass_lep1jet1', 'reg2j1b'}
    """
    return formulate.from_auto(selection).variables

from __future__ import annotations

from glob import glob
from pathlib import PosixPath
import re
import uproot


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
    return {"weights": list(weights), "kin": list(kins), "meta": list(metas)}


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
    return {
        "ttbar": ttbar_files,
        "tW_DR": tW_DR_files,
        "tW_DS": tW_DS_files,
        "Diboson": Diboson_files,
        "Zjets": Zjets_files,
        "MCNP": MCNP_files,
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

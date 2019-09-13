from __future__ import annotations

from glob import glob
from pathlib import PosixPath
import re


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
    """get a dictionary of ``{sample_str : file_list}`` for quick file access

    These types of samples are currently tested:

    - ``ttbar``
    - ``tW_DR``
    - ``tW_DS``

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
    ['/path/to/some/files/ttbar_410472_FS_MC16d_nominal.root',
     '/path/to/some/files/ttbar_410472_FS_MC16e_nominal.root',
     '/path/to/some/files/ttbar_410472_FS_MC16a_nominal.root']

    """
    path = str(PosixPath(datapath).resolve())
    ttbar_files = glob(f"{path}/ttbar_410472_FS*nominal.root")
    tW_DR_files = glob(f"{path}/tW_DR_41064*FS*nominal.root")
    tW_DS_files = glob(f"{path}/tW_DS_41065*FS*nominal.root")
    return {"ttbar": ttbar_files, "tW_DR": tW_DR_files, "tW_DS": tW_DS_files}

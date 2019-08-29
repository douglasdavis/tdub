from glob import glob
from pathlib import PosixPath
from typing import Dict, List
import re


def categorize_branches(branches: List[str]) -> Dict[str, List[str]]:
    """categorize branches into a separate lists

    The categories:

    - ``meta`` for meta information (final state information)
    - ``kin`` for kinematic features (used for classifiers)
    - ``weights`` for any branch satisfying the regex ``^weight_``

    Parameters
    ----------
    branches : List[str]
       whole set of branches (columns from dataframes)

    Returns
    -------
    dict(str, list(str))
       dictionary of ``{category : list-of-branches}``
    """
    metas = {
        "reg1j1b",
        "reg2j1b",
        "reg2j2b",
        "reg3j",
        "OS",
        "SS",
        "elmu",
        "elel",
        "mumu",
        "runNumber",
        "eventNumber",
    }
    weight_re = re.compile("^weight_")
    weights = set(filter(weight_re.match, branches))
    metas = metas & set(branches)
    kins = (set(branches) ^ weights) ^ metas
    return {"weights": list(weights), "kin": list(kins), "meta": list(metas)}


def quick_files(datapath: str) -> Dict[str, List[str]]:
    """get a dictionary of ``{sample_str : file_list}`` for quick file access

    Parameters
    ----------
    datapath : str
        path where all of the ROOT files live

    Returns
    -------
    dict(str, list(str))
        dictionary for quick file access

    """
    path = str(PosixPath(datapath).resolve())
    ttbar_files = glob(f"{path}/ttbar_410472_FS*nominal.root")
    tW_DR_files = glob(f"{path}/tW_DR_41064*FS*nominal.root")
    tW_DS_files = glob(f"{path}/tW_DS_41065*FS*nominal.root")
    return {"ttbar": ttbar_files, "tW_DR": tW_DR_files, "tW_DS": tW_DS_files}

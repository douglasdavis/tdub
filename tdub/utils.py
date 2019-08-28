from glob import glob
from pathlib import PosixPath
from typing import Dict, List


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

"""
Module for handling datasets
"""

from __future__ import annotations

import numpy as np
import uproot
import dask.dataframe as dd
from dask.delayed import delayed
from typing import List, Union, Optional, Dict
import logging

log = logging.getLogger(__name__)

def delayed_dataframe(
    root_files: Union[str, List[str]],
    tree_name: str = "WtLoop_nominal",
    branches: Optional[List[str]] = None,
) -> dd.DataFrame:
    """Construct a dask flavored DataFrame from delayed uproot tree reading

    We use uproot's :meth:`uproot.TTreeMethods_pandas.df` implementation.

    Parameters
    ----------
    root_files : list(str) or str
       a single ROOT file or list of ROOT files
    tree_name : str
       the tree name to turn into a dataframe
    branches : list(str), optional
       a list of branches to include as columns in the dataframe,
       default is ``None``, includes all branches.

    Returns
    -------
    df : :obj:`dask.dataframe.DataFrame`
       a dask dataframe created via :meth:`dask.dataframe.from_delayed`

    Examples
    --------
    >>> from glob import glob
    >>> files = glob("/path/to/files/*.root")
    >>> ddf = delayed_df(files, branches=["branch_a", "branch_b"])

    """
    if isinstance(root_files, str):
        files = [root_files]
    else:
        files = root_files

    @delayed
    def get_frame(f, tn):
        tree = uproot.open(f)[tn]
        return tree.pandas.df(branches=branches)

    dfs = [get_frame(f, tree_name) for f in files]

    df = dd.from_delayed(dfs)
    return df


def selected_dataframes(
    root_files: Union[str, List[str]],
    tree_name: str = "WtLoop_nominal",
    selections: Dict[str, str] = {},
    branches: Optional[List[str]] = None,
) -> Dict[str, dd.DataFrame]:
    """Construct a set of dataframes based on a list of selection queries

    Parameters
    ----------
    root_files : list(str) or str
       a single ROOT file or list of ROOT files
    tree_name : str
       the tree name to turn into a dataframe
    selections : dict(str,str)
       the list of selections to apply on the dataframe in the form
       ``(name, query)``.
    branches : list(str), optional
       a list of branches to include as columns in the dataframe,
       default is ``None`` (all branches)

    Returns
    -------
    selected_dfs : dict(str, dask.dataframe.DataFrame)
       list of DataFrames satisfying selection string

    Examples
    --------
    >>> from glob import glob
    >>> files = glob("/path/to/files/*.root")
    >>> selections = {"r2j2b": "(reg2j2b == True) & (OS == True)",
    ...               "r2j1b": "(reg2j1b == True) & (OS == True)"}
    >>> frames = selected_dataframes(files, selections=selections)
    """
    df = delayed_dataframe(root_files, tree_name, branches)
    return {sel_name: df.query(sel_query) for sel_name, sel_query in selections.items()}


def stdregion_dataframes(
    root_files: Union[str, List[str]],
    tree_name: str = "WtLoop_nominal",
    branches: Optional[List[str]] = None,
) -> Dict[str, dd.DataFrame]:
    """Prepare our standard regions (selections) from a master dataframe

    This is just a hardcoded call of :meth:`selected_dataframes`. By
    standard selection we mean our good ole:

      - ``1j1b``
      - ``2j1b``
      - ``2j2b``
      - ``3j``

    Parameters
    ----------
    root_files : list(str) or str
       a single ROOT file or list of ROOT files
    tree_name : str
       the tree name to turn into a dataframe
    branches : list(str), optional
       a list of branches to include as columns in the dataframe,
       default is ``None`` (all branches)

    Returns
    -------
    selected_dfs : dict(str, dask.dataframe.DataFrame)
       dictionary of ``{name, DataFrame}`` satisfying standard region selections

    Examples
    --------
    >>> from glob import glob
    >>> files = glob("/path/to/files/*.root")
    >>> standard_regions = stdregion_dataframes(files)

    """

    selections = {
        "reg1j1b": "(reg1j1b == True) & (OS == True)",
        "reg2j1b": "(reg2j1b == True) & (OS == True)",
        "reg2j2b": "(reg2j2b == True) & (OS == True)",
        "reg3j": "(reg3j == True) & (OS == True)",
    }
    use_branches = None
    if branches is not None:
        use_branches = set(branches) | set(["reg1j1b", "reg2j1b", "reg2j2b", "reg3j", "OS"])
    return selected_dataframes(root_files, tree_name, selections, use_branches)

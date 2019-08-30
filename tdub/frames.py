"""
Module for handling dataframes
"""

from __future__ import annotations

import uproot
import logging
import cachetools
import dask
import dask.dataframe as dd
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass, field
from tdub.regions import *
from tdub.utils import categorize_branches

log = logging.getLogger(__name__)


class DataFramesInMemory:
    """A dataset structured with everything living on RAM

    Attributes
    ----------
    name : str
       dataset name
    df : :obj:`pandas.DataFrame`
       payload dataframe, for meaningful kinematic features
    weights : :obj:`pandas.DataFrame`
       dataframe to hold weight information

    Parameters
    ----------
    name : str
        dataset name
    ddf : :obj:`dask.dataframe.DataFrame`
        dask dataframe with all information (normal payload and weights)
    dropnonkin : bool
        drop columns that are not kinematic information (e.g. ``OS`` or ``reg2j1b``)
    """

    def __init__(
        self, name: str, ddf: dd.DataFrame, dropnonkin: bool = True
    ) -> DataFramesInMemory:
        self.name = name
        all_columns = list(ddf.columns)
        categorized = categorize_branches(all_columns)
        nonweights = categorized["kin"]
        if not dropnonkin:
            nonweights += categorized["meta"]
        self._df = ddf[nonweights].compute()
        self._weights = ddf[categorized["weights"]].compute()

    @property
    def df(self):
        return self._df

    @property
    def weights(self):
        return self._weights

    def __repr__(self):
        return "DataFramesInMemory(name={}, df_shape={}, weights_shape={})".format(
            self.name, self.df.shape, self.weights.shape
        )


@dataclass
class SelectedDataFrame:
    """DataFrame constructed from a selection string

    Attributes
    ----------
    name : str
       shorthand name of the selection
    selection : str
       the selection string (in :py:func:`pandas.eval` form)
    df : :obj:`dask.dataframe.DataFrame`
       the dask DataFrame
    """

    name: str
    selection: str
    df: dd.DataFrame = field(repr=False, compare=False)

    def to_ram(self, **kwargs):
        """create a dataset that lives in memory

        kwargs are passed to the :obj:`DataFramesInMemory` constructor

        Examples
        --------
        >>> sdf = specific_dataframe(files, "2j2b", name="ttbar_2j2b")
        >>> dim = sdf.to_ram(dropnonkin=False)
        """
        return DataFramesInMemory(self.name, self.df, **kwargs)


def delayed_dataframe(
    files: Union[str, List[str]],
    tree: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    branches: Optional[List[str]] = None,
    repartition_kw: Optional[Dict[str, Any]] = None,
) -> dd.DataFrame:
    """Construct a dask flavored DataFrame from uproot's pandas utility

    Parameters
    ----------
    files : list(str) or str
       a single ROOT file or list of ROOT files
    tree : str
       the tree name to turn into a dataframe
    weight_name: str
       weight branch (we make sure to grab it if you give something
       other than ``None`` to ``branches``).
    branches : list(str), optional
       a list of branches to include as columns in the dataframe,
       default is ``None``, includes all branches.
    repartition_kw : dict(str, Any), optional
       arguments to pass to :py:func:`dask.dataframe.DataFrame.repartition`

    Returns
    -------
    :obj:`dask.dataframe.DataFrame`

    Examples
    --------
    >>> from glob import glob
    >>> files = glob("/path/to/files/*.root")
    >>> ddf = delayed_dataframe(files, branches=["branch_a", "branch_b"])

    """
    use_branches = branches
    if branches is not None:
        use_branches = list(set(branches) | set([weight_name]))

    @dask.delayed
    def get_frame(f, tn):
        t = uproot.open(f)[tn]
        return t.pandas.df(branches=use_branches)

    dfs = [get_frame(f, tree) for f in files]
    ddf = dd.from_delayed(dfs)
    if repartition_kw is not None:
        log.info(f"repartition with {repartition_kw}")
        ddf = ddf.repartition(**repartition_kw)
    return ddf


def selected_dataframes(
    files: Union[str, List[str]],
    selections: Dict[str, str],
    tree: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    branches: Optional[List[str]] = None,
    delayed_dataframe_kw: Optional[Dict[str, Any]] = None,
) -> Dict[str, SelectedDataFrame]:
    """Construct a set of dataframes based on a list of selection queries

    Parameters
    ----------
    files : list(str) or str
       a single ROOT file or list of ROOT files
    selections : dict(str,str)
       the list of selections to apply on the dataframe in the form
       ``(name, query)``.
    tree : str
       the tree name to turn into a dataframe
    weight_name: str
       weight branch
    branches : list(str), optional
       a list of branches to include as columns in the dataframe,
       default is ``None`` (all branches)
    delayed_dataframe_kw : dict(str, Any), optional
       set of arguments to pass to :py:func:`delayed_dataframe`

    Returns
    -------
    dict(str, :obj:`SelectedDataFrame`)
       dictionary containing queried dataframes.

    Examples
    --------
    >>> from glob import glob
    >>> files = glob("/path/to/files/*.root")
    >>> selections = {"r2j2b": "(reg2j2b == True) & (OS == True)",
    ...               "r2j1b": "(reg2j1b == True) & (OS == True)"}
    >>> frames = selected_dataframes(files, selections=selections)
    """
    df = delayed_dataframe(files, tree, weight_name, branches, **delayed_dataframe_kw)
    return {
        sel_name: SelectedDataFrame(sel_name, sel_query, df.query(sel_query))
        for sel_name, sel_query in selections.items()
    }


def specific_dataframe(
    files: Union[str, List[str]],
    region: Union[Region, str],
    name: str = "nameless",
    tree: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    extra_branches: List[str] = [],
    to_ram: bool = False,
    to_ram_kw: Dict[str, Any] = {},
) -> Union[SelectedDataFrame, DataFramesInMemory]:
    """Construct a set of dataframes based on a list of selection queries

    Parameters
    ----------
    files : list(str) or str
       a single ROOT file or list of ROOT files
    region : tdub.regions.Region or str
       which predefined tW region to select
    name : str
       give your selection a name
    tree : str
       the tree name to turn into a dataframe
    weight_name: str
       weight branch
    extra_branches : list(str), optional
       a list of additional branches to save (the standard branches
       associated as features for the region you selected will be
       included by default).
    to_ram : bool
       automatically send dataset to memory via :py:func:`SelectedDataFrame.to_ram`
    to_ram_kw: dict(str, Any)
       keywords to send to :py:func:`SelectedDataFrame.to_ram` function


    Returns
    -------
    :obj:`SelectedDataFrame` or :obj:`DataFramesInMemory`

    Examples
    --------
    >>> from glob import glob
    >>> files = glob("/path/to/files/*.root")
    >>> frame_2j1b = specific_dataframe(files, Region.r2j1b, extra_branches=["pT_lep1"])
    >>> frame_2j2b = specific_dataframe(files, "2j2b", extra_branches=["met"])

    """
    if isinstance(region, str):
        if region.startswith("r"):
            r = Region[region]
        else:
            r = Region[f"r{region}"]
    elif isinstance(region, Region):
        r = region
    else:
        raise TypeError("region argument must be tdub.regions.Region or str")
    if r == Region.r1j1b:
        branches = list(set(FSET_1j1b) | set(extra_branches) | {"reg1j1b", "OS"})
        q = SEL_1j1b
    elif r == Region.r2j1b:
        branches = list(set(FSET_2j1b) | set(extra_branches) | {"reg2j1b", "OS"})
        q = SEL_2j1b
    elif r == Region.r2j2b:
        branches = list(set(FSET_2j2b) | set(extra_branches) | {"reg2j2b", "OS"})
        q = SEL_2j2b
    elif r == Region.r3j:
        branches = list(set(FSET_3j) | set(extra_branches) | {"reg3j", "OS"})
        q = SEL_3j
    sdf = SelectedDataFrame(
        name, q, delayed_dataframe(files, tree, weight_name, branches).query(q)
    )
    if to_ram:
        return sdf.to_ram(**to_ram_kw)
    return sdf


def stdregion_dataframes(
    files: Union[str, List[str]],
    tree: str = "WtLoop_nominal",
    branches: Optional[List[str]] = None,
    partitioning: Optional[Union[int, str]] = None,
) -> Dict[str, SelectedDataFrame]:
    """Prepare our standard regions (selections) from a master dataframe

    This is just a call of :meth:`selected_dataframes` with hardcoded
    selections (using our standard regions): 1j1b, 2j1b, 2j2b, 3j.

    Parameters
    ----------
    files : list(str) or str
       a single ROOT file or list of ROOT files
    tree : str
       the tree name to turn into a dataframe
    branches : list(str), optional
       a list of branches to include as columns in the dataframe,
       default is ``None`` (all branches)
    partitioning : int or str, optional
       partion size for the dask dataframes

    Returns
    -------
    dict(str, :obj:`SelectedDataFrame`)
       dictionary containing queried dataframes.

    Examples
    --------
    >>> from glob import glob
    >>> files = glob("/path/to/files/*.root")
    >>> standard_regions = stdregion_dataframes(files)

    """

    selections = {"r1j1b": SEL_1j1b, "r2j1b": SEL_2j1b, "r2j2b": SEL_2j2b, "r3j": SEL_3j}
    use_branches = None
    if branches is not None:
        use_branches = list(
            set(branches) | set(["reg1j1b", "reg2j1b", "reg2j2b", "reg3j", "OS"])
        )
    repart_kw = None
    if isinstance(partitioning, str):
        repart_kw = {"partition_size": partitioning}
    elif isinstance(partitioning, int):
        repart_kw = {"npartitions": partitioning}
    return selected_dataframes(
        files,
        selections,
        tree,
        use_branches,
        delayed_dataframe_kw={"repartition_kw": repart_kw},
    )

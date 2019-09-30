"""
Module for handling dataframes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cachetools
import uproot
import dask
import dask.dataframe as dd
import pandas as pd

from tdub.utils import categorize_branches, conservative_branches
from tdub.regions import SELECTIONS, FEATURESETS
from tdub.regions import Region


log = logging.getLogger(__name__)


class DataFramesInMemory:
    """A dataset structured with everything living on RAM

    Parameters
    ----------
    name : str
        dataset name
    ddf : :obj:`dask.dataframe.DataFrame`
        dask dataframe with all information (normal payload and weights)
    dropnonkin : bool
        drop columns that are not kinematic information (e.g. ``OS`` or ``reg2j1b``)

    Attributes
    ----------
    name : str
       dataset name
    df : :obj:`pandas.DataFrame`
       payload dataframe, for meaningful kinematic features
    weights : :obj:`pandas.DataFrame`
       dataframe to hold weight information

    Examples
    --------
    Manually constructing in memory dataframes from a dask dataframe:

    >>> from tdub.frames import DataFramesInMemory, delayed_dataframe, quick_files
    >>> ttbar_files = quick_files("/path/to/data")["ttbar"]
    >>> branches = ["pT_lep1", "met", "mass_lep1jet1"]
    >>> ddf = delayed_dataframe(ttbar_files, branches=branches)
    >>> dfim = DataFramesInMemory("ttbar", ddf)

    Having the legwork done by other module features (see :py:func:`specific_dataframe`):

    >>> from tdub.frames import specific_dataframe
    >>> dfim = specific_dataframe(ttbar_files, "2j2b", to_ram=True)

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

    Notes
    -----
    This class is not designed for instance creation on the user end,
    we have factory functions for creating instances

    See Also
    --------
    specific_dataframe
    selected_dataframes
    stdregion_dataframes

    """

    name: str
    selection: str
    df: dd.DataFrame = field(repr=False, compare=False)

    def to_ram(self, **kwargs) -> DataFramesInMemory:
        """create a dataset that lives in memory

        kwargs are passed to the :obj:`DataFramesInMemory` constructor

        Returns
        -------
        :obj:`DataFramesInMemory`
           the wrapper around two pandas-backed DataFrames in memory

        Examples
        --------
        >>> from tdub.frames import specific_dataframe, quick_files
        >>> files = quick_files("/path/to/data")["ttbar"]
        >>> sdf = specific_dataframe(files, "2j2b", name="ttbar_2j2b")
        >>> dfim = sdf.to_ram(dropnonkin=False)
        """
        return DataFramesInMemory(self.name, self.df, **kwargs)


def raw_dataframe(
    files: Union[str, List[str]],
    tree: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    branches: Optional[List[str]] = None,
    entrysteps: Optional[Any] = None,
) -> pandas.DataFrame:
    """Construct a raw pandas flavored Dataframe with help from uproot

    We call this dataframe "raw" because it hasn't been parsed by any
    other tdub.frames functionality (no selection performed, kinematic
    and weight branches won't be separated, etc.) -- just a pure raw
    dataframe from some ROOT files.

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
    entrysteps : Any, optional
       see the ``entrysteps`` keyword for ``uproot.iterate``

    Returns
    -------
    :obj:`pandas.DataFrame`
       the pandas flavored DataFrame with all requested branches

    Examples
    --------

    >>> from tdub.frames import raw_dataframe
    >>> from tdub.utils import quick_files
    >>> files = quick_files("/path/to/files")["ttbar"]
    >>> df = raw_dataframe(files)

    """
    bs = branches
    if branches is not None:
        bs = sorted(set(branches) | set([weight_name]), key=str.lower)
    return pd.concat(
        [d for d in uproot.pandas.iterate(files, tree, branches=bs, entrysteps=entrysteps)]
    )


def conservative_dataframe(
    files: Union[str, List[str]],
    tree: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    entrysteps: Optional[Any] = None,
) -> pandas.DataFrame:
    """Construct a raw pandas flavored dataframe with conservative branches

    This function does some hand-holding and grabs a conservative set
    of branches from the input file(s). The branches that will be
    columns in the dataframe are determined by
    :py:func:`tdub.utils.conservative_branches`.

    Parameters
    ----------
    files : list(str) or str
       a single ROOT file or list of ROOT files
    tree : str
       the tree name to turn into a dataframe
    weight_name: str
       weight branch (we make sure to grab it)
    entrysteps : Any, optional
       see the ``entrysteps`` keyword for ``uproot.iterate``

    Returns
    -------
    :obj:`pandas.DataFrame`
       the pandas flavored DataFrame with all requested branches

    Examples
    --------

    >>> from tdub.frames import conservative_dataframe
    >>> from tdub.utils import quick_files
    >>> files = quick_files("/path/to/files")["ttbar"]
    >>> df = conservative_dataframe(files)

    """
    if isinstance(files, str):
        bs = conservative_branches(files)
    else:
        bs = conservative_branches(files[0])
    bs = list(set(bs) | set([weight_name]))
    return raw_dataframe(
        files, tree=tree, weight_name=weight_name, entrysteps=entrysteps, branches=bs
    )


def delayed_dataframe(
    files: Union[str, List[str]],
    tree: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    branches: Optional[List[str]] = None,
    repartition_kw: Optional[Dict[str, Any]] = None,
    experimental: bool = False,
) -> dask.dataframe.DataFrame:
    """Construct a dask flavored DataFrame with help from uproot

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
    experimental: bool
       if ``True`` try letting uproot create the Dask DataFrame instead of
       using ``dask.delayed`` on pandas DataFrames grabbed via uproot.

    Returns
    -------
    :obj:`dask.dataframe.DataFrame`
       the dask flavored DataFrame with all requested branches

    Examples
    --------

    >>> from tdub.frames import delayed_dataframe
    >>> from tdub.utils import quick_files
    >>> files = quick_files("/path/to/files")["tW_DR"]
    >>> ddf = delayed_dataframe(files, branches=["branch_a", "branch_b"])

    """
    bs = branches
    if branches is not None:
        bs = sorted(set(branches) | set([weight_name]), key=str.lower)

    # fmt: off
    if experimental:
        print("using experimental dataframe creation via uproot.daskframe")
        import cachetools
        cache = cachetools.LRUCache(1)
        ddf = uproot.daskframe(files, tree, bs, namedecode="utf-8", basketcache=cache)
    else:
        @dask.delayed
        def get_frame(f, tn):
            t = uproot.open(f)[tn]
            return t.pandas.df(branches=bs)
        if isinstance(files, str):
            dfs = [get_frame(files, tree)]
        else:
            dfs = [get_frame(f, tree) for f in files]
        ddf = dd.from_delayed(dfs)
    # fmt: on

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

    >>> from tdub.frames import selected_dataframes
    >>> from tdub.utils import quick_files
    >>> files = quick_files("/path/to/files")["tW_DS"]
    >>> selections = {"r2j2b": "(reg2j2b == True) & (OS == True)",
    ...               "r2j1b": "(reg2j1b == True) & (OS == True)"}
    >>> frames = selected_dataframes(files, selections=selections)

    """
    if delayed_dataframe_kw is None:
        df = delayed_dataframe(files, tree, weight_name, branches)
    else:
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
    extra_branches: Optional[List[str]] = None,
    to_ram: bool = False,
    to_ram_kw: Optional[Dict[str, Any]] = None,
) -> Union[SelectedDataFrame, DataFramesInMemory]:
    """Construct a dataframe based on specific predefined region selection

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
       if ``to_ram`` is ``False``, we return the dask-backed ``SelectedDataFrame``,
       if ``to_ram`` is ``True``, we return the pandas-backed ``DataFrameseInMemory``.

    Examples
    --------

    >>> from tdub.frames import specific_dataframe
    >>> from tdub.utils import quick_files
    >>> files = quick_files("/path/to/files")["ttbar"]
    >>> frame_2j1b = specific_dataframe(files, Region.r2j1b, extra_branches=["pT_lep1"])
    >>> frame_2j2b = specific_dataframe(files, "2j2b", extra_branches=["met"])

    """
    if isinstance(region, str):
        reg = Region.from_str(region)
    elif isinstance(region, Region):
        reg = region
    else:
        raise TypeError("region argument must be tdub.regions.Region or str")
    if extra_branches is None:
        extra_branches = []
    if reg == Region.r1j1b:
        branches = sorted(
            set(FEATURESETS[reg]) | set(extra_branches) | {"reg1j1b", "OS"}, key=str.lower
        )
    elif reg == Region.r2j1b:
        branches = sorted(
            set(FEATURESETS[reg]) | set(extra_branches) | {"reg2j1b", "OS"}, key=str.lower
        )
    elif reg == Region.r2j2b:
        branches = sorted(
            set(FEATURESETS[reg]) | set(extra_branches) | {"reg2j2b", "OS"}, key=str.lower
        )
    q = SELECTIONS[reg]
    sdf = SelectedDataFrame(
        name, q, delayed_dataframe(files, tree, weight_name, branches).query(q)
    )
    if to_ram:
        if to_ram_kw is None:
            return sdf.to_ram()
        else:
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
    selections (using our standard regions): 1j1b, 2j1b, 2j2b.

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
    >>> from tdub.frames import stdregion_dataframes
    >>> from tdub.utils import quick_files
    >>> files = quick_files("/path/to/files")["tW_DR"]
    >>> standard_regions = stdregion_dataframes(files)

    """

    use_branches = None
    if branches is not None:
        use_branches = sorted(
            set(branches) | set(["reg1j1b", "reg2j1b", "reg2j2b", "OS"]), key=str.lower
        )
    repart_kw = None
    if isinstance(partitioning, str):
        repart_kw = {"partition_size": partitioning}
    elif isinstance(partitioning, int):
        repart_kw = {"npartitions": partitioning}
    return selected_dataframes(
        files,
        SELECTIONS,
        tree,
        use_branches,
        delayed_dataframe_kw={"repartition_kw": repart_kw, "experimental": False},
    )

"""
Module for handling dataframes
"""

from __future__ import annotations

# stdlib
import logging
from dataclasses import dataclass, field

# externals
import cachetools
import dask
import dask.dataframe as dd
import pandas as pd
import uproot

# tdub
from tdub.utils import (
    AVOID_IN_CLF,
    FEATURESETS,
    SELECTIONS,
    Region,
    categorize_branches,
    conservative_branches,
    get_branches,
)


log = logging.getLogger(__name__)


class DataFramesInMemory:
    """A dataset structured with everything living on RAM

    Parameters
    ----------
    name : str
        dataset name
    ddf : :obj:`dask.dataframe.DataFrame`
        dask dataframe with all information (normal payload and
        weights)
    dropnonkin : bool
        drop columns that are not kinematic information (e.g. ``OS``
        or ``reg2j1b``)
    skeleton : bool
        initialize class with empty dataframes, ddf is ignored if ``True``.
    consolidate : bool
        call the ``consolidate`` function at the end of initialization

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

    >>> from tdub.utils import quick_files
    >>> from tdub.frames import DataFramesInMemory, delayed_dataframe
    >>> ttbar_files = quick_files("/path/to/data")["ttbar"]
    >>> branches = ["pT_lep1", "met", "mass_lep1jet1"]
    >>> ddf = delayed_dataframe(ttbar_files, branches=branches)
    >>> dfim = DataFramesInMemory(name="ttbar", ddf=ddf)

    Having the legwork done by other module features (see
    :py:func:`specific_dataframe`):

    >>> from tdub.frames import specific_dataframe
    >>> dfim = specific_dataframe(ttbar_files, "2j2b", to_ram=True)

    """

    def __init__(
        self,
        *,
        name: str = "",
        ddf: Optional[dask.dataframe.DataFrame] = None,
        skeleton: bool = False,
        dropnonkin: bool = False,
        consolidate: bool = False,
    ) -> DataFramesInMemory:
        if skeleton:
            self.name = name
            self._df = pd.DataFrame()
            self._weights = pd.DataFrame()
        else:
            if ddf is None:
                raise ValueError("if skeleton=False, a dask dataframe is required")
            self.name = name
            all_columns = list(ddf.columns)
            categorized = categorize_branches(all_columns)
            nonweights = categorized["kinematics"]
            if not dropnonkin:
                nonweights += categorized["meta"]
            self._df = ddf[sorted(nonweights)].compute()
            self._weights = ddf[categorized["weights"]].compute()
            if consolidate:
                self.consolidate()

    @property
    def df(self):
        return self._df

    @property
    def weights(self):
        return self._weights

    def __repr__(self):
        if self.weights is not None:
            return "DataFramesInMemory(name={}, df_shape={}, weights_shape={})".format(
                self.name, self.df.shape, self.weights.shape
            )
        else:
            return "DataFramesInMemory(name={}, df_shape={}, weights=None)".format(
                self.name, self.df.shape
            )

    def consolidate(self) -> None:
        """consolidate the data into a single dataframe

        this will remove the weights from the separate dataframe and
        move the data to the ``df`` attribute; the ``weights attribute
        will be reassigned to ``None``.
        """
        weight_cols = list(self.weights.columns)
        for weight_col in weight_cols:
            log.info(f"Moving {weight_col} to the main dataframe")
            self._df[weight_col] = self._weights.pop(weight_col)
        self._weights = None


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

    def to_ram(self, **dfim_opts) -> DataFramesInMemory:
        """create a dataset that lives in memory

        Parameters
        ----------
        dfim_opts
           options passed to the :py:obj:`DataFramesInMemory`
           initializer

        Returns
        -------
        :obj:`DataFramesInMemory`
           the dataset now living in memory

        Examples
        --------

        >>> from tdub.utils import quick_files
        >>> from tdub.frames import specific_dataframe
        >>> files = quick_files("/path/to/data")["ttbar"]
        >>> sdf = specific_dataframe(files, "2j2b", name="ttbar_2j2b")
        >>> dfim = sdf.to_ram(dropnonkin=False)

        """
        return DataFramesInMemory(name=self.name, ddf=self.df, **dfim_opts)


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

    >>> from tdub.utils import quick_files
    >>> from tdub.frames import raw_dataframe
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

    >>> from tdub.utils import quick_files
    >>> from tdub.frames import conservative_dataframe
    >>> files = quick_files("/path/to/files")["ttbar"]
    >>> df = conservative_dataframe(files)

    """
    if isinstance(files, str):
        bs = conservative_branches(files, tree)
    else:
        bs = conservative_branches(files[0], tree)
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

    >>> from tdub.utils import quick_files
    >>> from tdub.frames import delayed_dataframe
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

    >>> from tdub.utils import quick_files
    >>> from tdub.frames import selected_dataframes
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
    bypass_dask: bool = False,
    to_ram: bool = False,
    **kwargs,
) -> Union[SelectedDataFrame, DataFramesInMemory]:
    """Construct a dataframe based on specific predefined region selection

    Parameters
    ----------
    files : list(str) or str
       a single ROOT file or list of ROOT files
    region : tdub.utils.Region or str
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
    bypass_dask: bool
       bypass going through a delayed dataframe, just use pandas; this
       will cause the function to return a :obj:`DataFramesInMemory`
       instance (``to_ram`` will be ignored)
    to_ram : bool
       automatically send dataset to memory via
       :py:func:`SelectedDataFrame.to_ram`

    Keyword Args
    ------------
    consolidate : bool
       consolidate weight columns into the main DataFrame (usually
       they are seperate, see :py:obj:`DataFramesInMemory`)
    dropnonkin : bool
       drop non-kinematic columns (metadata like columns, like ``OS``,
       ``reg2j1b``, etc.).

    Returns
    -------
    :obj:`SelectedDataFrame` or :obj:`DataFramesInMemory`
       if ``to_ram`` is ``False``, we return the dask-backed ``SelectedDataFrame``,
       if ``to_ram`` is ``True``, we return the pandas-backed ``DataFramesInMemory``.
       if ``pybass_dask`` is ``True``, we return the pandas-backed ``DataFramesInMemory``.

    Examples
    --------

    >>> from tdub.utils import quick_files
    >>> from tdub.frames import specific_dataframe
    >>> files = quick_files("/path/to/files")["ttbar"]
    >>> frame_2j1b = specific_dataframe(files, Region.r2j1b, extra_branches=["pT_lep1"])
    >>> frame_2j2b = specific_dataframe(files, "2j2b", extra_branches=["met"])

    Here we bypass dask for converting the ROOT files to a dataframe
    (this is faster for smaller dataframes, for large dataframes
    (i.e. ttbar samples) we should stick to dask.

    >>> frame_2j2b = specific_dataframe(files, "2j2b", bypass_dask=True)

    Ask that the weight branches/columns are stored in the main
    ``df`` DataFrame (not in the ``weights`` DataFrame).

    >>> from tdub.frames import specific_dataframe
    >>> dfim = specific_dataframe(ttbar_files, "2j2b", to_ram=True, consolidate=True)
    >>> dfim.weights is None:
    True

    Or perform the consolidation independent of initialization:

    >>> from tdub.frames import specific_dataframe
    >>> dfim = specific_dataframe(ttbar_files, "2j2b", to_ram=True)
    >>> dfim.consolidate()

    """
    if isinstance(region, str):
        reg = Region.from_str(region)
    elif isinstance(region, Region):
        reg = region
    else:
        raise TypeError("region argument must be tdub.utils.Region or str")
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

    if bypass_dask:
        raw_df = raw_dataframe(files, tree, weight_name, branches=branches)
        weight_df = pd.DataFrame(raw_df.pop(weight_name))
        sel = raw_df.eval(q)
        dfim = DataFramesInMemory(skeleton=True)
        categorized = categorize_branches(raw_df)
        use_branches, meta_branches = categorized["kin"], categorized["meta"]
        if not kwargs.pop("dropnonkin", False):
            use_branches += meta_branches
        dfim._df = raw_df[sel][sorted(use_branches, key=str.lower)]
        dfim._weights = weight_df[sel]
        if kwargs.pop("consolidate", False):
            dfim.consolidate()
        return dfim

    sdf = SelectedDataFrame(
        name, q, delayed_dataframe(files, tree, weight_name, branches).query(q)
    )
    if to_ram:
        return sdf.to_ram(**kwargs)
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

    >>> from tdub.utils import quick_files
    >>> from tdub.frames import stdregion_dataframes
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


def satisfying_selection(*dfs: pandas.DataFrame, selection: str) -> List[pandas.DataFrame]:
    """get subsets of dataframes that satisfy a selection

    Parameters
    ----------
    *dfs : sequence of :py:obj:`pandas.DataFrame`
       the dataframes to apply the selection to
    selection : str
       the selection string (in :py:func:`pandas.eval` form)

    Returns
    -------
    list(pandas.DataFrame)
       the dataframes satisfying the selection string

    Examples
    --------

    >>> from tdub.utils import quick_files
    >>> from tdub.frames import specific_dataframe, satisfying_selection
    >>> qf = quick_files("/path/to/files")
    >>> dfim_tW_DR = specific_dataframe(qf["tW_DR"], to_ram=True)
    >>> dfim_ttbar = specific_dataframe(qf["ttbar"], to_ram=True)
    >>> low_bdt = "(bdt_response < 0.4)"
    >>> tW_DR_selected, ttbar_selected = satisfying_selection(dfim_tW_DR.df, dfim_ttbar.df,
    ...                                                       selection=low_bdt)

    """
    return [df.query(selection) for df in dfs]


def iterative_selection(
    files: Union[str, List[str]],
    selection: str,
    tree: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    branches: Optional[List[str]] = None,
    concat: bool = False,
    keep_category: Optional[str] = None,
    ignore_avoid: bool = False,
    **iterate_opts,
) -> pandas.DataFrame:
    """build a selected dataframe via uproot's iterate

    if we want to build a memory-hungry dataframe and apply a
    selection this helps us avoid crashing due to using all of our
    RAM.

    ``iterate_opts`` are fed to :py:func:`uproot.pandas.iterate`

    this dataframe construction function is useful when we want to
    grab all of the branches in a large dataset that won't fit in
    memory before the selection.

    Parameters
    ----------
    files : list(str) or str
       a single ROOT file or list of ROOT files
    selection : str
       the selection string (in :py:func:`pandas.eval` form)
    tree : str
       the tree name to turn into a dataframe
    weight_name: str
       weight branch to preserve
    branches : list(str), optional
       a list of branches to include as columns in the dataframe,
       default is ``None`` (all branches)
    concat : bool
       concatenate the resulting selected dataframes to return
       a single dataframe
    keep_category : str, optional
       if not ``None``, the selected dataframe(s) will only include
       columns which are part of the given category (see
       :py:func:`tdub.utils.categorize_branches`). The weight branch
       is always kept.
    ignore_avoid : bool
       ignore branches defined by :py:data:`tdub.utils.AVOID_IN_CLF`

    Returns
    -------
    list(pandas.DataFrame) or pandas.DataFrame
       the final selected dataframe(s) from the files

    Examples
    --------

    Creating a ``ttbar_dfs`` list of dataframes and a single ``tW_df``
    dataframe:

    >>> from tdub.frames import iterative_selection
    >>> from tdub.utils import quick_files
    >>> from tdub.utils import SELECTION_2j2b
    >>> qf = quick_files("/path/to/files")
    >>> ttbar_dfs = iterative_selection(qf["ttbar"], SELECTION_2j2b, entrysteps="1 GB")
    >>> tW_df = iterative_selection(qf["tW_DR"], SELECTION_2j2b, concat=True)

    Keep only the kinematic branches after selection and ignore avoided columns:

    >>> tW_df = iterative_selection(qf["tW_DR"], SELECTION_2j2b, concat=True,
    ...                             keep_category="kinematics", ignore_avoid=True)

    """
    if keep_category is not None:
        if isinstance(files, list):
            branch_cats = categorize_branches(files[0], tree=tree)
        else:
            branch_cats = categorize_branches(files, tree=tree)
        keep_branches = branch_cats.get(keep_category)
        if weight_name not in keep_branches:
            keep_branches.append(weight_name)

    bs = branches
    if branches is not None:
        bs = sorted(set(branches) | set([weight_name]), key=str.lower)

    if ignore_avoid:
        if bs is None:
            if isinstance(files, list):
                filebs = get_branches(files[0], tree=tree)
            else:
                filebs = get_branches(files, tree=tree)
            bs = sorted(set(filebs) - set(AVOID_IN_CLF), key=str.lower)
        else:
            bs = sorted(set(bs) - set(AVOID_IN_CLF), key=str.lower)
        if keep_category is not None:
            keep_branches = sorted(set(keep_branches) & set(bs), key=str.lower)

    dfs = []
    itr = uproot.pandas.iterate(files, tree, branches=bs, **iterate_opts)
    for i, df in enumerate(itr):
        idf = df.query(selection)
        if keep_category is not None:
            idf = idf[keep_branches]
        dfs.append(idf)
        log.debug(f"finished iteration {i}")
    if concat:
        return pd.concat(dfs)
    return dfs


def drop_cols(df: pandas.DataFrame, *cols: str) -> None:
    """drop some columns from a dataframe

    this is a convenient function because it just ignores branches
    that don't exist in the dataframe that are present in ``cols``.

    we augment :py:class:`pandas.DataFrame` with this function

    Parameters
    ----------
    df : :py:obj:`pandas.DataFrame`
       the df which we want to slim
    *cols : sequence of strings
       the columns to remove

    Examples
    --------

    >>> import pandas as pd
    >>> from tdub.utils import drop_cols
    >>> df = pd.read_parquet("some_file.parquet")
    >>> "E_jet1" in df.columns:
    True
    >>> "E_mass1" in df.columns:
    True
    >>> drop_cols(df, "E_jet1", "mass_jet1")
    >>> "E_jet1" in df.columns:
    False
    >>> "E_mass1" in df.columns:
    False

    """
    in_dataframe = set(df.columns)
    in_cols = set(cols)
    in_both = list(in_dataframe & in_cols)
    df.drop(columns=in_both, inplace=True)


pd.DataFrame.drop_cols = drop_cols


def drop_avoid(df: pandas.DataFrame) -> None:
    """drop columns that we avoid in classifiers

    this uses :py:func:`tdub.frames.drop_cols` with a predefined set
    of columns (:py:data:`tdub.utils.AVOID_IN_CLF`).

    we augment :py:class:`pandas.DataFrame` with this function

    Parameters
    ----------
    df : :py:obj:`pandas.DataFrame`
       the df which we want to slim

    Examples
    --------

    >>> from tdub.utils improt drop_avoid
    >>> import pandas as pd
    >>> df = pd.read_parquet("some_file.parquet")
    >>> "E_jetL1" in df.columns:
    True
    >>> drop_avoid(df)
    >>> "E_jetL1" in df.columns:
    False

    """
    drop_cols(df, *AVOID_IN_CLF)


pd.DataFrame.drop_avoid = drop_avoid


def drop_jet2(df: pandas.DataFrame) -> None:
    """drop all columns with jet2 properties

    in the 1j1b region we obviously don't have a second jet; so this
    lets us get rid of all columns dependent on jet2 kinematic
    properties.

    we augment :py:class:`pandas.DataFrame` with this function

    Parameters
    ----------
    df : :py:obj:`pandas.DataFrame`
       the df which we want to slim

    Examples
    --------

    >>> from tdub.utils improt drop_jet2
    >>> import pandas as pd
    >>> df = pd.read_parquet("some_file.parquet")
    >>> "pTsys_lep1lep2jet1jet2met" in df.columns:
    True
    >>> drop_jet2(df)
    >>> "pTsys_lep1lep2jet1jet2met" in df.columns:
    False

    """
    j2cols = [col for col in df.columns if "jet2" in col]
    drop_cols(df, *j2cols)


pd.DataFrame.drop_jet2 = drop_jet2

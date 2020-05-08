"""Module for handling dataframes."""

# stdlib
import logging
import re

from typing import Optional, Union, List, Iterable

# externals
import pandas as pd
import uproot

# tdub
from tdub.constants import AVOID_IN_CLF
from tdub.utils import Region, get_avoids
from tdub.branches import (
    categorize_branches,
    get_branches,
    minimal_branches,
    numexpr_selection,
)


log = logging.getLogger(__name__)


def raw_dataframe(
    files: Union[str, List[str]],
    tree: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    branches: Optional[Iterable[str]] = None,
    drop_weight_sys: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Construct a raw pandas flavored Dataframe with help from uproot.

    We call this dataframe "raw" because it hasn't been parsed by any
    other tdub.frames functionality (no selection performed, kinematic
    and weight branches won't be separated, etc.) -- just a pure raw
    dataframe from some ROOT files.

    Extra `kwargs` are fed to uproot.pandas.iterate

    Parameters
    ----------
    files : list(str) or str
        Single ROOT file or list of ROOT files.
    tree : str
        The tree name to turn into a dataframe.
    weight_name: str
        Weight branch (we make sure to grab it if you give something
        other than ``None`` to ``branches``).
    branches : list(str), optional
        List of branches to include as columns in the dataframe,
        default is ``None``, includes all branches.
    drop_weight_sys : bool
        Drop all weight systematics from the being grabbed.

    Returns
    -------
    pandas.DataFrame
        The pandas flavored DataFrame with all requested branches

    Examples
    --------
    >>> from tdub.utils import quick_files
    >>> from tdub.frames import raw_dataframe
    >>> files = quick_files("/path/to/files")["ttbar"]
    >>> df = raw_dataframe(files)
    """
    if branches is not None:
        branches = sorted(set(branches) | set([weight_name]), key=str.lower)
    else:
        branches = get_branches(files, tree)
    if weight_name not in branches:
        raise RuntimeError(f"{weight_name} not present in {tree}")
    if drop_weight_sys:
        weight_sys_re = re.compile(r"^weight_sys\w+")
        branches = sorted(
            set(branches) ^ set(filter(weight_sys_re.match, branches)), key=str.lower
        )
    itr = uproot.pandas.iterate(files, tree, branches=branches, **kwargs)
    result = pd.concat([d for d in itr])
    result.selection_used = None
    return result


def iterative_selection(
    files: Union[str, List[str]],
    selection: str,
    tree: str = "WtLoop_nominal",
    weight_name: str = "weight_nominal",
    branches: Optional[List[str]] = None,
    keep_category: Optional[str] = None,
    exclude_avoids: bool = False,
    use_campaign_weight: bool = False,
    use_tptrw: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Build a selected dataframe via uproot's iterate.

    If we want to build a memory-hungry dataframe and apply a
    selection this helps us avoid crashing due to using all of our
    RAM. Constructing a dataframe with this function is useful when we
    want to grab many branches in a large dataset that won't fit in
    memory before the selection.

    The selection can be in either numexpr or ROOT form, we ensure
    that a ROOT style selection is converted to numexpr for use with
    :py:func:`pandas.eval`.

    Parameters
    ----------
    files : list(str) or str
        A single ROOT file or list of ROOT files.
    selection : str
        Selection string (numexpr or ROOT form accepted).
    tree : str
        Tree name to turn into a dataframe.
    weight_name: str
        Weight branch to preserve.
    branches : list(str), optional
        List of branches to include as columns in the dataframe,
        default is ``None`` (all branches).
    keep_category : str, optional
        If not ``None``, the selected dataframe(s) will only include
        columns which are part of the given category (see
        :py:func:`tdub.branches.categorize_branches`). The weight branch
        is always kept.
    exclude_avoids : bool
        Exclude branches defined by :py:data:`tdub.utils.AVOID_IN_CLF`.
    use_campaign_weight : bool
        Multiply the nominal weight by the campaign weight. this is
        potentially necessary if the samples were prepared without the
        campaign weight included in the product which forms the nominal
        weight.
    use_tptrw : bool
        Apply the top pt reweighting factor.

    Returns
    -------
    pandas.DataFrame
        The final selected dataframe(s) from the files.

    Examples
    --------
    Creating a ``ttbar_df`` dataframe a single ``tW_df`` dataframe:

    >>> from tdub.frames import iterative_selection
    >>> from tdub.utils import quick_files
    >>> from tdub.utils import get_selection
    >>> qf = quick_files("/path/to/files")
    >>> ttbar_dfs = iterative_selection(qf["ttbar"], get_selection("2j2b"),
    ...                                 entrysteps="1 GB")
    >>> tW_df = iterative_selection(qf["tW_DR"], get_selection("2j2b"))

    Keep only kinematic branches after selection and ignore avoided columns:

    >>> tW_df = iterative_selection(qf["tW_DR"],
    ...                             get_selection("2j2b"),
    ...                             exclue_avoids=True,
    ...                             keep_category="kinematics")

    """
    # determine which branches will be used for selection only and
    # which branches we need for weights
    selection_branches = minimal_branches(selection)
    weights_to_grab = set([weight_name])
    if use_campaign_weight:
        weights_to_grab.add("weight_campaign")
        log.info("applying the campaign weight")
    if use_tptrw:
        weights_to_grab.add("weight_tptrw_tool")
        log.info("applying the top pt reweighting factor")
    if branches is None:
        branches = set(get_branches(files, tree=tree))
    branches = set(branches)
    sel_only_branches = selection_branches - branches

    # determine which branches to keep after reading dataframes and
    # are necessary during reading.
    if keep_category is not None:
        branches_cated = categorize_branches(list(branches), tree=tree)
        keep_cat = set(branches_cated.get(keep_category))
        keep = keep_cat & branches
        read_branches = keep | weights_to_grab | selection_branches
    else:
        keep = branches
        read_branches = branches | weights_to_grab | selection_branches

    # drop avoided classifier variables
    if exclude_avoids:
        keep = keep - set(AVOID_IN_CLF)

    # always drop selection only branches
    keep = keep - sel_only_branches

    # always keep the requested weight (enforce here just in
    # case). sort into a list and move on to dataframes
    keep.add(weight_name)
    keep = sorted(keep, key=str.lower)

    numexpr_sel = numexpr_selection(selection)

    dfs = []
    itr = uproot.pandas.iterate(files, tree, branches=list(read_branches), **kwargs)
    for i, df in enumerate(itr):
        if use_campaign_weight:
            apply_weight_campaign(df)
        if use_tptrw:
            apply_weight_tptrw(df)
        idf = df.query(numexpr_sel)
        idf = idf[keep]
        dfs.append(idf)
        log.debug(f"finished iteration {i}")
    result = pd.concat(dfs)
    result.selection_used = selection
    return result


def satisfying_selection(*dfs: pd.DataFrame, selection: str) -> List[pd.DataFrame]:
    """Get subsets of dataframes that satisfy a selection.

    The selection string can be in either ROOT or numexpr form (we
    ensure to convert ROOT to numexpr).

    Parameters
    ----------
    *dfs : sequence of :py:obj:`pandas.DataFrame`
        Dataframes to apply the selection to.
    selection : str
        Selection string (in numexpr or ROOT form).

    Returns
    -------
    list(pandas.DataFrame)
        Dataframes satisfying the selection string.

    Examples
    --------
    >>> from tdub.utils import quick_files
    >>> from tdub.frames import raw_dataframe, satisfying_selection
    >>> qf = quick_files("/path/to/files")
    >>> df_tW_DR = raw_dataframe(qf["tW_DR"])
    >>> df_ttbar = raw_dataframe(qf["ttbar"])
    >>> low_bdt = "(bdt_response < 0.4)"
    >>> tW_DR_selected, ttbar_selected = satisfying_selection(
    ...     dfim_tW_DR.df, dfim_ttbar.df, selection=low_bdt
    ... )

    """
    numexprsel = numexpr_selection(selection)
    return [df.query(numexprsel) for df in dfs]


def drop_cols(df: pd.DataFrame, *cols: str) -> None:
    """Drop some columns from a dataframe.

    This is a convenient function because it just ignores branches
    that don't exist in the dataframe that are present in ``cols``. We
    augment :py:class:`pandas.DataFrame` with this function

    Parameters
    ----------
    df : :py:obj:`pandas.DataFrame`
        Dataframe which we want to slim.
    *cols : sequence of strings
        Columns to remove

    Examples
    --------
    >>> import pandas as pd
    >>> from tdub.utils import drop_cols
    >>> df = pd.read_parquet("some_file.parquet")
    >>> "E_jet1" in df.columns:
    True
    >>> "mass_jet1" in df.columns:
    True
    >>> "mass_jet2" in df.columns:
    True
    >>> drop_cols(df, "E_jet1", "mass_jet1")
    >>> "E_jet1" in df.columns:
    False
    >>> "mass_jet1" in df.columns:
    False
    >>> df.drop_cols("mass_jet2") # use augmented df class
    >>> "mass_jet2" in df.columns:
    False

    """
    in_dataframe = set(df.columns)
    in_cols = set(cols)
    in_both = list(in_dataframe & in_cols)
    log.debug("Dropping columns:")
    for c in in_both:
        log.debug(f" - {c}")
    df.drop(columns=in_both, inplace=True)


def drop_avoid(df: pd.DataFrame, region: Optional[Union[str, Region]] = None) -> None:
    """Drop columns that we avoid in classifiers.

    Uses :py:func:`tdub.frames.drop_cols` with a predefined set of
    columns (:py:data:`tdub.utils.AVOID_IN_CLF`). We augment
    :py:class:`pandas.DataFrame` with this function.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that you want to slim.
    region : optional, str or tdub.utils.Region
        Region to augment the list of dropped columns (see the region
        specific AVOID constants in the constants module).

    Examples
    --------
    >>> from tdub.utils import drop_avoid
    >>> import pandas as pd
    >>> df = pd.read_parquet("some_file.parquet")
    >>> "E_jetL1" in df.columns:
    True
    >>> drop_avoid(df)
    >>> "E_jetL1" in df.columns:
    False

    """
    to_drop = AVOID_IN_CLF
    if region is not None:
        to_drop += get_avoids(region)
    drop_cols(df, *to_drop)


def drop_jet2(df: pd.DataFrame) -> None:
    """Drop all columns with jet2 properties.

    In the 1j1b region we obviously don't have a second jet; so this
    lets us get rid of all columns dependent on jet2 kinematic
    properties. We augment :py:class:`pandas.DataFrame` with this
    function.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that we want to slim.

    Examples
    --------
    >>> from tdub.utils import drop_jet2
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


def apply_weight(
    df: pd.DataFrame, weight_name: str, exclude: Optional[List[str]] = None
) -> None:
    """Apply (multiply) a weight to all other weights in the DataFrame.

    This will multiply the nominal weight and all systematic weights
    in the DataFrame by the ``weight_name`` column. We augment
    :py:class:`pandas.DataFrame` with this function.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataaframe to operate on.
    weight_name : str
        Column name to multiple all other weight columns by.
    exclude : list(str), optional
        List of columns ot exclude when determining the other weight
        columns to operate on.

    Examples
    --------
    >>> import tdub.frames
    >>> df = tdub.frames.raw_dataframe("/path/to/file.root")
    >>> df.apply_weight("weight_campaign")
    """
    sys_weight_cols = [c for c in df.columns if "weight_sys" in c]
    cols = ["weight_nominal"] + sys_weight_cols
    if exclude is not None:
        for entry in exclude:
            if entry in cols:
                cols.remove(entry)
    if weight_name in cols:
        log.warn(f"{weight_name} is in the columns list, dropping")
        cols.remove(weight_name)

    df.loc[:, cols] = df.loc[:, cols].multiply(df.loc[:, weight_name], axis="index")


def apply_weight_campaign(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> None:
    """Multiply nominal and systematic weights by the campaign weight.

    This is useful for samples that were produced without the campaign
    weight term already applied to all other weights. We augment
    :py:class:`pandas.DataFrame` with this function.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to operate on.
    exclude : list(str), optional
        List of columns to exclude when determining the other weight
        columns to operate on.

    Examples
    --------
    >>> import tdub.frames
    >>> df = tdub.frames.raw_dataframe("/path/to/file.root")
    >>> df.weight_nominal[5]
    0.003
    >>> df.weight_campaign[5]
    0.4
    >>> df.apply_weight_campaign()
    >>> df.weight_nominal[5]
    0.0012
    """
    apply_weight(df, "weight_campaign", exclude=exclude)


def apply_weight_tptrw(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> None:
    """Multiply nominal and systematic weights by the top pt reweight term.

    This is useful for samples that were produced without the top pt
    reweighting term already applied to all other weights. We augment
    :py:class:`pandas.DataFrame` with this function.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to operate on.
    exclude : list(str), optional
        List of columns to exclude when determining the other weight
        columns to operate on.

    Examples
    --------
    >>> import tdub.frames
    >>> df = tdub.frames.raw_dataframe("/path/to/file.root")
    >>> df.weight_nominal[5]
    0.002
    >>> df.weight_tptrw_tool[5]
    0.98
    >>> df.apply_weight_tptrw()
    >>> df.weight_nominal[5]
    0.00196
    """
    excludes = ["weight_sys_noreweight"]
    if exclude is not None:
        excludes += exclude
    apply_weight(df, "weight_tptrw_tool", exclude=excludes)


pd.DataFrame.drop_cols = drop_cols
pd.DataFrame.drop_avoid = drop_avoid
pd.DataFrame.drop_jet2 = drop_jet2
pd.DataFrame.apply_weight = apply_weight
pd.DataFrame.apply_weight_campaign = apply_weight_campaign
pd.DataFrame.apply_weight_tptrw = apply_weight_tptrw

"""
A module to aid working with histograms
"""

# stdlib
from typing import Tuple, Optional, Union, List, Dict, Any

# ext
import numpy as np
import pandas as pd
from uproot_methods.classes import TH1
from pygram11 import fix1dmw


class CustomTH1(TH1.Methods, list):
    """A TH1 like skeleton object"""

    pass


class CustomTAxis:
    """A TAxis like object"""

    def __init__(self, edges: np.ndarray) -> None:
        self._fNbins = len(edges) - 1
        self._fXmin = edges[0]
        self._fXmax = edges[-1]
        self._fXbins = edges.astype(np.float64)


def prepare_padded(
    content: np.ndarray, errors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare arrays for saving to ROOT histogram with over/underflow

    Parameters
    ----------
    content : :py:obj:`numpy.ndarray`
       the bin contents
    error : :py:obj:`numpy.ndarray`
       the error on the bin content (the square-root of the variances)

    Returns
    -------
    :py:obj:`numpy.ndarray`
       the padded content
    :py:obj:`numpy.ndarray`
       the padded sumw2
    """
    nbins = content.shape[0] + 2
    content_padded = np.empty(nbins, dtype=content.dtype)
    content_padded[1:-1] = content
    content_padded[0] = 0.0
    content_padded[-1] = 0.0
    sumw2_padded = np.empty(nbins, dtype=np.float64)
    sumw2_padded[1:-1] = errors ** 2
    sumw2_padded[0] = 0.0
    sumw2_padded[-1] = 0.0
    return content_padded, sumw2_padded


def arrays2th1(
    content: np.ndarray, error: np.ndarray, bins: np.ndarray, title: str = "none"
) -> CustomTH1:
    """Create a TH1-like object built from arrays

    Parameters
    ----------
    content : :py:obj:`numpy.ndarray`
       the bin contents
    error : :py:obj:`numpy.ndarray`
       the error on the bin content (the square-root of the variances)
    bins : :py:obj:`numpy.ndarray`
       the binning definition
    title : str
       title the histogram

    Returns
    -------
    :obj:`CustomTH1`
       the ROOT like histogram object
    """

    output = CustomTH1.__new__(CustomTH1)
    if content.dtype == np.float32:
        output._classname = "TH1F"
    elif content.dtype == np.float64:
        output._classname = "TH1D"
    output._fXaxis = CustomTAxis(bins)
    output._fEntries = content.sum()
    output._fTitle = title

    content_padded, output._fSumw2 = prepare_padded(content, error)
    output.extend(content_padded)

    return output


def df2th1(
    dfc: pd.DataFrame,
    dfe: pd.DataFrame,
    weight_col: Optional[Union[List[str], str]] = None,
) -> Union[CustomTH1, Dict[str, CustomTH1]]:
    """Create a TH1-like object built from a dataframe structure

    Parameters
    ----------
    dfc : pandas.DataFrame
       the dataframe holding the bin content
    dfe : pandas.DataFrame
       the dataframe holding the bin errors
    weight_name : str or list(str), optional
       name of the weight(s) (column(s) in the dataframe) to use. If
       ``None``, just ``weight_nominal`` is used. if "ALL", all
       weights are used.

    Returns
    -------
    :obj:`CustomTH1` or dict(str, :obj:`CustomTH1`)
       the ROOT like histogram object(s)

    """
    binning = np.linspace(dfc._xmin, dfc._xmax, dfc._nbins + 1)
    if weight_col is None:
        weight_col = "weight_nominal"
    if isinstance(weight_col, str):
        if weight_col == "ALL":
            res = {}
            for weight_name in dfc.columns:
                res[weight_name] = arrays2th1(
                    dfc[weight_name].to_numpy(),
                    dfe[weight_name].to_numpy(),
                    binning,
                    title=dfc._var_used,
                )
            return res
        else:
            return {
                weight_col: arrays2th1(
                    dfc[weight_col].to_numpy(),
                    dfe[weight_col].to_numpy(),
                    binning,
                    title=dfc._var_used,
                )
            }
    else:
        res = {}
        for weight_name in weight_col:
            res[weight_name] = arrays2th1(
                dfc[weight_name].to_numpy(),
                dfe[weight_name].to_numpy(),
                binning,
                title=dfc._var_used,
            )
        return res


def generate_from_df(
    df: pd.DataFrame,
    var: str,
    bins: int,
    range: Tuple[float, float],
    nominal_weight: bool = True,
    systematic_weights: bool = False,
) -> Any:
    """Generate histogram(s) from a dataframe

    Parameters
    ----------
    df : pandas.DataFrame
       the dataframe with our variable and weights of interest
    var : str
       the variable (name of distribution) we want to histogram
    bins : int
       the number of bins
    range : tuple(float, float)
       the axis limits (min, max) for the histogram
    nominal_weight : bool
       histogram the data using the nominal weight
    systematic_weights : bool
       histogram the data using the systematic weights in the dataframe

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame)
       the resulting histogram bin counts are in the first dataframe
       and the bin uncertainties are in the second frame. the columns
       give the weight used to calculate the histograms


    Examples
    --------

    >>> from tdub.utils import quick_files
    >>> from tdub.frames import raw_dataframe
    >>> from tdub.hist import generate_from_df
    >>> qf = quick_files("/path/to/data")
    >>> df_tW_DR = raw_dataframe(qf["tW_DR"])
    >>> hist_result = generate_from_df(
    ...     df_tW_DR,
    ...     "met",
    ...     bins=20,
    ...     range=(0.0, 200.0),
    ...     systematic_weights=True
    ... )

    """
    weight_cols: List[str] = []
    if nominal_weight:
        weight_cols += ["weight_nominal"]
    if systematic_weights:
        weight_cols += [c for c in df.columns if "weight_sys" in c]

    res = fix1dmw(df[var], df[weight_cols], bins=bins, range=range, flow=True, omp=True)
    res0 = pd.DataFrame(res[0], columns=weight_cols)
    res1 = pd.DataFrame(res[1], columns=weight_cols)
    res0._var_used = var
    res0._nbins = bins
    res0._xmin = range[0]
    res0._xmax = range[1]
    res1.var_used = var
    return (res0, res1)

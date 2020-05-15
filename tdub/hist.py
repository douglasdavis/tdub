"""A module to aid working with histograms."""

# stdlib
from typing import Tuple, Optional, Union, List, Dict, Any, Iterable

# ext
import numpy as np
import pandas as pd
from uproot_methods.classes import TH1
from pygram11 import fix1dmw


class CustomTH1(TH1.Methods, list):
    """A TH1 like skeleton object."""

    pass


class CustomTAxis:
    """A TAxis like object."""

    def __init__(self, edges: np.ndarray) -> None:
        self._fNbins = len(edges) - 1
        self._fXmin = edges[0]
        self._fXmax = edges[-1]
        self._fXbins = edges.astype(np.float64)


def prepare_padded(
    content: np.ndarray, errors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare arrays for saving to ROOT with over/underflow.

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


def arrays_to_th1(
    content: np.ndarray, error: np.ndarray, bins: np.ndarray, title: str = "none"
) -> CustomTH1:
    """Create a TH1-like object built from arrays.

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


def df_to_th1(
    dfc: pd.DataFrame,
    dfe: pd.DataFrame,
    weight_col: Optional[Union[List[str], str]] = None,
) -> Union[CustomTH1, Dict[str, CustomTH1]]:
    """Create a TH1-like object built from a dataframe structure.

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
                res[weight_name] = arrays_to_th1(
                    dfc[weight_name].to_numpy(),
                    dfe[weight_name].to_numpy(),
                    binning,
                    title=dfc._var_used,
                )
            return res
        else:
            return {
                weight_col: arrays_to_th1(
                    dfc[weight_col].to_numpy(),
                    dfe[weight_col].to_numpy(),
                    binning,
                    title=dfc._var_used,
                )
            }
    else:
        res = {}
        for weight_name in weight_col:
            res[weight_name] = arrays_to_th1(
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
    """Generate histogram(s) from a dataframe.

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
    >>> from tdub.data import quick_files
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


def bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """Get bin centers given bin edges.

    Parameters
    ----------
    bin_edges : numpy.ndarray
       edges defining binning

    Returns
    -------
    numpy.ndarray
       the centers associated with the edges

    Examples
    --------
    >>> import numpy as np
    >>> from tdub.hist import bin_centers
    >>> bin_edges = np.linspace(25, 225, 11)
    >>> centers = bin_centers(bin_edges)
    >>> bin_edges
    array([ 25.,  45.,  65.,  85., 105., 125., 145., 165., 185., 205., 225.])
    >>> centers
    array([ 35.,  55.,  75.,  95., 115., 135., 155., 175., 195., 215.])
    """
    return (bin_edges[1:] + bin_edges[:-1]) * 0.5


def to_uniform_bins(bin_edges: np.ndarray):
    """Convert a set of variable width bins to arbitrary uniform bins.

    This will create a set of bin edges such that the bin centers are
    at whole numbers, i.e. 5 variable width bins will return an array
    from 0.5 to 5.5: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5].

    Parameters
    ----------
    bin_edges : numpy.ndarray
        Array of bin edges.

    Returns
    -------
    numpy.ndarray
        The new set of uniform bins

    Examples
    --------
    >>> import numpy as np
    >>> from tdub.hist import to_uniform_bins
    >>> var_width = [0, 1, 3, 7, 15]
    >>> to_uniform_bins(var_width)
    array([0.5, 1.5, 2.5, 3.5, 4.5])
    """
    return np.arange(0.5, len(bin_edges) + 0.5, dtype=np.float64)


def edges_and_centers(
    bins: Union[int, Iterable], range: Optional[Tuple[float, float]] = None
) -> np.array:
    """Create arrays for edges and bin centers.

    Parameters
    ----------
    bins : int or sequence of scalers
       the number of bins or sequence representing bin edges
    range : tuple(float, float), optional
       the minimum and maximum defining the bin range (used if bins is integral)

    Returns
    -------
    :py:obj:`numpy.ndarray`
       the bin edges
    :py:obj:`numpy.ndarray`
       the bin centers

    Examples
    --------
    from bin multiplicity and a range

    >>> from tdub.hist import edges_and_centers
    >>> edges, centers = edges_and_centers(bins=20, range=(25, 225))

    from pre-existing edges

    >>> edges, centers = edges_and_centers(np.linspace(0, 10, 21))
    """
    if isinstance(bins, int):
        if range is None:
            raise ValueError("for integral bins we require the range argument")
        edges = np.linspace(range[0], range[1], bins + 1)
    else:
        edges = np.asarray(bins)
        if not np.all(edges[1:] >= edges[:-1]):
            raise ValueError("bins edges must monotonically increase")
    centers = bin_centers(edges)
    return edges, centers

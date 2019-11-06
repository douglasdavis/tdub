"""
A module to aid working with histograms
"""

from __future__ import annotations

# ext
import numba
import numba.types
import numpy as np
from uproot_methods.classes import TH1


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


@numba.njit(numba.types.UniTuple(numba.float64[:], 2)(numba.float64[:], numba.float64[:]))
def prepare_padded(
    content: numpy.ndarray, errors: numpy.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare arrays for saving to ROOT histogram with over/underflow

    This is accelerated with numba because it's 4x faster than
    non-jit'd version.

    Paramters
    ---------
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
    content_padded = np.empty(len(content) + 2, dtype=content.dtype)
    content_padded[1:-1] = content
    content_padded[0] = 0.0
    content_padded[-1] = 0.0
    sumw2_padded = np.empty(len(content) + 2, dtype=np.float64)
    sumw2_padded[1:-1] = error ** 2
    sumw2_padded[0] = 0.0
    sumw2_padded[-1] = 0.0
    return content_padded, sumw2_padded


def from_pygram11(
    content: numpy.ndarray, error: numpy.ndarray, bins: numpy.ndarray, title: str = "none"
) -> CustomTH1:
    """create a TH1-like object built from pygram11 output

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
       the ROOT like histogram structure
    """

    output = CustomTH1.__new__(CustomTH1)
    if content.dtype == np.float32:
        output._classname = "TH1F"
    elif content.dtype == np.float64:
        output._classname = "TH1D"
    output._fXaxis = CustomTAxis(bins)
    output._fEntries = content.sum()
    output._fTitle = name

    content_padded, output._fSumw2 = prepare_padded(content, error)
    output.extend(content_padded)

    return output

"""
Module for selecting features
"""

from __future__ import annotations

# stdlib
import gc
import logging
from pathlib import PosixPath

log = logging.getLogger(__name__)

# externals
import lightgbm as lgbm
import numpy as np
import pandas as pd

try:
    import pyarrow
except ImportError:
    pyarrow = None

# tdub
from tdub.frames import iterative_selection
from tdub.utils import quick_files, get_selection


def create_parquet_files(
    qf_dir: Union[str, os.PathLike],
    out_dir: Optional[Union[str, os.PathLike]] = None,
    entrysteps: Optional[Any] = None,
) -> None:
    """create slimmed and selected parquet files from ROOT files

    this function requires pyarrow_.

    .. _pyarrow: https://arrow.apache.org/docs/python/

    Parameters
    ----------
    qf_dir : str or os.PathLike
       directory to run :py:func:`tdub.utils.quick_files`
    out_dir : str or os.PathLike, optional
       directory to save output files
    entrysteps : any, optional
       entrysteps option forwarded to
       :py:func:`tdub.frames.iterative_selection`

    """
    if pyarrow is None:
        log.error("pyarrow required, doing nothing")
        return None
    indir = str(PosixPath(qf_dir).resolve())
    qf = quick_files(indir)
    if out_dir is None:
        out_dir = PosixPath(".")
    else:
        out_dir = PosixPath(out_dir)
    if entrysteps is None:
        entrysteps = "500 MB"

    for r in ("1j1b", "2j1b", "2j2b"):
        always_drop = ["eta_met", "bdt_response"]
        if r == "1j1b":
            always_drop.append("minimaxmbl")
        for sample in ("tW_DR", "tW_DS", "ttbar"):
            log.info(f"preparing to save a {sample} {r} parquet file using the files:")
            for f in qf[sample]:
                log.info(f" - {f}")
            df = iterative_selection(
                qf[sample],
                get_selection(r),
                keep_category="kinematics",
                concat=True,
                entrysteps=entrysteps,
            )
            df.drop_cols(*always_drop)
            df.drop_avoid()
            if r == "1j1b":
                df.drop_jet2()
            outname = str(out_dir / f"{sample}_{r}.parquet")
            df.to_parquet(outname, engine="pyarrow")
            log.info(f"{outname} saved")


class Selector:
    """A class to steer the steps of feature selection

    Parameters
    ----------
    data : pandas.DataFrame
       The dataframe which contains signal and background events; it
       should also only contain features we with to test for (it is
       expected to be "clean" from non-kinematic information, like
       metadata and weights).
    weights : numpy.ndarray
       the weights array compatible with the dataframe
    labels : numpy.ndarray
       array of labels compatible with the dataframe (``1`` for
       :math:`tW` and ``0`` for :math:`t\\bar{t}`.
    corr_threshold : float
       the threshold for excluding features based on correlations

    Attributes
    ----------
    data : pandas.DataFrame
       the raw dataframe as fed to the class instance
    weights : numpy.ndarray
       the raw weights array compatible with the dataframe
    labels : numpy.ndarray
       the raw labels array compatible with teh dataframe
    corr_threshold : float
       the threshold for excluding features based on correlations
    corr_matrix : pandas.DataFrame
       the correlation matrix for the features (requires calling the
       ``test_collinearity`` function)
    importances : pandas.DataFrame
       the importances as determined by a vanilla GBDT (requires
       calling the ``test_importances`` function)

    """

    def __init__(
        self,
        data: pandas.DataFrame,
        weights: numpy.ndarray,
        labels: np.ndarray,
        corr_threshold: float = 0.85,
    ) -> None:
        self._data = data
        self._weights = weights
        self._labels = labels
        self._raw_features = data.columns.to_list()
        self.corr_threshold = corr_threshold

        ## Calculated later by some member functions
        self._corr_matrix = None
        self._importances = None

    @property
    def data(self) -> pandas.DataFrame:
        return self._data

    @property
    def weights(self) -> numpy.ndarray:
        return self._weights

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def raw_features(self) -> List[str]:
        return self._raw_features

    def check_for_uniques(self, and_drop: bool = True) -> None:
        """check the dataframe for features that have a single unique value

        Parameters
        ----------
        and_drop : bool
           if ``True``, and_drop any unique columns

        """
        uqcounts = pd.DataFrame(self.data.nunique()).T
        to_drop = []
        for col in uqcounts.columns:
            if uqcounts[col].to_numpy()[0] == 1:
                to_drop.append(col)
        if not to_drop:
            log.info("we didn't find any features with single unique values")
        if to_drop and and_drop:
            for d in to_drop:
                log.info(f"dropping {d} because it's a feature with a single unique value")
            self._data.and_drop(columns=to_drop, inplace=True)

    def test_collinearity(self, threshold: Optional[float] = None) -> None:
        """calculate the correlations of the features

        given a correlation threshold this will construct a list of
        features that should be dropped based on the correlation
        values. This also adds a new property to the

        If the ``threshold`` argument is not None then the class
        instance's ``corr_threshold`` property is updated.

        Parameters
        ----------
        threshold : float, optional
           override the existing correlations threshold

        """
        if threshold is not None:
            self.corr_threshold = threshold

        self._corr_matrix = self.data.corr()

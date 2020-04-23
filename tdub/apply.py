"""
Module for applying trained models.
"""

# stdlib
import json
import logging
import os
from pathlib import PosixPath
from typing import List, Dict, Any, Union

# external
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator

# fmt: off
try:
    import lightgbm as lgbm
except ImportError:
    class lgbm:
        LGBMClassifier = None
try:
    import xgboost as xgb
except ImportError:
    class xgb:
        XGBClassifier = None
# fmt: on

# tdub
from tdub.utils import Region, FileLike


log = logging.getLogger(__name__)


class BaseResult:
    """Base class for describing a training result to apply to other data."""

    @property
    def features(self) -> List[str]:
        """The list of features used."""
        return self._features

    @property
    def region(self) -> Region:
        """The region where the training was executed."""
        return self._region

    @property
    def selection_used(self) -> str:
        """The selection used on the trained datasets."""
        return self._selection_used

    @property
    def summary(self) -> Dict[str, Any]:
        """The training summary dictionary from the training json."""
        return self._summary

    def parse_summary_json(self, summary_file: os.PathLike) -> None:
        """Parse a traning's summary json file.

        This populates the class properties with values and the
        resulting dictionary is saved to be accessible via the summary
        property. The common class properties (which all BaseResults
        have by defition) besides `summary are `features`, `region`,
        and `selecton_used`. This function will define those, so all
        BaseResult inheriting classes should call the super
        implementation of this method if a daughter implementation is
        necessary to add additional summary properties.

        Parameters
        ----------
        summary_file : str or os.PathLike
            the summary json file

        """
        self._summary = json.loads(summary_file.read_text())
        self._features = self.summary["features"]
        self._region = self.summary["region"]
        self._selection_used = self.summary["selection_used"]

    def apply_to_dataframe(
        self, df: pd.DataFrame, column_name: str, do_query: bool
    ) -> None:
        """Apply trained model(s) to events in a dataframe `df`.

        All BaseResult classes must implement this function.
        """
        raise NotImplementedError("This method must be implemented")


class FoldedResult(BaseResult):
    """Provides access to the properties of a folded training result.

    Parameters
    ----------
    fold_output : str
       the directory with the folded training output

    Examples
    --------
    >>> from tdub.apply import FoldedResult
    >>> fr_1j1b = FoldedResult("/path/to/folded_training_1j1b")

    """

    def __init__(self, fold_output: str) -> None:
        fold_path = PosixPath(fold_output).resolve()
        self._model0 = joblib.load(fold_path / "model_fold0.joblib.gz")
        self._model1 = joblib.load(fold_path / "model_fold1.joblib.gz")
        self._model2 = joblib.load(fold_path / "model_fold2.joblib.gz")
        self.parse_summary_json(fold_path / "summary.json")

    def parse_summary_json(self, summary_file: os.PathLike) -> None:
        """Parse a training's summary json file

        Parameters
        ----------
        summary_file : str or os.PathLike
            the summary json file
        """
        super().parse_summary_json(summary_file)
        self._folder = KFold(**(self.summary["kfold"]))

    @property
    def model0(self) -> BaseEstimator:
        """The model for the 0th fold"""
        return self._model0

    @property
    def model1(self) -> BaseEstimator:
        """The model for the 1st fold"""
        return self._model1

    @property
    def model2(self) -> BaseEstimator:
        """The model for the 2nd fold"""
        return self._model2

    @property
    def folder(self) -> KFold:
        """the folding object used during training"""
        return self._folder

    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        column_name: str = "unnamed_response",
        do_query: bool = False,
    ) -> None:
        """Apply trained models to an arbitrary dataframe.

        This function will augment the dataframe with a new column
        (with a name given by the ``column_name`` argument) if it
        doesn't already exist. If the dataframe is empty this function
        does nothing.

        Parameters
        ----------
        df : pandas.DataFrame
           the dataframe to read and augment
        column_name : str
           name to give the BDT response variable
        do_query : bool
           perform a query on the dataframe to select events belonging to
           the region associated with ``fr`` necessary if the dataframe
           hasn't been pre-filtered

        Examples
        --------
        >>> from tdub.apply import FoldedResult
        >>> from tdub.frames import raw_dataframe
        >>> df = raw_dataframe("/path/to/file.root")
        >>> fr_1j1b = FoldedResult("/path/to/folded_training_1j1b")
        >>> fr_1j1b.apply_to_dataframe(df, do_query=True)

        """
        if df.shape[0] == 0:
            log.info("Dataframe is empty, doing nothing")
            return None

        if column_name not in df.columns:
            log.info(f"Creating {column_name} column")
            df[column_name] = -9999.0

        if do_query:
            log.info(f"applying selection filter '{self.selection_used}'")
            mask = df.eval(self.selection_used)
            X = df[self.features].to_numpy()[mask]
        else:
            X = df[self.features].to_numpy()

        if X.shape[0] == 0:
            return None

        y0 = self.model0.predict_proba(X)[:, 1]
        y1 = self.model1.predict_proba(X)[:, 1]
        y2 = self.model2.predict_proba(X)[:, 1]
        y = np.mean([y0, y1, y2], axis=0)

        if do_query:
            df.loc[mask, column_name] = y
        else:
            df[column_name] = y


class SingleResult(BaseResult):
    """Provides access to the properties of a single result

    Parameters
    ----------
    training_output : str
        the directory containing the training result

    Examples
    --------
    >>> from tdub.apply import SingleResult
    >>> res_1j1b = SingleResult("/path/to/some_1j1b_training_outdir")

    """

    def __init__(self, training_output: os.PathLike) -> None:
        training_path = PosixPath(training_output)
        self._model = joblib.load(training_path / "model.joblib.gz")
        self.parse_summary_json(training_path / "summary.json")

    @property
    def model(self) -> BaseEstimator:
        """The trained model"""
        return self._model

    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        column_name: str = "unnamed_response",
        do_query: bool = False,
    ) -> None:
        """Apply trained model to an arbitrary dataframe.

        This function will augment the dataframe with a new column
        (with a name given by the ``column_name`` argument) if it
        doesn't already exist. If the dataframe is empty this function
        does nothing.

        Parameters
        ----------
        df : pandas.DataFrame
            the dataframe to read and augment
        column_name : str
            name to give the BDT response variable
        do_query : bool
            perform a query on the dataframe to select events belonging to
            the region associated with ``fr`` necessary if the dataframe
            hasn't been pre-filtered

        Examples
        --------
        >>> from tdub.apply import FoldedResult
        >>> from tdub.frames import raw_dataframe
        >>> df = raw_dataframe("/path/to/file.root")
        >>> sr_1j1b = SingleResult("/path/to/single_training_1j1b")
        >>> sr_1j1b.apply_to_dataframe(df, do_query=True)

        """

        if df.shape[0] == 0:
            log.info("Dataframe is empty, doing nothing")
            return None

        if column_name not in df.columns:
            log.info(f"Creating {column_name} column")
            df[column_name] = -9999.0

        if do_query:
            log.info(f"applying selection filter '{self.selection_used}'")
            mask = df.eval(self.selection_used)
            X = df[self.features].to_numpy()[mask]
        else:
            X = df[self.features].to_numpy()

        if X.shape[0] == 0:
            return None

        yhat = self.model.predict_proba(X)[:, 1]

        if do_query:
            df.loc[mask, column_name] = yhat
        else:
            df[column_name] = yhat


def build_array(results: List[BaseResult], df: pd.DataFrame) -> np.ndarray:
    """Get a NumPy array which is the response for all events in `df`

    This will use the :py:func:`~BaseResult.apply_to_dataframe` function
    from the list of results. We query the input dataframe to ensure
    that we apply to the correct events. If the input dataframe is
    empty then an empty array is written to disk

    Parameters
    ----------
    results : list(BaseResult)
       the training results to use
    df : pandas.DataFrame
       the dataframe of events to get the responses for

    Examples
    --------
    Using folded results:

    >>> from tdub.apply import FoldedResult, build_array
    >>> from tdub.frames import raw_dataframe
    >>> df = raw_dataframe("/path/to/file.root")
    >>> fr_1j1b = FoldedResult("/path/to/folded_training_1j1b")
    >>> fr_2j1b = FoldedResult("/path/to/folded_training_2j1b")
    >>> fr_2j2b = FoldedResult("/path/to/folded_training_2j2b")
    >>> res = build_array([fr_1j1b, fr_2j1b, fr_2j2b], df)

    Using single results:

    >>> from tdub.apply import SingleResult, build_array
    >>> from tdub.frames import raw_dataframe
    >>> df = raw_dataframe("/path/to/file.root")
    >>> sr_1j1b = SingleResult("/path/to/single_training_1j1b")
    >>> sr_2j1b = SingleResult("/path/to/single_training_2j1b")
    >>> sr_2j2b = SingleResult("/path/to/single_training_2j2b")
    >>> res = build_array([sr_1j1b, sr_2j1b, sr_2j2b], df)

    """

    if df.shape[0] == 0:
        log.info(f"Saving empty array to {output_file}")
        return np.array([], dtype=np.float64)

    colname = "_temp_col"
    log.info(f"The {colname} column will be deleted at the end of this function")
    for tr in results:
        tr.apply_to_dataframe(df, column_name=colname, do_query=True)
    return df.pop(colname).to_numpy()

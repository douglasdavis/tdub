"""
Module for applying trained models
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

Classifier = BaseEstimator

# tdub
from tdub.utils import Region


log = logging.getLogger(__name__)


class BaseResult:
    """Base class for encapsulating a BDT result to apply to other data.

    Attributes
    ----------
    features : list(str)
        the list of kinematic features used by the model
    region : Region
        the region for this training
    selection_used : str
        the selection that was used on the datasets used in training
    summary : dict(str, Any)
        the contents of the ``summary.json`` file.

    """

    _features: List[str] = []
    _region: Region = Region.rUnkn
    _selection_used: str = ""
    _summary: Dict[str, Any] = dict()

    @property
    def features(self) -> List[str]:
        return self._features

    @property
    def region(self) -> Region:
        return self._region

    @property
    def selection_used(self) -> str:
        return self._selection_used

    @property
    def summary(self) -> Dict[str, Any]:
        return self._summary

    def to_dataframe(
        self,
        df: pd.DataFrame,
        column_name: str = "unnamed_response",
        do_query: bool = False,
    ) -> None:
        """apply trained model(s) to an arbitrary dataframe.

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
            the region associated with result necessary (if the dataframe
            hasn't been pre-filtered)
        """
        raise NotImplementedError("Must be implemented in daughter class")


class FoldedResult(BaseResult):
    """Provides access to the properties of a folded training result

    Parameters
    ----------
    fold_output : str
       the directory with the folded training output

    Attributes
    ----------
    model0 : lightgbm.LGBMClassifier
       the model for the 0th fold from training
    model1 : lightgbm.LGBMClassifier
       the model for the 1st fold from training
    model2 : lightgbm.LGBMClassifier
       the model for the 2nd fold from training
    folder : sklearn.model_selection.KFold
       the folding object that the training session used

    Examples
    --------
    >>> from tdub.apply import FoldedResult
    >>> fr_1j1b = FoldedResult("/path/to/folded_training_1j1b")

    """

    def __init__(self, fold_output: str) -> None:
        fold_path = PosixPath(fold_output)
        if not fold_path.exists():
            raise ValueError(f"{fold_output} does not exist")
        fold_path = fold_path.resolve()
        self._model0 = joblib.load(fold_path / "model_fold0.joblib.gz")
        self._model1 = joblib.load(fold_path / "model_fold1.joblib.gz")
        self._model2 = joblib.load(fold_path / "model_fold2.joblib.gz")

        summary_file = fold_path / "summary.json"
        self._summary = json.loads(summary_file.read_text())
        self._features = self._summary["features"]
        self._folder = KFold(**(self._summary["kfold"]))
        self._region = Region.from_str(self._summary["region"])
        self._selection_used = self._summary["selection_used"]

    @property
    def model0(self) -> Classifier:
        return self._model0

    @property
    def model1(self) -> Classifier:
        return self._model1

    @property
    def model2(self) -> Classifier:
        return self._model2

    @property
    def folder(self) -> KFold:
        return self._folder

    def to_dataframe(
        self,
        df: pd.DataFrame,
        column_name: str = "unnamed_response",
        do_query: bool = False,
    ) -> None:
        """apply trained models to an arbitrary dataframe.

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
        >>> from tdub.frames import conservative_dataframe
        >>> df = conservative_dataframe("/path/to/file.root")
        >>> fr_1j1b = FoldedResult("/path/to/folded_training_1j1b")
        >>> fr_1j1b.to_dataframe(df, do_query=True)

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

    Attributes
    ----------
    model : lightgbm.LGBMClassifier
        the trained lightgbm model object

    Examples
    --------
    >>> from tdub.apply import SingleResult
    >>> res_1j1b = SingleResult("/path/to/some_1j1b_training_outdir")

    """

    def __init__(self, training_output: str):
        training_path = PosixPath(training_output)
        if not training_path.exists():
            raise ValueError(f"{training_output} does not exist")
        training_path = training_path.resolve()
        summary_file = training_path / "summary.json"
        self._features = self._summary["features"]
        self._region = Region.from_str(self._summary["region"])
        self._selection_used = self._summary["selection_used"]
        self._summary = json.loads(summary_file.read_text())

        self._model = joblib.load(training_path / "model.joblib.gz")

    @property
    def model(self) -> Classifier:
        return self._model

    def to_dataframe(
        self,
        df: pd.DataFrame,
        column_name: str = "unnamed_response",
        do_query: bool = False,
    ) -> None:
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


def generate_npy(
    results: List[BaseResult], df: pd.DataFrame, output_file: Union[str, os.PathLike]
) -> None:
    """create a NumPy npy file which is the response for all events in a DataFrame.

    This will use the to_dataframe function (see BaseResult docs) from
    the list of results. We query the input dataframe to ensure that
    we apply to the correct events. If the input dataframe is empty
    then an empty array is written to disk

    Parameters
    ----------
    results : list(BaseResult)
       the training results to use
    df : pandas.DataFrame
       the dataframe of events to get the responses for
    output_file : str or os.PathLike
       name of the output NumPy file

    Examples
    --------
    Using folded results:

    >>> from tdub.apply import FoldedResult, generate_npy
    >>> from tdub.frames import raw_dataframe
    >>> df = raw_dataframe("/path/to/file.root")
    >>> fr_1j1b = FoldedResult("/path/to/folded_training_1j1b")
    >>> fr_2j1b = FoldedResult("/path/to/folded_training_2j1b")
    >>> fr_2j2b = FoldedResult("/path/to/folded_training_2j2b")
    >>> generate_npy([fr_1j1b, fr_2j1b, fr_2j2b], df, "output.npy")

    Using single results:

    >>> from tdub.apply import SingleResult, generate_npy
    >>> from tdub.frames import raw_dataframe
    >>> df = raw_dataframe("/path/to/file.root")
    >>> sr_1j1b = SingleResult("/path/to/single_training_1j1b")
    >>> sr_2j1b = SingleResult("/path/to/single_training_2j1b")
    >>> sr_2j2b = SingleResult("/path/to/single_training_2j2b")
    >>> generate_npy([sr_1j1b, sr_2j1b, sr_2j2b], df, "output.npy")

    """

    if df.shape[0] == 0:
        log.info(f"Saving empty array to {output_file}")
        np.save(output_file, np.array([], dtype=np.float64))
        return None

    outfile = PosixPath(output_file)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    colname = "_temp_col"
    log.info(f"The {colname} column will be deleted at the end of this function")
    for tr in results:
        tr.to_dataframe(df, column_name=colname, do_query=True)
    np.save(outfile, df[colname].to_numpy())
    log.info(f"Saved output to {outfile}")
    df.drop(columns=[colname], inplace=True)
    log.info(f"Temporary column '{colname}' deleted")

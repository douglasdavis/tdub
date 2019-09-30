"""
Module for applying trained models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PosixPath
import json
import logging

import uproot
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold

from tdub.regions import Region, SELECTIONS
from tdub.frames import specific_dataframe

# fmt: off

try:
    import root_pandas
    _has_root_pandas = True
except ImportError:
    _has_root_pandas = False

# fmt: on

log = logging.getLogger(__name__)


class FoldedResult:
    """A class to hold the output from a folded training result

    Parameters
    ----------
    fold_output : str
       the directory with the folded training output
    region : Region or str
       the region where the training was performed

    Attributes
    ----------
    model0 : lightgbm.LGBMClassifier
       the model for the 0th fold from training
    model1 : lightgbm.LGBMClassifier
       the model for the 1st fold from training
    model2 : lightgbm.LGBMClassifier
       the model for the 2nd fold from training
    region : Region
       the region for this training
    features : list(str)
       the list of kinematic features used by the model
    folder : sklearn.model_selection.KFold
       the folding object that the training session used

    """

    def __init__(self, fold_output: str, region: Union[Region, str]) -> FoldedResult:
        fold_path = PosixPath(fold_output)
        if not fold_path.exists():
            raise ValueError(f"{fold_output} does not exit")
        fold_path = fold_path.resolve()
        self._model0 = joblib.load(fold_path / "model_fold0.joblib")
        self._model1 = joblib.load(fold_path / "model_fold1.joblib")
        self._model2 = joblib.load(fold_path / "model_fold2.joblib")

        if isinstance(region, str):
            if not region.startswith("r"):
                self._region = Region[f"r{region}"]
            else:
                self._region = Region[region]
        else:
            self._region = region

        feature_file = fold_path / "features.txt"
        self._features = feature_file.read_text().split("\n")[:-1]

        fold_file = fold_path / "kfold.json"
        self._folder = KFold(**(json.loads(fold_file.read_text())))

    @property
    def model0(self) -> lightgbm.LGBMClassifier:
        return self._model0

    @property
    def model1(self) -> lightgbm.LGBMClassifier:
        return self._model1

    @property
    def model2(self) -> lightgbm.LGBMClassifier:
        return self._model2

    @property
    def features(self) -> List[str]:
        return self._features

    @property
    def region(self) -> Region:
        return self._region

    @property
    def folder(self) -> KFold:
        return self._folder

    def to_files(
        self, files: Union[str, List[str]], tree: str = "WtLoop_nominal"
    ) -> numpy.ndarray:
        """apply the folded result to a set of files

        Parameters
        ----------
        files : str or list(str)
          the input file(s) to open and apply to
        tree : str
          the name of the tree to extract data from

        Returns
        -------
        numpy.ndarray
          the classifier output for the region associated with ``fr``

        """
        dfim = specific_dataframe(files, self.region, tree=tree, to_ram=True)
        dfim._df = dfim.df[self.features]

        X = dfim.df.to_numpy()
        y0 = self.model0.predict_proba(X)[:, 1]
        y1 = self.model1.predict_proba(X)[:, 1]
        y2 = self.model2.predict_proba(X)[:, 1]
        y = np.mean([y0, y1, y2], axis=0)

        return y

    def to_dataframe(
        self, df: pandas.DataFrame, column_name: str = "bdt_response", query: bool = False
    ) -> None:
        """apply trained models to an arbitrary dataframe.

        This function will augment the dataframe with a new column
        (with a name given by the ``column_name`` argument) if it
        doesn't already exist.

        Parameters
        ----------
        df : pandas.DataFrame
           the dataframe to read and augment
        column_name : str
           name to give the BDT response variable
        query : bool
           perform a query on the dataframe to select events belonging to
           the region associated with ``fr`` necessary if the dataframe
           hasn't been pre-filtered

        """

        if column_name not in df.columns:
            log.info(f"creating {column_name} column")
            df[column_name] = -9999.0

        if query:
            log.info(f"applying selection filter {SELECTIONS[self.region]}")
            mask = df.eval(SELECTIONS[self.region])
            X = df[self.features].to_numpy()[mask]
        else:
            X = df[self.features].to_numpy()

        y0 = self.model0.predict_proba(X)[:, 1]
        y1 = self.model1.predict_proba(X)[:, 1]
        y2 = self.model2.predict_proba(X)[:, 1]
        y = np.mean([y0, y1, y2], axis=0)

        if query:
            df.loc[mask, column_name] = y
        else:
            df[column_name] = y

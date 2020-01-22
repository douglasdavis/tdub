"""
Module for applying trained models
"""

from __future__ import annotations

# stdlib
import json
import logging
from pathlib import PosixPath

# external
import numpy as np
import joblib
from sklearn.model_selection import KFold

# tdub
from tdub.frames import specific_dataframe
from tdub.utils import Region, get_selection


log = logging.getLogger(__name__)


class FoldedResult:
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
    region : Region
       the region for this training
    features : list(str)
       the list of kinematic features used by the model
    folder : sklearn.model_selection.KFold
       the folding object that the training session used
    summary : dict(str, Any)
       the contents of the ``summary.json`` file.

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

    @property
    def summary(self) -> Dict[str, Any]:
        return self._summary

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

        Examples
        --------
        >>> from tdub.apply import FoldedResult
        >>> fr_1j1b = FoldedResult("/path/to/folded_training_1j1b")
        >>> y = fr_1j1b.to_files(["/path/to/file1.root", "/path/to/file2.root"])

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
        self,
        df: pandas.DataFrame,
        column_name: str = "unnamed_bdt_response",
        query: bool = False,
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
        query : bool
           perform a query on the dataframe to select events belonging to
           the region associated with ``fr`` necessary if the dataframe
           hasn't been pre-filtered

        Examples
        --------

        >>> from tdub.apply import FoldedResult
        >>> from tdub.frames import conservative_dataframe
        >>> df = conservative_dataframe("/path/to/file.root")
        >>> fr_1j1b = FoldedResult("/path/to/folded_training_1j1b")
        >>> fr_1j1b.to_dataframe(df, query=True)

        """
        if df.shape[0] == 0:
            log.info("Dataframe is empty, doing nothing")
            return None

        if column_name not in df.columns:
            log.info(f"Creating {column_name} column")
            df[column_name] = -9999.0

        if query:
            log.info(f"applying selection filter [ {get_selection(self.region)} ]")
            mask = df.eval(get_selection(self.region))
            X = df[self.features].to_numpy()[mask]
        else:
            X = df[self.features].to_numpy()

        if X.shape[0] == 0:
            return None

        y0 = self.model0.predict_proba(X)[:, 1]
        y1 = self.model1.predict_proba(X)[:, 1]
        y2 = self.model2.predict_proba(X)[:, 1]
        y = np.mean([y0, y1, y2], axis=0)

        if query:
            df.loc[mask, column_name] = y
        else:
            df[column_name] = y


def generate_npy(
    frs: List[FoldedResult], df: pandas.DataFrame, output_file: Union[str, os.PathLike]
) -> None:
    """create a NumPy npy file which is the response for all events in a DataFrame

    this will use all folds in the ``frs`` argument to get BDT
    response any each region associated to a ``FoldedResult``. We
    query the input df to ensure that we apply to the correct
    event. If the input dataframe is empty (no rows) then an empty
    array is written to disk.

    Parameters
    ----------
    frs : list(FoldedResult)
       the folded results to use
    df : pandas.DataFrame
       the dataframe of events to get the responses for
    output_file : str or os.PathLike
       name of the output NumPy file

    Examples
    --------

    >>> from tdub.apply import FoldedResult, generate_npy
    >>> from tdub.frames import conservative_dataframe
    >>> df = conservative_dataframe("/path/to/file.root")
    >>> fr_1j1b = FoldedResult("/path/to/folded_training_1j1b")
    >>> fr_2j1b = FoldedResult("/path/to/folded_training_2j1b")
    >>> fr_2j2b = FoldedResult("/path/to/folded_training_2j2b")
    >>> generate_npy([fr_1j1b, fr_2j1b, fr_2j2b], df, "output.npy")

    """

    if df.shape[0] == 0:
        log.info(f"Saving empty array to {output_file}")
        np.save(output_file, np.array([], dtype=np.float64))
        return None

    outfile = PosixPath(output_file)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    colname = "_temp_col"
    log.info(f"The {colname} column will be deleted at the end of this function")
    for fr in frs:
        fr.to_dataframe(df, column_name=colname, query=True)
    np.save(outfile, df[colname].to_numpy())
    log.info(f"Saved output to {outfile}")
    df.drop(columns=[colname], inplace=True)
    log.info(f"Temporary column '{colname}' deleted")

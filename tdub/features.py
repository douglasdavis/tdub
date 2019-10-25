"""
Module for selecting features
"""

from __future__ import annotations

# stdlib
import gc
import logging
from dataclasses import dataclass, field
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

    Examples
    --------

    >>> from tdub.features import create_parquet_files
    >>> create_parquet_files("/path/to/root/files", "/path/to/pq/output", entrysteps="250 MB")

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


def prepare_from_parquet(
    data_dir: Union[str, os.PathLike],
    region: Union[str, tdub.utils.Region],
    nlo_method: str = "DR",
    ttbar_frac: Union[str, float] = "auto",
    weight_scale: float = 1000.0,
) -> Tuple[pandas.DataFrame, np.ndarray, np.ndarray]:
    """prepare feature selection data from parquet files

    Parameters
    ----------
    data_dir : str or os.PathLike
       directory where the parquet files live
    region : str or tdub.utils.Region
       the region where we're going to select features
    nlo_method : str
       the :math:`tW` sample (``DR`` or ``DS``)
    ttbar_frac : str or float
       the fraction of :math:`t\\bar{t}` events to use, "auto" (the
       default) uses some sensible defaults to fit in memory: 0.70 for
       2j2b and 0.60 for 2j1b.
    weight_scale : float
       factor to scale sum of weights

    Returns
    -------
    pandas.DataFrame
       the dataframe which contains kinematic features
    numpy.ndarray
       the labels array for the events
    numpy.ndarray
       the weights array for the events

    Examples
    --------

    >>> from tdub.features import prepare_from_parquet
    >>> df, labels, weights = prepare_from_parquet("/path/to/pq/output", "2j1b", "DR")

    """
    data_path = PosixPath(data_dir)
    if not data_path.exists():
        raise RuntimeError(f"{data_dir} doesn't exist")
    sig_file = str(data_path / f"tW_{nlo_method}_{region}.parquet")
    bkg_file = str(data_path / f"ttbar_{region}.parquet")
    sig = pd.read_parquet(sig_file)
    bkg = pd.read_parquet(bkg_file)
    log.info(f"signal file loaded: {sig_file}")
    log.info(f"background file loaded {bkg_file}")

    for c in sig.columns.to_list():
        if c not in bkg.columns.to_list():
            log.warn(f"{c} not in bkg")
    for c in bkg.columns.to_list():
        if c not in sig.columns.to_list():
            log.warn(f"{c} not in sig")

    if ttbar_frac == "auto":
        if region == "2j2b":
            ttbar_frac = 0.70
        if region == "2j1b":
            ttbar_frac = 0.60
        if region == "1j1b":
            ttbar_frac = 1.00

    if ttbar_frac < 1:
        bkg = bkg.sample(frac=ttbar_frac, random_state=414)
        weight_scale = weight_scale / ttbar_frac

    sig_weights = sig.pop("weight_nominal").to_numpy() * weight_scale
    bkg_weights = bkg.pop("weight_nominal").to_numpy() * weight_scale
    sig_weights *= bkg_weights.sum() / sig_weights.sum()

    sig_labels = np.ones_like(sig_weights)
    bkg_labels = np.zeros_like(bkg_weights)

    df = pd.concat([sig, bkg])
    labels = np.concatenate([sig_labels, bkg_labels])
    weights = np.concatenate([sig_weights, bkg_weights])
    gc.enable()
    del sig, bkg, sig_labels, bkg_labels, sig_weights, bkg_weights
    gc.collect()

    return df, labels, weights


class FeatSelector:
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
    raw_features : list(str)
       the list of all features determined at initialization
    corr_threshold : float
       the threshold for excluding features based on correlations
    corr_matrix : pandas.DataFrame
       the raw correlation matrix for the features (requires calling the
       ``check_collinearity`` function)
    correlated : pandas.DataFrame
       a dataframe matching features that satisfy the correlation threshold
    importances : pandas.DataFrame
       the importances as determined by a vanilla GBDT (requires
       calling the ``check_importances`` function)

    Examples
    --------

    >>> from tdub.features import FeatSelector, prepare_from_parquet
    >>> df, labels, weights = prepare_from_parquet("/path/to/pq/output", "2j1b", "DR")
    >>> fs = FeatSelector(df=df, labels=labels, weights=weights, corr_threshold=0.90)

    """

    def __init__(
        self,
        df: pandas.DataFrame,
        labels: np.ndarray,
        weights: numpy.ndarray,
        corr_threshold: float = 0.85,
    ) -> None:
        assert np.unique(labels).shape[0] == 2, "labels should have 2 unique values"
        assert labels.shape == weights.shape, "labels and weights must have identical shape"
        assert corr_threshold < 1.0, "corr_threshold must be less than 1.0"
        assert (
            df.shape[0] == weights.shape[0]
        ), "df and weights must have the same number of entries"

        self._df = df
        self._weights = weights
        self._labels = labels
        self._raw_features = df.columns.to_list()
        self.corr_threshold = corr_threshold

        ## Calculated later by some member functions
        self._corr_matrix = None
        self._correlated = None
        self._importances = None

    @property
    def df(self) -> pandas.DataFrame:
        return self._df

    @property
    def weights(self) -> numpy.ndarray:
        return self._weights

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def raw_features(self) -> List[str]:
        return self._raw_features

    @property
    def corr_matrix(self) -> pandas.DataFrame:
        return self._corr_matrix

    @property
    def correlated(self) -> pandas.DataFrame:
        return self._correlated

    @property
    def importances(self) -> pandas.DataFrame:
        return self._importances

    def check_for_uniques(self, and_drop: bool = True) -> None:
        """check the dataframe for features that have a single unique value

        Parameters
        ----------
        and_drop : bool
           if ``True``, and_drop any unique columns

        Examples
        --------

        >>> fs.check_for_unique(and_drop=True)

        """
        uqcounts = pd.DataFrame(self.df.nunique()).T
        to_drop = []
        for col in uqcounts.columns:
            if uqcounts[col].to_numpy()[0] == 1:
                to_drop.append(col)
        if not to_drop:
            log.info("we didn't find any features with single unique values")
        if to_drop and and_drop:
            for d in to_drop:
                log.info(f"dropping {d} because it's a feature with a single unique value")
            self._df.and_drop(columns=to_drop, inplace=True)

    def check_collinearity(self, threshold: Optional[float] = None) -> None:
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

        Examples
        --------

        Overriding the exclusion threshold:

        >>> fs.corr_threshold
        0.90
        >>> fs.check_collinearity(threshold=0.85)
        >>> fs.corr_threshold
        0.85

        """
        log.info("checking correlations")

        if threshold is not None:
            self.corr_threshold = threshold

        self._corr_matrix = self.df.corr()

        uptri = np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        uptri = self._corr_matrix.where(uptri)
        dropcols = [c for c in uptri.columns if any(uptri[c].abs() > self.corr_threshold)]

        self._correlated = pd.DataFrame(columns=["drop", "because", "coeff"])
        for col in dropcols:
            other_features = list(uptri.index[uptri[col].abs() > self.corr_threshold])
            coeffs = list(uptri.index[uptri[col].abs() > self.corr_threshold])
            this_col = [col for i in range(len(other_features))]
            self._correlated.append(
                pd.DataFrame(dict(drop=this_col, because=other_features, coeff=coeffs))
            )

    def check_importances(
        self,
        extra_clf_opts: Optional[Dict[str, Any]] = None,
        extra_fit_opts: Optional[Dict[str, Any]] = None,
        n_fits: int = 5,
        random_state: int = 414,
    ) -> None:
        """train vanilla GBDT to calculate feature importance

        some default ``clf_dict`` and ``fit_dict`` are used (see
        implementation); you can provide extras via function some
        arguments.

        Parameters
        ----------
        extra_clf_opts : dict
           extra arguments forwarded to :py:class:`lightgbm.LGBMClassifier`.
        extra_fit_opts : dict
           extra arguments forwarded to :py:func:`lightgbm.LGBMClassifier.fit`.
        n_fits : int
           number of models to fit to determine importances
        test_size : float
           forwarded to :py:func:`sklearn.model_selection.train_test_split`
        random_state : int
           forwarded to :py:func:`sklearn.model_selection.train_test_split`

        Examples
        --------

        >>> fs.check_importances(extra_fit_opts=dict(verbose=40, early_stopping_round=15))

        """
        clf_opts = dict(
            boosting_type="gbdt",
            num_leaves=42,
            learning_rate=0.05,
            subsample_for_bin=180000,
            min_child_samples=40,
            reg_alpha=0.4,
            reg_lambda=0.5,
            colsample_bytree=0.8,
            n_estimators=250,
            max_depth=5,
        )
        if extra_clf_opts is not None:
            for k, v in extra_clf_opts.items():
                clf_ops[k] = v

        importance_counter = np.zeros(len(self.raw_features))

        log.info("starting training iterations")
        for i in range(n_fits):
            log.info(f"iteration {i}/{nfits}")
            model = LGBMClassifier(**clf_dict)
            train_df, test_df, train_y, test_y, train_w, test_w = train_test_split(
                self.df,
                self.labels,
                self.weights,
                shuffle=True,
                test_size=test_size,
                random_state=(i * random_state),
            )
            fit_opts = dict(
                eval_metric="auc",
                sample_weight=train_w,
                eval_set=[(test_df, test_y)],
                eval_sample_weight=[test_w],
                early_stopping_rounds=15,
                verbose=20,
            )
            if extra_fit_opts is not None:
                for k, v in extra_fit_opts:
                    fit_opts[k] = v
            model.fit(train_df, train_y, **fit_opts)
            importance_counter += model.feature_importances_
            gc.enable()
            del train_df, test_df, train_y, test_y, train_w, test_w
            gc.collect()

        self._importances = pd.DataFrame(
            dict(feature=self.raw_features, importance=(importance_counter / n_fits))
        )
        self._importances.sort_values("importance", ascending=False, inplace=True)
        self._importances.reset_index(inplace=True)

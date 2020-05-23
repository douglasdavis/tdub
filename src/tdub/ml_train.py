"""Module for training BDTs."""

# stdlib
import json
import logging
import os
from pathlib import PosixPath
from pprint import pformat
from typing import Optional, Tuple, List, Union, Dict, Any

# externals
import matplotlib

matplotlib.use("Agg")
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygram11
from scipy import interp
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import auc, roc_auc_score, roc_curve, plot_roc_curve
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

# fmt: off
try:
    import lightgbm as lgbm # noqa
except ImportError:
    class lgbm:
        LGBMClassifier = None
try:
    import xgboost as xgb # noqa
except ImportError:
    class xgb:
        XGBClassifier = None
# fmt: on

# tdub
from tdub.art import setup_tdub_style
from tdub.data import Region, features_for, quick_files, selection_for, selection_as_numexpr
from tdub.frames import iterative_selection, drop_cols
from tdub.math import ks_twosample_binned
from tdub.hist import bin_centers


setup_tdub_style()
log = logging.getLogger(__name__)

# _fig_adjustment_dict = dict(left=0.125, bottom=0.095, right=0.965, top=0.95)


def prepare_from_root(
    sig_files: List[str],
    bkg_files: List[str],
    region: Union[Region, str],
    branches: Optional[List[str]] = None,
    override_selection: Optional[str] = None,
    weight_mean: Optional[float] = None,
    weight_scale: Optional[float] = None,
    scale_sum_weights: bool = True,
    use_campaign_weight: bool = False,
    use_tptrw: bool = False,
    test_case_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare the data to train in a region with signal and background ROOT files.

    Parameters
    ----------
    sig_files : list(str)
        list of signal ROOT files
    bkg_files : list(str)
        list of background ROOT files
    region : Region or str
        the region where we're going to perform the training
    branches : list(str), optional
        if not None, we override the list of features defined by the region
    override_selection : str, optional
        a manual selection string to apply to the dataset (this will override
        the region defined selection).
    weight_mean : float, optional
        scale all weights such that the mean weight is this value. Cannot be
        used with `weight_scale`.
    weight_scale : float, optional
        value to scale all weights by, cannot be used with `weight_mean`.
    scale_sum_weights : bool
        scale sum of weights of signal to be sum of weights of background
    use_campaign_weight : bool
        see the parameter description for
        :py:func:`tdub.frames.iterative_selection`
    use_tptrw : bool
        apply the top pt reweighing factor.
    test_case_size : int, optional
        if defined, prepare a small test case dataset using this many background
        and training samples

    Returns
    -------
    pandas.DataFrame
       Event feature matrix
    numpy.ndarray
       Event labels (0 for background; 1 for signal)
    numpy.ndarray
       Event weights

    Examples
    --------
    >>> from tdub.data import quick_files
    >>> from tdub.train import prepare_from_root
    >>> qfiles = quick_files("/path/to/data")
    >>> df, labels, weights = prepare_from_root(qfiles["tW_DR"], qfiles["ttbar"], "2j2b")

    """
    if weight_scale is not None and weight_mean is not None:
        raise ValueError("weight_scale and weight_mean cannot be used together")

    log.info("Preparing a single dataset from ROOT files")
    log.info("Signal files:")
    for f in sig_files:
        log.info(" - %s" % f)
    log.info("Background files:")
    for f in bkg_files:
        log.info(" - %s" % f)

    if override_selection is not None:
        selection = override_selection
        log.info("Overriding selection (in region %s) to %s" % (region, override_selection))
    else:
        selection = selection_for(region)
    selection = selection_as_numexpr(selection)
    log.info("Total selection is: '%s'" % selection)

    if branches is None:
        log.info("Using features defined by the region")
        branches = features_for(region)

    sig_df = iterative_selection(
        files=sig_files,
        selection=selection,
        weight_name="weight_nominal",
        keep_category="kinematics",
        branches=branches,
        exclude_avoids=True,
        use_campaign_weight=use_campaign_weight,
    )
    bkg_df = iterative_selection(
        files=bkg_files,
        selection=selection,
        weight_name="weight_nominal",
        keep_category="kinematics",
        branches=branches,
        exclude_avoids=True,
        use_campaign_weight=use_campaign_weight,
        use_tptrw=use_tptrw,
        entrysteps="1 GB",
    )

    if test_case_size is not None:
        if test_case_size > 5000:
            log.warn("why bother with test_case_size > 5000?")
        sig_df = sig_df.sample(n=test_case_size, random_state=414)
        bkg_df = bkg_df.sample(n=test_case_size, random_state=414)

    w_sig = sig_df.pop("weight_nominal").to_numpy()
    w_bkg = bkg_df.pop("weight_nominal").to_numpy()
    w_sig[w_sig < 0] = 0.0
    w_bkg[w_bkg < 0] = 0.0
    if scale_sum_weights:
        w_sig *= w_bkg.sum() / w_sig.sum()
    if "weight_campaign" in sig_df:
        drop_cols(sig_df, "weight_campaign")
    if "weight_campaign" in bkg_df:
        drop_cols(bkg_df, "weight_campaign")

    sorted_cols = sorted(sig_df.columns.to_list(), key=str.lower)
    sig_df = sig_df[sorted_cols]
    bkg_df = bkg_df[sorted_cols]

    cols = sig_df.columns.to_list()
    assert cols == bkg_df.columns.to_list(), "sig/bkg columns are different. bad."
    log.info("Features in prepared dataset:")
    for c in cols:
        log.info(" - %s" % c)

    df = pd.concat([sig_df, bkg_df])
    y = np.concatenate([np.ones_like(w_sig), np.zeros_like(w_bkg)])
    w = np.concatenate([w_sig, w_bkg])

    if weight_scale is not None:
        w *= weight_scale
    if weight_mean is not None:
        w *= weight_mean * len(w) / np.sum(w)

    df.selection_used = selection
    return df, y, w


class SingleTrainingResult:
    """Describes the properties of a single training.

    Parameters
    ----------
    auc : float
        the AUC value for the model
    ks_test_sig : float
        the binned KS test value for signal
    ks_pvalue_sig : float
        the binned KS test p-value for signal
    ks_test_bkg : float
        the binned KS test value for background
    ks_pvalue_bkg : float
        the binned KS test p-value for background
    kwargs : dict
        currently unused

    Attributes
    ----------
    auc : float
        the AUC value for the model
    ks_test_sig : float
        the binned KS test value for signal
    ks_pvalue_sig : float
        the binned KS test p-value for signal
    ks_test_bkg : float
        the binned KS test value for background
    ks_pvalue_bkg : float
        the binned KS test p-value for background
    """

    def __init__(
        self,
        *,
        auc: float = -1.0,
        ks_test_sig: float = -1.0,
        ks_pvalue_sig: float = -1.0,
        ks_test_bkg: float = -1.0,
        ks_pvalue_bkg: float = -1.0,
        **kwargs,
    ) -> None:
        """Class init."""
        self.auc = auc
        self.ks_test_sig = ks_test_sig
        self.ks_pvalue_sig = ks_pvalue_sig
        self.ks_test_bkg = ks_test_bkg
        self.ks_pvalue_bkg = ks_pvalue_bkg
        self.bad_ks = self.ks_pvalue_sig < 0.2 or self.ks_pvalue_bkg < 0.2

    def __repr__(self) -> str:
        """Clean representation of the result."""
        p1 = f"auc={self.auc:0.3}"
        p2 = f"ks_test_sig={self.ks_test_sig:0.5}"
        p3 = f"ks_pvalue_sig={self.ks_pvalue_sig:0.5}"
        p4 = f"ks_test_bkg={self.ks_test_bkg:0.5}"
        p5 = f"ks_pvalue_bkg={self.ks_pvalue_bkg:0.5}"
        return f"SingleTrainingResult({p1}, {p2}, {p3}, {p4}, {p5})"


def single_training(
    df: pd.DataFrame,
    labels: np.ndarray,
    weights: np.ndarray,
    clf_params: Dict[str, Any],
    output_dir: Union[str, os.PathLike],
    test_size: float = 0.33,
    random_state: int = 414,
    early_stopping_rounds: Optional[int] = None,
    extra_summary_entries: Optional[Dict[str, Any]] = None,
    use_xgboost: bool = False,
    use_sklearn: bool = False,
) -> SingleTrainingResult:
    """Execute a single training with some parameters.

    The model and some useful information (mostly plots) are saved to
    `output_dir`.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature matrix in dataframe format
    labels : numpy.ndarray
        Event labels (`1` for signal; `0` for background)
    weights : numpy.ndarray
        Event weights
    clf_params : dict
        Dictionary of parameters to pass to
        :py:obj:`lightgbm.LGBMClassifier`.
    output_dir : str or os.PathLike
        Directory to save results of training
    test_size : float
        Test size for splitting into training and testing sets
    random_state : int
        Random seed for
        :py:func:`sklearn.model_selection.train_test_split`.
    early_stopping_rounds : int, optional
        Number of rounds to have no improvement for stopping training.
    extra_summary_entries : dict, optional
        Extra entries to save in the JSON output summary.
    use_xgboost : bool
        Use XGBoost classifier.
    use_sklearn : bool
        Use Scikit-learn's HistGradientBoostingClassifier.

    Returns
    -------
    SingleTrainingResult
        Useful information about the training

    Examples
    --------
    >>> from tdub.data import quick_files
    >>> from tdub.train import prepare_from_root, single_round
    >>> qfiles = quick_files("/path/to/data")
    >>> df, labels, weights = prepare_from_root(qfiles["tW_DR"], qfiles["ttbar"], "2j2b")
    >>> params = dict(
    ...     learning_rate=0.05
    ...     max_depth=5,
    ... )
    >>> single_round(
    ...     df,
    ...     labels,
    ...     weights,
    ...     params,
    ...     "training_output",
    ... )

    """
    starting_dir = os.getcwd()
    output_path = PosixPath(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    log.info("Saving training output to %s" % output_path.resolve())
    os.chdir(output_path)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        df, labels, weights, test_size=test_size, random_state=random_state, shuffle=True
    )

    log.info(f"test size used: {test_size}")
    log.info("selection used on the datasets:")
    log.info("  '%s'" % df.selection_used)
    log.info("features training with:")
    for c in X_train.columns:
        log.info(" - %s" % c)

    validation_data = [(X_test, y_test)]
    validation_w = w_test

    if use_xgboost:
        model = xgb.XGBClassifier(booster="gbtree", verbosity=1, **clf_params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=validation_data,
            eval_metric="auc",
            early_stopping_rounds=early_stopping_rounds,
            sample_weight_eval_set=[validation_w],
        )

    elif use_sklearn:
        model = HistGradientBoostingClassifier(
            learning_rate=clf_params.get("learning_rate", 0.1),
            max_depth=clf_params.get("max_depth", 5),
            min_samples_leaf=clf_params.get("min_child_samples", 50),
            max_leaf_nodes=clf_params.get("num_leaves", 31),
            validation_fraction=0.33,
            early_stopping=True,
            n_iter_no_change=10,
            max_iter=250,
            verbose=1,
        )
        model.fit(X_train, y_train, sample_weight=w_train)

    else:
        model = lgbm.LGBMClassifier(boosting_type="gbdt", **clf_params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=validation_data,
            eval_metric="auc",
            verbose=20,
            early_stopping_rounds=early_stopping_rounds,
            eval_sample_weight=[validation_w],
        )

    joblib.dump(model, "model.joblib.gz", compress=("gzip", 3))

    fig_proba, ax_proba = plt.subplots()
    fig_pred, ax_pred = plt.subplots()
    fig_roc, ax_roc = plt.subplots()
    fig_imp, (ax_imp_gain, ax_imp_split) = plt.subplots(2, 1)

    trainres = _inspect_single_training(
        ax_proba,
        ax_pred,
        ax_roc,
        model,
        X_test,
        X_train,
        y_test,
        y_train,
        w_test,
        w_train,
        is_xgb_model=use_xgboost,
        is_skl_model=use_sklearn,
    )

    if not use_xgboost and not use_sklearn:
        lgbm.plot_importance(model, ax=ax_imp_gain, importance_type="gain")
        lgbm.plot_importance(model, ax=ax_imp_split, importance_type="split")
        ax_imp_gain.set_xlabel("Importance (gain)")
        ax_imp_split.set_xlabel("Importance (split)")
        ax_imp_gain.set_title("")
        ax_imp_split.set_title("")
        fig_imp.subplots_adjust(left=0.475, top=0.975, bottom=0.09, right=0.925)
        fig_imp.savefig("imp.pdf")

    # fig_proba.subplots_adjust(**_fig_adjustment_dict)
    # fig_pred.subplots_adjust(**_fig_adjustment_dict)
    # fig_roc.subplots_adjust(**_fig_adjustment_dict)
    fig_proba.savefig("proba.pdf")
    fig_pred.savefig("pred.pdf")
    fig_roc.savefig("roc.pdf")

    summary = {"auc": round(trainres.auc, 5)}
    summary["selection_used"] = df.selection_used
    summary["bad_ks"] = trainres.bad_ks
    summary["ks_test_sig"] = trainres.ks_test_sig
    summary["ks_test_bkg"] = trainres.ks_test_bkg
    summary["ks_pvalue_sig"] = trainres.ks_pvalue_sig
    summary["ks_pvalue_bkg"] = trainres.ks_pvalue_bkg
    summary["features"] = [c for c in df.columns]
    summary["set_params"] = clf_params
    summary["all_params"] = model.get_params()
    summary["best_iteration"] = -1
    if early_stopping_rounds is not None:
        if use_xgboost:
            summary["best_iteration"] = int(model.best_iteration)
        elif use_sklearn:
            summary["best_iteration"] = int(model.n_iter_)
        else:
            summary["best_iteration"] = int(model.best_iteration_)
    if extra_summary_entries is not None:
        for k, v in extra_summary_entries.items():
            summary[k] = v
    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    os.chdir(starting_dir)

    return trainres


def _inspect_single_training(
    ax_proba: plt.Axes,
    ax_pred: plt.Axes,
    ax_roc: plt.Axes,
    model: Any,
    X_test: pd.DataFrame,
    X_train: np.ndarray,
    y_test: np.ndarray,
    y_train: np.ndarray,
    w_test: np.ndarray,
    w_train: np.ndarray,
    nbins_proba: int = 30,
    nbins_pred: int = 30,
    is_xgb_model: bool = False,
    is_skl_model: bool = False,
) -> SingleTrainingResult:
    """Inspect a single training round and make some plots."""
    # fmt: off
    # get the selection arrays
    test_is_sig = y_test == 1
    test_is_bkg = np.invert(test_is_sig)
    train_is_sig = y_train == 1
    train_is_bkg = np.invert(train_is_sig)

    # test and train output
    if is_xgb_model:
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = model.predict(X_test, output_margin=True)
        train_proba = model.predict_proba(X_train)[:, 1]
        train_pred = model.predict(X_train, output_margin=True)
    elif is_skl_model:
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = model.decision_function(X_test)
        train_proba = model.predict_proba(X_train)[:, 1]
        train_pred = model.decision_function(X_train)
    else:
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = model.predict(X_test, raw_score=True)
        train_proba = model.predict_proba(X_train)[:, 1]
        train_pred = model.predict(X_train, raw_score=True)

    # test and train weights
    test_w_sig = w_test[test_is_sig]
    test_w_bkg = w_test[test_is_bkg]
    train_w_sig = w_train[train_is_sig]
    train_w_bkg = w_train[train_is_bkg]

    # test and train signal and background proba arrays
    test_proba_sig = test_proba[test_is_sig]
    test_proba_bkg = test_proba[test_is_bkg]
    train_proba_sig = train_proba[train_is_sig]
    train_proba_bkg = train_proba[train_is_bkg]

    # test and train signal and background pred arrays
    test_pred_sig = test_pred[test_is_sig]
    test_pred_bkg = test_pred[test_is_bkg]
    train_pred_sig = train_pred[train_is_sig]
    train_pred_bkg = train_pred[train_is_bkg]

    # bins for proba and for pred
    proba_bins = np.linspace(0, 1, nbins_proba + 1)
    proba_bc = bin_centers(proba_bins)
    pred_xmin = min(test_pred_bkg.min(), train_pred_bkg.min())
    pred_xmax = max(test_pred_sig.max(), train_pred_sig.max())
    pred_bins = np.linspace(pred_xmin, pred_xmax, nbins_pred + 1)
    pred_bc = bin_centers(pred_bins)

    # calculate the proba histograms
    test_h_proba_sig = pygram11.histogram(test_proba_sig, bins=proba_bins, density=True, weights=test_w_sig)
    test_h_proba_bkg = pygram11.histogram(test_proba_bkg, bins=proba_bins, density=True, weights=test_w_bkg)
    train_h_proba_sig = pygram11.histogram(train_proba_sig, bins=proba_bins, density=True, weights=train_w_sig)
    train_h_proba_bkg = pygram11.histogram(train_proba_bkg, bins=proba_bins, density=True, weights=train_w_bkg)

    # calculate the pred histograms
    test_h_pred_sig = pygram11.histogram(test_pred_sig, bins=pred_bins, density=True, weights=test_w_sig)
    test_h_pred_bkg = pygram11.histogram(test_pred_bkg, bins=pred_bins, density=True, weights=test_w_bkg)
    train_h_pred_sig = pygram11.histogram(train_pred_sig, bins=pred_bins, density=True, weights=train_w_sig)
    train_h_pred_bkg = pygram11.histogram(train_pred_bkg, bins=pred_bins, density=True, weights=train_w_bkg)

    # plot the proba distributions
    ax_proba.hist(proba_bc, bins=proba_bins, weights=train_h_proba_sig[0],
                  label="Sig (train)", histtype="stepfilled", alpha=0.5, edgecolor="C0", color="C0")
    ax_proba.hist(proba_bc, bins=proba_bins, weights=train_h_proba_bkg[0],
                  label="Bkg (train)", histtype="stepfilled", alpha=0.4, edgecolor="C3", color="C3")
    ax_proba.errorbar(proba_bc, test_h_proba_sig[0], yerr=test_h_proba_sig[1],
                      label="Sig (test)", color="C0", fmt="o", markersize=4)
    ax_proba.errorbar(proba_bc, test_h_proba_bkg[0], yerr=test_h_proba_bkg[1],
                      label="Bkg (test)", color="C3", fmt="o", markersize=4)
    ax_proba.set_ylim([0, 1.4 * ax_proba.get_ylim()[1]])
    ax_proba.legend(loc="upper right", ncol=2, frameon=False, numpoints=1)
    ax_proba.set_ylabel("Arbitrary Units")
    ax_proba.set_xlabel("Classifier Response")

    # plot the pred distributions
    ax_pred.hist(pred_bc, bins=pred_bins, weights=train_h_pred_sig[0],
                 label="Sig (train)", histtype="stepfilled", alpha=0.5, edgecolor="C0", color="C0")
    ax_pred.hist(pred_bc, bins=pred_bins, weights=train_h_pred_bkg[0],
                 label="Bkg (train)", histtype="stepfilled", alpha=0.4, edgecolor="C3", color="C3")
    ax_pred.errorbar(pred_bc, test_h_pred_sig[0], yerr=test_h_pred_sig[1],
                     label="Sig (test)", color="C0", fmt="o", markersize=4)
    ax_pred.errorbar(pred_bc, test_h_pred_bkg[0], yerr=test_h_pred_bkg[1],
                     label="Bkg (test)", color="C3", fmt="o", markersize=4)
    ax_pred.set_ylim([0, 1.4 * ax_pred.get_ylim()[1]])
    ax_pred.legend(loc="upper right", ncol=2, frameon=False, numpoints=1)
    ax_pred.set_ylabel("Arbitrary Units")
    ax_pred.set_xlabel("Classifier Response")

    # plot the auc
    # use the new sklearn plot_roc_curve API
    rcd = plot_roc_curve(model, X_test, y_test, sample_weight=w_test, ax=ax_roc, lw=2)
    # fpr, tpr, thresholds = roc_curve(y_test, test_proba, sample_weight=w_test)
    # roc_auc = auc(fpr, tpr)
    # ax_roc.plot(fpr, tpr, lw=1, label=f"AUC = {roc_auc:0.3}")
    ax_roc.set_ylabel("True postive rate")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.grid(color="black", alpha=0.125)
    ax_roc.legend(loc="lower right")
    # fmt: on

    ks_stat_sig, ks_p_sig = ks_twosample_binned(
        test_h_proba_sig[0], train_h_proba_sig[0], test_h_proba_sig[1], train_h_proba_sig[1]
    )
    ks_stat_bkg, ks_p_bkg = ks_twosample_binned(
        test_h_proba_bkg[0], train_h_proba_bkg[0], test_h_proba_bkg[1], train_h_proba_bkg[1]
    )

    return SingleTrainingResult(
        auc=float(rcd.roc_auc),
        ks_test_sig=float(ks_stat_sig),
        ks_pvalue_sig=float(ks_p_sig),
        ks_test_bkg=float(ks_stat_bkg),
        ks_pvalue_bkg=float(ks_p_bkg),
    )


def folded_training(
    df: pd.DataFrame,
    labels: np.ndarray,
    weights: np.ndarray,
    params: Dict[str, Any],
    fit_kw: Dict[str, Any],
    output_dir: Union[str, os.PathLike],
    region: str,
    kfold_kw: Dict[str, Any] = None,
) -> float:
    """Execute a folded training.

    Train a :obj:`lightgbm.LGBMClassifier` model using :math:`k`-fold
    cross validation using the given input data and parameters.  The
    models resulting from the training (and other important training
    information) are saved to ``output_dir``. The entries in the
    ``kfold_kw`` argument are forwarded to the
    :obj:`sklearn.model_selection.KFold` class for data
    preprocessing. The default arguments that we use are:

    - ``n_splits``: 3
    - ``shuffle``: ``True``
    - ``random_state``: 414

    Parameters
    ----------
    df : pandas.DataFrame
        Feature matrix in dataframe format
    labels : numpy.ndarray
        Event labels (``1`` for signal; ``0`` for background)
    weights : :obj:`numpy.ndarray`
        Event weights
    params : dict(str, Any)
        Dictionary of :obj:`lightgbm.LGBMClassifier` parameters
    fit_kw : dict(str, Any)
        Dictionary of arguments forwarded to :py:func:`lightgbm.LGBMClassifier.fit`.
    output_dir : str or os.PathLike
        Directory to save results of training
    region : str
        String representing the region
    kfold_kw : optional dict(str, Any)
        Arguments passed to :obj:`sklearn.model_selection.KFold`

    Returns
    -------
    float
        Negative mean area under the ROC curve (AUC)

    Examples
    --------
    >>> from tdub.data import quick_files
    >>> from tdub.train import prepare_from_root
    >>> from tdub.train import folded_training
    >>> qfiles = quick_files("/path/to/data")
    >>> df, labels, weights = prepare_from_root(qfiles["tW_DR"], qfiles["ttbar"], "2j2b")
    >>> params = dict(
    ...     boosting_type="gbdt",
    ...     num_leaves=42,
    ...     learning_rate=0.05
    ...     reg_alpha=0.2,
    ...     reg_lambda=0.8,
    ...     max_depth=5,
    ... )
    >>> folded_training(
    ...     df,
    ...     labels,
    ...     weights,
    ...     params,
    ...     {"verbose": 20},
    ...     "/path/to/train/output",
    ...     "2j2b",
    ...     kfold_kw={"n_splits": 5, "shuffle": True, "random_state": 17}
    ... )
    """
    starting_dir = os.getcwd()
    output_path = PosixPath(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    os.chdir(output_path)

    log.info("selection used on the datasets:")
    log.info("  '%s'" % df.selection_used)
    log.info("features training with:")
    for c in df.columns:
        log.info(" - %s" % c)

    log.info("model parameters:")
    for k, v in params.items():
        if v is not None:
            log.info(f"{k:>20} | {v:<12}")

    log.info("saving output to %s" % output_path.resolve())

    fig_proba_hists, ax_proba_hists = plt.subplots()
    fig_pred_hists, ax_pred_hists = plt.subplots()
    fig_rocs, ax_rocs = plt.subplots()

    tprs = []
    aucs = []
    importances = np.zeros((len(df.columns)))
    mean_fpr = np.linspace(0, 1, 100)
    folder = KFold(**kfold_kw)
    fold_number = 0
    nfits = 0
    for train_idx, test_idx in folder.split(df):
        # fmt: off
        X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        w_train, w_test = weights[train_idx], weights[test_idx]
        validation_data = [(X_test, y_test)]
        validation_w = w_test

        # n_sig = y_train[y_train == 1].shape[0]
        # n_bkg = y_train[y_train == 0].shape[0]
        # scale_pos_weight = n_bkg / n_sig
        # log.info(f"n_bkg / n_sig = {n_bkg} / {n_sig} = {scale_pos_weight}")
        # params["scale_pos_weight"] = scale_pos_weight

        model = lgbm.LGBMClassifier(**params)
        fitted_model = model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=validation_data,
            eval_metric="auc",
            eval_sample_weight=[validation_w],
            **fit_kw,
        )

        joblib.dump(fitted_model, f"model_fold{fold_number}.joblib.gz", compress=("gzip", 3))

        nfits += 1
        importances += fitted_model.feature_importances_

        fold_fig_proba, fold_ax_proba = plt.subplots()
        fold_fig_pred, fold_ax_pred = plt.subplots()

        test_proba = fitted_model.predict_proba(X_test)[:, 1]
        train_proba = fitted_model.predict_proba(X_train)[:, 1]
        test_pred = fitted_model.predict(X_test, raw_score=True)
        train_pred = fitted_model.predict(X_train, raw_score=True)

        proba_sig_test = test_proba[y_test == 1]
        proba_bkg_test = test_proba[y_test == 0]
        proba_sig_train = train_proba[y_train == 1]
        proba_bkg_train = train_proba[y_train == 0]
        pred_sig_test = test_pred[y_test == 1]
        pred_bkg_test = test_pred[y_test == 0]
        pred_sig_train = train_pred[y_train == 1]
        pred_bkg_train = train_pred[y_train == 0]
        w_sig_test = w_test[y_test == 1]
        w_bkg_test = w_test[y_test == 0]
        w_sig_train = w_train[y_train == 1]
        w_bkg_train = w_train[y_train == 0]
        proba_bins = np.linspace(0, 1, 31)
        proba_bc = bin_centers(proba_bins)
        predxmin = min(pred_bkg_test.min(), pred_bkg_train.min())
        predxmax = max(pred_sig_test.max(), pred_sig_train.max())
        pred_bins = np.linspace(predxmin, predxmax, 31)

        # Axis with all folds (proba histograms)
        ax_proba_hists.hist(proba_sig_test, bins=proba_bins, label=f"F{fold_number} Sig. (test)",
                            density=True, histtype="step", weights=w_sig_test)
        ax_proba_hists.hist(proba_bkg_test, bins=proba_bins, label=f"F{fold_number} Bkg. (test)",
                            density=True, histtype="step", weights=w_bkg_test)
        ax_proba_hists.hist(proba_sig_train, bins=proba_bins, label=f"F{fold_number} Sig. (train)",
                            density=True, histtype="step", weights=w_sig_train)
        ax_proba_hists.hist(proba_bkg_train, bins=proba_bins, label=f"F{fold_number} Bkg. (train)",
                            density=True, histtype="step", weights=w_bkg_train)

        # Axis specific to the fold (proba histograms)
        fold_ax_proba.hist(proba_sig_train, bins=proba_bins, label=f"F{fold_number} Sig. (train)", density=True,
                           weights=w_sig_train, histtype="stepfilled", color="C0", edgecolor="C0", alpha=0.5, linewidth=1)
        fold_ax_proba.hist(proba_bkg_train, bins=proba_bins, label=f"F{fold_number} Bkg. (train)", density=True,
                           weights=w_bkg_train, histtype="stepfilled", color="C3", edgecolor="C3", alpha=0.4, linewidth=1)

        train_h_sig = pygram11.histogram(proba_sig_test, bins=proba_bins, weights=w_sig_test, flow=False, density=True)
        train_h_bkg = pygram11.histogram(proba_bkg_test, bins=proba_bins, weights=w_bkg_test, flow=False, density=True)

        fold_ax_proba.errorbar(proba_bc, train_h_sig[0], yerr=train_h_sig[1], markersize=4,
                               color="C0", fmt="o", label=f"F{fold_number} Sig. (test)")
        fold_ax_proba.errorbar(proba_bc, train_h_bkg[0], yerr=train_h_bkg[1], markersize=4,
                               color="C3", fmt="o", label=f"F{fold_number} Bkg. (test)")
        fold_ax_proba.set_ylim([0, 1.5 * fold_ax_proba.get_ylim()[1]])

        # Axis with all
        ax_pred_hists.hist(pred_sig_test, bins=pred_bins, label=f"F{fold_number} Sig. (test)",
                           density=True, histtype="step", weights=w_sig_test)
        ax_pred_hists.hist(pred_bkg_test, bins=pred_bins, label=f"F{fold_number} Bkg. (test)",
                           density=True, histtype="step", weights=w_bkg_test)
        ax_pred_hists.hist(pred_sig_train, bins=pred_bins, label=f"F{fold_number} Sig. (train)",
                           density=True, histtype="step", weights=w_sig_train)
        ax_pred_hists.hist(pred_bkg_train, bins=pred_bins, label=f"F{fold_number} Bkg. (train)",
                           density=True, histtype="step", weights=w_bkg_train)

        fold_ax_pred.hist(pred_sig_test, bins=pred_bins, label=f"F{fold_number} Sig. (test)",
                          density=True, histtype="step", weights=w_sig_test)
        fold_ax_pred.hist(pred_bkg_test, bins=pred_bins, label=f"F{fold_number} Bkg. (test)",
                          density=True, histtype="step", weights=w_bkg_test)
        fold_ax_pred.hist(pred_sig_train, bins=pred_bins, label=f"F{fold_number} Sig. (train)",
                          density=True, histtype="step", weights=w_sig_train)
        fold_ax_pred.hist(pred_bkg_train, bins=pred_bins, label=f"F{fold_number} Bkg. (train)",
                          density=True, histtype="step", weights=w_bkg_train)
        fold_ax_pred.set_ylim([0, 1.5 * fold_ax_pred.get_ylim()[1]])

        fpr, tpr, thresholds = roc_curve(y_test, test_proba, sample_weight=w_test)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax_rocs.plot(fpr, tpr, lw=1, alpha=0.45, label=f"fold {fold_number}, AUC = {roc_auc:0.3}")

        fold_ax_proba.set_ylabel("Arb. Units")
        fold_ax_proba.set_xlabel("Classifier Output")
        fold_ax_proba.legend(ncol=2, loc="upper center")

        fold_ax_pred.set_ylabel("Arb. Units")
        fold_ax_pred.set_xlabel("Classifier Output")
        fold_ax_pred.legend(ncol=2, loc="upper center")

        # fold_fig_proba.subplots_adjust(**_fig_adjustment_dict)
        fold_fig_proba.savefig(f"fold{fold_number}_histograms_proba.pdf")
        # fold_fig_pred.subplots_adjust(**_fig_adjustment_dict)
        fold_fig_pred.savefig(f"fold{fold_number}_histograms_pred.pdf")

        plt.close(fold_fig_proba)
        plt.close(fold_fig_pred)

        # fmt: on
        fold_number += 1

    relative_importances = importances / nfits
    relative_importances = relative_importances / relative_importances.sum()

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax_rocs.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=f"AUC = {mean_auc:0.2} $\\pm$ {std_auc:0.2}",
        lw=2,
        alpha=0.8,
    )
    ax_rocs.set_xlabel("False Positive Rate")
    ax_rocs.set_ylabel("True Positive Rate")

    ax_proba_hists.set_ylabel("Arb. Units")
    ax_proba_hists.set_xlabel("Classifier Output")
    ax_proba_hists.legend(ncol=3, loc="upper center", fontsize="small")
    ax_proba_hists.set_ylim([0, 1.5 * ax_proba_hists.get_ylim()[1]])
    # fig_proba_hists.subplots_adjust(**_fig_adjustment_dict)
    fig_proba_hists.savefig("histograms_proba.pdf")

    ax_pred_hists.set_ylabel("Arb. Units")
    ax_pred_hists.set_xlabel("Classifier Output")
    ax_pred_hists.legend(ncol=3, loc="upper center", fontsize="small")
    ax_pred_hists.set_ylim([0, 1.5 * ax_pred_hists.get_ylim()[1]])
    # fig_pred_hists.subplots_adjust(**_fig_adjustment_dict)
    fig_pred_hists.savefig("histograms_pred.pdf")

    ax_rocs.grid(color="black", alpha=0.15)
    ax_rocs.legend(ncol=2, loc="lower right")
    # fig_rocs.subplots_adjust(**_fig_adjustment_dict)
    fig_rocs.savefig("roc.pdf")

    summary: Dict[str, Any] = {}
    summary["selection_used"] = df.selection_used
    summary["region"] = region
    summary["features"] = [str(c) for c in df.columns]
    summary["importances"] = list(relative_importances)
    summary["kfold"] = kfold_kw
    summary["roc"] = {
        "auc": mean_auc,
        "std": std_auc,
        "mean_fpr": list(mean_fpr),
        "mean_tpr": list(mean_tpr),
    }

    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    os.chdir(starting_dir)
    negative_roc_score = -1.0 * np.mean(aucs)
    return negative_roc_score


_best_fit = None
_best_auc = None
_best_parameters = None
_best_paramdict = None
_ifit = None


def gp_minimize_auc(
    data_dir: str,
    region: Union[Region, str],
    nlo_method: str,
    output_dir: Union[str, os.PathLike] = "_unnamed_optimization",
    n_calls: int = 15,
    esr: Optional[int] = 10,
    random_state: int = 414,
):
    """Minimize AUC using Gaussian processes.

    This is our hyperparameter optimization procedure which uses the
    :py:func:`skopt.gp_minimize` functions from Scikit-Optimize.

    Parameters
    ----------
    data_dir : str
        Path containing ROOT files
    region : Region or str
        Rregion where we're going to perform the training
    nlo_method : str
        Which tW NLO method sample ('DR' or 'DS' or 'Both')
    output_dir : str or os.PathLike
        Path to save optimization output
    n_calls : int
        Number of times to train during the minimization procedure
    esr : int, optional
        Early stopping rounds for fitting the model
    random_state: int
        Random state for splitting data into training/testing

    Examples
    --------
    >>> from tdub.data import Region
    >>> from tdub.train import prepare_from_root, gp_minimize_auc
    >>> gp_minimize_auc("/path/to/data", Region.r2j1b, "DS", "opt_DS_2j1b")
    >>> gp_minimize_auc("/path/to/data", Region.r2j1b, "DR", "opt_DR_2j1b")
    """

    from skopt.utils import use_named_args
    from skopt.space import Real, Integer
    from skopt.plots import plot_convergence
    from skopt import gp_minimize

    qfiles = quick_files(f"{data_dir}")
    if nlo_method == "DR":
        tW_files = qfiles["tW_DR"]
    elif nlo_method == "DS":
        tW_files = qfiles["tW_DS"]
    elif nlo_method == "Both":
        tW_files = qfiles["tW_DR"] + qfiles["tW_DS"]
        tW_files.sort()
    else:
        raise ValueError("nlo_method must be 'DR' or 'DS' or 'Both'")

    df, labels, weights = prepare_from_root(tW_files, qfiles["ttbar"], region)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        df, labels, weights, train_size=0.333, random_state=random_state, shuffle=True
    )
    validation_data = [(X_test, y_test)]
    validation_w = w_test

    # n_sig = y_train[y_train == 1].shape[0]
    # n_bkg = y_train[y_train == 0].shape[0]
    # scale_pos_weight = n_bkg / n_sig
    # sample_size = n_bkg + n_sig
    # log.info(f"n_bkg / n_sig = {n_bkg} / {n_sig} = {scale_pos_weight}")

    dimensions = [
        Real(low=0.01, high=0.2, prior="log-uniform", name="learning_rate"),
        Integer(low=40, high=250, name="num_leaves"),
        Integer(low=20, high=250, name="min_child_samples"),
        Integer(low=3, high=10, name="max_depth"),
    ]
    default_parameters = [0.1, 100, 50, 5]

    run_from_dir = os.getcwd()
    save_dir = PosixPath(output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    os.chdir(save_dir)

    global _best_fit
    global _best_auc
    global _best_parameters
    global _best_paramdict
    global _ifit
    _best_fit = 0
    _best_auc = 0.0
    _best_parameters = [{"teste": 1}]
    _best_paramdict = {}
    _ifit = 0

    @use_named_args(dimensions=dimensions)
    def afit(
        learning_rate, num_leaves, min_child_samples, max_depth,
    ):
        global _ifit
        global _best_fit
        global _best_auc
        global _best_parameters
        global _best_paramdict

        log.info(f"on iteration {_ifit} out of {n_calls}")
        log.info(f"learning_rate: {learning_rate}")
        log.info(f"num_leaves: {num_leaves}")
        log.info(f"min_child_samples: {min_child_samples}")
        log.info(f"max_depth: {max_depth}")

        curdir = os.getcwd()
        p = PosixPath(f"training_{_ifit}")
        p.mkdir(exist_ok=False)
        os.chdir(p.resolve())

        with open("features.txt", "w") as f:
            for c in df.columns:
                print(c, file=f)

        model = lgbm.LGBMClassifier(
            boosting_type="gbdt",
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            max_depth=max_depth,
            n_estimators=500,
        )

        fitted_model = model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=validation_data,
            eval_metric="auc",
            verbose=20,
            early_stopping_rounds=esr,
            eval_sample_weight=[validation_w],
        )

        pred = fitted_model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, pred, sample_weight=w_test)

        train_pred = fitted_model.predict_proba(X_train)[:, 1]
        fig, ax = plt.subplots()
        bins = np.linspace(0, 1, 31)
        ax.hist(
            train_pred[y_train == 0],
            bins=bins,
            label="Bkg. (train)",
            density=True,
            histtype="step",
            weights=w_train[y_train == 0],
        )
        ax.hist(
            train_pred[y_train == 1],
            bins=bins,
            label="Sig. (train)",
            density=True,
            histtype="step",
            weights=w_train[y_train == 1],
        )
        ax.hist(
            pred[y_test == 0],
            bins=bins,
            label="Bkg. (test)",
            density=True,
            histtype="step",
            weights=w_test[y_test == 0],
        )
        ax.hist(
            pred[y_test == 1],
            bins=bins,
            label="Sig. (test)",
            density=True,
            histtype="step",
            weights=w_test[y_test == 1],
        )
        ax.set_ylim([0, 1.5 * ax.get_ylim()[1]])
        ax.legend(ncol=2, loc="upper center")
        # fig.subplots_adjust(**_fig_adjustment_dict)
        fig.savefig("histograms.pdf")
        plt.close(fig)

        binning_sig = np.linspace(0, 1, 31)
        binning_bkg = np.linspace(0, 1, 31)

        h_sig_test, err_sig_test = pygram11.histogram(
            pred[y_test == 1], bins=binning_sig, weights=w_test[y_test == 1]
        )
        h_sig_train, err_sig_train = pygram11.histogram(
            train_pred[y_train == 1], bins=binning_sig, weights=w_train[y_train == 1]
        )

        h_bkg_test, err_bkg_test = pygram11.histogram(
            pred[y_test == 0], bins=binning_bkg, weights=w_test[y_test == 0]
        )
        h_bkg_train, err_bkg_train = pygram11.histogram(
            train_pred[y_train == 0], bins=binning_bkg, weights=w_train[y_train == 0]
        )

        ks_statistic_sig, ks_pvalue_sig = ks_twosample_binned(
            h_sig_test, h_sig_train, err_sig_test, err_sig_train
        )
        ks_statistic_bkg, ks_pvalue_bkg = ks_twosample_binned(
            h_bkg_test, h_bkg_train, err_bkg_test, err_bkg_train
        )

        if ks_pvalue_sig < 0.1 or ks_pvalue_bkg < 0.1:
            score = score * 0.9

        log.info(f"ksp sig: {ks_pvalue_sig}")
        log.info(f"ksp bkg: {ks_pvalue_bkg}")

        curp = pformat(model.get_params())
        curp = eval(curp)

        with open("params.json", "w") as f:
            json.dump(curp, f, indent=4)

        with open("auc.txt", "w") as f:
            print(f"{score}", file=f)

        with open("ks.txt", "w") as f:
            print(f"sig: {ks_pvalue_sig}", file=f)
            print(f"bkg: {ks_pvalue_bkg}", file=f)

        os.chdir(curdir)

        if score > _best_auc:
            _best_parameters[0] = model.get_params()
            _best_auc = score
            _best_fit = _ifit
            _best_paramdict = curp

        _ifit += 1

        del model
        return -score

    search_result = gp_minimize(
        func=afit,
        dimensions=dimensions,
        acq_func="gp_hedge",
        n_calls=n_calls,
        x0=default_parameters,
    )

    summary = {
        "region": region,
        "nlo_method": nlo_method,
        "features": list(df.columns),
        "best_iteration": _best_fit,
        "best_auc": _best_auc,
        "best_params": _best_paramdict,
    }

    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    fig, ax = plt.subplots()
    plot_convergence(search_result, ax=ax)
    # fig.subplots_adjust(**_fig_adjustment_dict)
    fig.savefig("gpmin_convergence.pdf")

    os.chdir(run_from_dir)
    return 0

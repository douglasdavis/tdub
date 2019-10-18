"""
Module for training BDTs
"""

from __future__ import annotations

# stdlib
import json
import logging
import os
from pathlib import PosixPath
from pprint import pformat

# externals
import joblib
import lightgbm as lgbm
import numpy as np
import matplotlib.pyplot as plt
import pygram11
from scipy import interp
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import auc, roc_auc_score, roc_curve

# tdub
from tdub.frames import specific_dataframe
from tdub.utils import Region, bin_centers, quick_files


log = logging.getLogger(__name__)


def prepare_from_root(
    sig_files: List[str],
    bkg_files: List[str],
    region: Union[Region, str],
    weight_scale: float = 1.0e3,
    scale_sum_weights: bool = True,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Prepare the data for training in a region with signal and
    background ROOT files

    Parameters
    ----------
    sig_files : list(str)
       list of signal ROOT files
    bkg_files : list(str)
       list of background ROOT files
    region : Region or str
       the region where we're going to perform the training
    weight_scale : float
       value to scale all weights by
    scale_sum_weights : bool
       scale sum of weights of signal to be sum of weights of background

    Returns
    -------
    X : :obj:`numpy.ndarray`
       the feature matrix
    y : :obj:`numpy.ndarray`
       the event labels
    w : :obj:`numpy.ndarray`
       the event weights
    cols : list(str)
       list of features which are columns of ``X``.

    Examples
    --------

    >>> from tdub.utils import quick_files
    >>> from tdub.train import prepare_from_root
    >>> qfiles = quick_files("/path/to/data")
    >>> X, y, w, cols = prepare_from_root(qfiles["tW_DR"], qfiles["ttbar"], "2j2b")

    """
    log.info("preparing training data")
    log.info("signal files:")
    for f in sig_files:
        log.info(f"  - {f}")
    log.info("background files:")
    for f in bkg_files:
        log.info(f"  - {f}")

    ## signal pretty much always be tW, no need for dask
    sig_dfim = specific_dataframe(
        sig_files, region, "train_sig", bypass_dask=True, dropnonkin=True
    )
    ## bkg is pretty much always ttbar, so lets use daks to be careful
    bkg_dfim = specific_dataframe(
        bkg_files, region, "train_bkg", to_ram=True, dropnonkin=True
    )

    sorted_cols = sorted(sig_dfim.df.columns.to_list(), key=str.lower)
    sig_dfim._df = sig_dfim._df[sorted_cols]
    bkg_dfim._df = bkg_dfim._df[sorted_cols]

    cols = sig_dfim.df.columns.to_list()
    assert cols == bkg_dfim.df.columns.to_list(), "sig/bkg columns are different. bad."
    log.info("features used:")
    for c in cols:
        log.info(f"  - {c}")

    w_sig = sig_dfim.weights.weight_nominal.to_numpy()
    w_bkg = bkg_dfim.weights.weight_nominal.to_numpy()
    w_sig[w_sig < 0] = 0.0
    w_bkg[w_bkg < 0] = 0.0
    w_sig *= weight_scale
    w_bkg *= weight_scale
    if scale_sum_weights:
        w_sig *= w_bkg.sum() / w_sig.sum()

    X = np.concatenate([sig_dfim.df.to_numpy(), bkg_dfim.df.to_numpy()])
    w = np.concatenate([w_sig, w_bkg])
    y = np.concatenate([np.ones_like(w_sig), np.zeros_like(w_bkg)])

    return X, y, w, cols


def folded_training(
    X: numpy.ndarray,
    y: numpy.ndarray,
    w: numpy.ndarray,
    cols: List[str],
    params: Dict[str, Any],
    fit_kw: Dict[str, Any],
    output_dir: Union[str, os.PathLike],
    region: str,
    use_sample_weights: bool = False,
    kfold_kw: Dict[str, Any] = None,
) -> float:
    """Train a :obj:`lightgbm.LGBMClassifier` model using :math:`k`-fold
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
    X : :obj:`numpy.ndarray`
       the feature matrix
    y : :obj:`numpy.ndarray`
       the event labels
    w : :obj:`numpy.ndarray`
       the event weights
    cols : list(str)
       list of features which are the columns of ``X``.
    params : dict(str, Any)
       dictionary of :obj:`lightgbm.LGBMClassifier` parameters
    fit_kw : dict(str, Any)
       dictionary of arguments forwarded to :py:func:`lightgbm.LGBMClassifier.fit`.
    output_dir : str or os.PathLike
       directory to save results of training
    region : str
        string representing the region
    use_sample_weights : bool
       if ``True``, use the sample weights in training instead of the
       ``is_unbalance=True`` (which is the default case when this
       argument is ``False``)
    kfold_kw : optional dict(str, Any)
       arguments fed to :obj:`sklearn.model_selection.KFold`

    Returns
    -------
    neg_roc_score : float
       -1 times the mean area under the ROC curve (AUC)

    Examples
    --------

    >>> from tdub.utils import quick_files
    >>> from tdub.train import prepare_from_root
    >>> from tdub.train import folded_training
    >>> qfiles = quick_files("/path/to/data")
    >>> X, y, w, cols = prepare_from_root(qfiles["tW_DR"], qfiles["ttbar"], "2j2b")
    >>> params = dict(
    ...     boosting_type="gbdt",
    ...     num_leaves=42,
    ...     learning_rate=0.1
    ...     subsample_for_bin=180000,
    ...     min_child_samples=40,
    ...     reg_alpha=0.4,
    ...     reg_lambda=0.5,
    ...     colsample_bytree=0.8,
    ...     n_estimators=200,
    ...     max_depth=5,
    ...     is_unbalance=True,
    ... )
    >>> folded_training(
    ...     X,
    ...     y,
    ...     w,
    ...     cols,
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

    fig_proba_hists, ax_proba_hists = plt.subplots()
    fig_pred_hists, ax_pred_hists = plt.subplots()
    fig_rocs, ax_rocs = plt.subplots()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    folder = KFold(**kfold_kw)
    fold_number = 0
    for train_idx, test_idx in folder.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train, w_test = w[train_idx], w[test_idx]
        validation_data = [(X_test, y_test)]
        validation_w = w_test

        if use_sample_weights:
            params["is_unbalance"] = False
            model = lgbm.LGBMClassifier(**params)
            fitted_model = model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=validation_data,
                eval_sample_weight=[validation_w],
                **fit_kw,
            )
        else:
            params["is_unbalance"] = True
            model = lgbm.LGBMClassifier(**params)
            fitted_model = model.fit(
                X_train,
                y_train,
                eval_set=validation_data,
                eval_sample_weight=[validation_w],
                **fit_kw,
            )

        joblib.dump(
            fitted_model, f"model_fold{fold_number}.joblib.gz", compress=("gzip", 3)
        )

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

        proba_bins = np.linspace(proba_bkg_test.min(), proba_sig_test.max(), 41)
        proba_bc = bin_centers(proba_bins)
        pred_bins = np.linspace(pred_bkg_test.min(), pred_sig_test.max(), 41)
        pred_bc = bin_centers(pred_bins)

        ### Axis with all folds (proba histograms)
        ax_proba_hists.hist(
            proba_sig_test,
            bins=proba_bins,
            label=f"F{fold_number} Sig. (test)",
            density=True,
            histtype="step",
            weights=w_sig_test,
        )
        ax_proba_hists.hist(
            proba_bkg_test,
            bins=proba_bins,
            label=f"F{fold_number} Bkg. (test)",
            density=True,
            histtype="step",
            weights=w_bkg_test,
        )
        ax_proba_hists.hist(
            proba_sig_train,
            bins=proba_bins,
            label=f"F{fold_number} Sig. (train)",
            density=True,
            histtype="step",
            weights=w_sig_train,
        )
        ax_proba_hists.hist(
            proba_bkg_train,
            bins=proba_bins,
            label=f"F{fold_number} Bkg. (train)",
            density=True,
            histtype="step",
            weights=w_bkg_train,
        )

        ### Axis specific to the fold (proba histograms)
        fold_ax_proba.hist(
            proba_sig_train,
            bins=proba_bins,
            label=f"F{fold_number} Sig. (train)",
            weights=w_sig_train,
            density=True,
            histtype="stepfilled",
            color="C0",
            edgecolor="C0",
            alpha=0.5,
            linewidth=1,
        )
        fold_ax_proba.hist(
            proba_bkg_train,
            bins=proba_bins,
            label=f"F{fold_number} Bkg. (train)",
            weights=w_bkg_train,
            density=True,
            histtype="step",
            hatch="//",
            edgecolor="C3",
            linewidth=1,
        )
        train_h_sig = pygram11.histogram(
            proba_sig_test, bins=proba_bins, weights=w_sig_test, flow=False, density=True
        )
        train_h_bkg = pygram11.histogram(
            proba_bkg_test, bins=proba_bins, weights=w_bkg_test, flow=False, density=True
        )
        fold_ax_proba.errorbar(
            proba_bc,
            train_h_sig[0],
            yerr=train_h_sig[1],
            color="C0",
            fmt="o",
            label=f"F{fold_number} Sig. (test)",
            markersize=4,
        )
        fold_ax_proba.errorbar(
            proba_bc,
            train_h_bkg[0],
            yerr=train_h_bkg[1],
            color="C3",
            fmt="o",
            label=f"F{fold_number} Bkg. (test)",
            markersize=4,
        )
        fold_ax_proba.set_ylim([0, 1.5 * fold_ax_proba.get_ylim()[1]])

        ### Axis with all
        ax_pred_hists.hist(
            pred_sig_test,
            bins=pred_bins,
            label=f"F{fold_number} Sig. (test)",
            density=True,
            histtype="step",
            weights=w_sig_test,
        )
        ax_pred_hists.hist(
            pred_bkg_test,
            bins=pred_bins,
            label=f"F{fold_number} Bkg. (test)",
            density=True,
            histtype="step",
            weights=w_bkg_test,
        )
        ax_pred_hists.hist(
            pred_sig_train,
            bins=pred_bins,
            label=f"F{fold_number} Sig. (train)",
            density=True,
            histtype="step",
            weights=w_sig_train,
        )
        ax_pred_hists.hist(
            pred_bkg_train,
            bins=pred_bins,
            label=f"F{fold_number} Bkg. (train)",
            density=True,
            histtype="step",
            weights=w_bkg_train,
        )

        fold_ax_pred.hist(
            pred_sig_test,
            bins=pred_bins,
            label=f"F{fold_number} Sig. (test)",
            density=True,
            histtype="step",
            weights=w_sig_test,
        )
        fold_ax_pred.hist(
            pred_bkg_test,
            bins=pred_bins,
            label=f"F{fold_number} Bkg. (test)",
            density=True,
            histtype="step",
            weights=w_bkg_test,
        )
        fold_ax_pred.hist(
            pred_sig_train,
            bins=pred_bins,
            label=f"F{fold_number} Sig. (train)",
            density=True,
            histtype="step",
            weights=w_sig_train,
        )
        fold_ax_pred.hist(
            pred_bkg_train,
            bins=pred_bins,
            label=f"F{fold_number} Bkg. (train)",
            density=True,
            histtype="step",
            weights=w_bkg_train,
        )
        fold_ax_pred.set_ylim([0, 1.5 * fold_ax_pred.get_ylim()[1]])

        fpr, tpr, thresholds = roc_curve(y_test, test_proba, sample_weight=w_test)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax_rocs.plot(
            fpr, tpr, lw=1, alpha=0.45, label=f"fold {fold_number}, AUC = {roc_auc:0.3}"
        )

        fold_ax_proba.legend(ncol=2, loc="upper center")
        fold_ax_pred.legend(ncol=2, loc="upper center")
        fold_fig_proba.savefig(f"fold{fold_number}_histograms_proba.pdf")
        fold_fig_pred.savefig(f"fold{fold_number}_histograms_pred.pdf")

        plt.close(fold_fig_proba)
        plt.close(fold_fig_pred)

        fold_number += 1

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

    ax_proba_hists.legend(ncol=3, loc="upper center", fontsize="small")
    ax_proba_hists.set_ylim([0, 1.5 * ax_proba_hists.get_ylim()[1]])
    fig_proba_hists.savefig("histograms_proba.pdf")

    ax_pred_hists.legend(ncol=3, loc="upper center", fontsize="small")
    ax_pred_hists.set_ylim([0, 1.5 * ax_pred_hists.get_ylim()[1]])
    fig_pred_hists.savefig("histograms_pred.pdf")

    ax_rocs.legend(ncol=2, loc="lower right")
    fig_rocs.savefig("roc.pdf")

    summary = {}
    summary["region"] = region
    summary["features"] = [str(c) for c in cols]
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
    neg_roc_score = -1.0 * np.mean(aucs)
    return neg_roc_score


def gp_minimize_auc(
    region: Union[Region, str],
    nlo_method: str,
    data_dir: str,
    output_dir: Union[str, os.PathLike] = "_unnamed_optimization",
    n_calls: int = 15,
    esr: int = 20,
):
    """Minimize AUC using Gaussian processes

    This is our hyperparameter optimization procedure which uses the
    :py:func:`skopt.gp_minimize` functions from Scikit-Optimize.

    Parameters
    ----------
    region : Region or str
       the region where we're going to perform the training
    nlo_method : str
       which tW NLO method sample ('DR' or 'DS' or 'Both')
    data_dir : str
       path containing ROOT files
    output_dir : str or os.PathLike
       path to save optimization output
    n_calls : int
       number of times to train during the minimization procedure
    esr : int
       early stopping rounds for fitting the model

    Examples
    --------

    >>> from tdub.utils import Region
    >>> from tdub.train import prepare_from_root, gp_minimize_auc
    >>> gp_minimize_auc(Region.r2j1b, "DS", "/path/to/data", "opt_DS_2j1b")
    >>> gp_minimize_auc(Region.r2j1b, "DR", "/path/to/data", "opt_DR_2j1b")

    """

    from skopt.utils import use_named_args
    from skopt.space import Real, Integer, Categorical
    from skopt import gp_minimize

    run_from_dir = os.getcwd()
    save_dir = PosixPath(output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    os.chdir(save_dir)

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
    X, y, w, cols = prepare_from_root(tW_files, qfiles["ttbar"], region)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, train_size=0.333, random_state=414, shuffle=True
    )
    validation_data = [(X_test, y_test)]
    validation_w = w_test

    dimensions = [
        Integer(low=30, high=150, name="num_leaves"),
        Real(low=1e-3, high=2e-1, prior="log-uniform", name="learning_rate"),
        Integer(low=20000, high=300000, name="subsample_for_bin"),
        Integer(low=20, high=500, name="min_child_samples"),
        Real(low=0, high=1, prior="uniform", name="reg_alpha"),
        Real(low=0, high=1, prior="uniform", name="reg_lambda"),
        Real(low=0.6, high=1, prior="uniform", name="colsample_bytree"),
        Integer(low=20, high=1000, name="n_estimators"),
        Integer(low=3, high=8, name="max_depth"),
    ]
    default_parameters = [42, 1e-1, 180000, 40, 0.4, 0.5, 0.8, 200, 5]

    global best_fit
    global best_auc
    global best_parameters
    global best_paramdict
    global ifit
    best_fit = 0
    best_auc = 0.0
    best_parameters = [{"teste": 1}]
    best_paramdict = {}
    ifit = 0

    @use_named_args(dimensions=dimensions)
    def afit(
        num_leaves,
        learning_rate,
        subsample_for_bin,
        min_child_samples,
        reg_alpha,
        reg_lambda,
        colsample_bytree,
        n_estimators,
        max_depth,
    ):
        global ifit
        global best_fit
        global best_auc
        global best_parameters
        global best_paramdict

        log.info(f"num_leaves: {num_leaves}")
        log.info(f"learning_rate: {learning_rate}")
        log.info(f"subsample_for_bin: {subsample_for_bin}")
        log.info(f"min_child_samples: {min_child_samples}")
        log.info(f"reg_alpha: {reg_alpha}")
        log.info(f"reg_lambda: {reg_lambda}")
        log.info(f"colsample_bytree: {colsample_bytree}")
        log.info(f"n_estimators: {n_estimators}")
        log.info(f"max_depth: {max_depth}")

        curdir = os.getcwd()
        p = PosixPath(f"training_{ifit}")
        p.mkdir(exist_ok=False)
        os.chdir(p.resolve())

        with open("features.txt", "w") as f:
            for c in cols:
                print(c, file=f)

        model = lgbm.LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample_for_bin=subsample_for_bin,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            colsample_bytree=colsample_bytree,
            n_estimators=n_estimators,
            max_depth=max_depth,
            is_unbalance=True,
        )

        fitted_model = model.fit(
            X_train,
            y_train,
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
        xmin = np.min(pred[y_test == 0])
        xmax = np.max(pred[y_test == 1])
        bins = np.linspace(xmin, xmax, 26)
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
        fig.savefig("histograms.pdf")
        plt.close(fig)

        curp = pformat(model.get_params())
        curp = eval(curp)

        with open("params.json", "w") as f:
            json.dump(curp, f, indent=4)

        with open("auc.txt", "w") as f:
            print(f"{score}", file=f)

        os.chdir(curdir)

        if score > best_auc:
            best_parameters[0] = model.get_params()
            best_auc = score
            best_fit = ifit
            best_paramdict = curp

        ifit += 1

        del model
        return -score

    search_result = gp_minimize(
        func=afit,
        dimensions=dimensions,
        acq_func="EI",
        n_calls=n_calls,
        x0=default_parameters,
    )

    summary = {
        "region": region,
        "nlo_method": nlo_method,
        "features": cols,
        "best_iteration": best_fit,
        "best_auc": best_auc,
        "best_params": best_paramdict,
    }

    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    os.chdir(run_from_dir)
    return 0

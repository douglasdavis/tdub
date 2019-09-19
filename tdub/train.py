from __future__ import annotations

import os
from pathlib import PosixPath
import logging
from pprint import pformat

import lightgbm as lgbm
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt

from tdub.frames import specific_dataframe
from tdub.regions import Region


log = logging.getLogger(__name__)


def prepare_from_root(
    sig_files: List[str],
    bkg_files: List[str],
    region: Union[Region, str],
    weight_scale: float = 1.0e3,
    scale_sum_weights: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    >>> X, y, w = prepare_from_root(qfiles["tW_DR"], qfiles["ttbar"], "2j2b")

    """
    log.info("preparing training data")
    log.info("signal files:")
    for f in sig_files:
        log.info(f"  - {f}")
    log.info("background files:")
    for f in bkg_files:
        log.info(f"  - {f}")

    sig_dfim = specific_dataframe(sig_files, region, "train_sig", to_ram=True)
    bkg_dfim = specific_dataframe(bkg_files, region, "train_bkg", to_ram=True)

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
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    cols: List[str],
    params: Dict[str, Any],
    fit_kw: Dict[str, Any],
    output_dir: Union[str, os.PathLike],
    use_sample_weights: bool = False,
    KFold_kw: Dict[str, Any] = None,
) -> float:
    """Train a :obj:`lightgbm.LGBMClassifier` model using :math:`k`-fold
    cross validation using the given input data and parameters.  The
    models resulting from the training (and other important training
    information) are saved to ``output_dir``. The entries in the
    ``KFold_kw`` argument are forwarded to the
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
    use_sample_weights : bool
       if ``True``, use the sample weights in training instead of the
       ``is_unbalanced=True`` (which is the default case when this
       argument is ``False``)
    KFold_kw : optional dict(str, Any)
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
    >>> X, y, w = prepare_from_root(qfiles["tW_DR"], qfiles["ttbar"], "2j2b")
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
    >>> folded_training(X, y, w, params, output_dir="/path/to/train/output",
    ...                 KFold_kw={"n_splits": 5, "shuffle": True, "random_state": 17})

    """
    starting_dir = os.getcwd()
    output_path = PosixPath(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    os.chdir(output_path)

    joblib.dump(cols, "columns.joblib")

    fig_hists, ax_hists = plt.subplots()
    fig_rocs, ax_rocs = plt.subplots()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    folder = KFold(**KFold_kw)
    fold_number = 0
    for train_idx, test_idx in folder.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train, w_test = w[train_idx], w[test_idx]
        validation_data = [(X_test, y_test)]
        validation_w = w_test

        if use_sample_weights:
            params["is_unbalanced"] = False
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
            params["is_unbalanced"] = True
            model = lgbm.LGBMClassifier(**params)
            fitted_model = model.fit(
                X_train,
                y_train,
                eval_set=validation_data,
                eval_sample_weight=[validation_w],
                **fit_kw,
            )

        joblib.dump(fitted_model, f"model_fold{fold_number}.joblib")

        test_proba = fitted_model.predict_proba(X_test)[:, 1]
        train_proba = fitted_model.predict_proba(X_train)[:, 1]

        bins = np.linspace(0, 1, 26)
        ax_hists.hist(
            test_proba[y_test == 1],
            bins=bins,
            label=f"f{fold_number} s (test)",
            density=True,
            histtype="step",
            weights=w_test[y_test == 1],
        )
        ax_hists.hist(
            test_proba[y_test == 0],
            bins=bins,
            label=f"f{fold_number} b (test)",
            density=True,
            histtype="step",
            weights=w_test[y_test == 0],
        )
        ax_hists.hist(
            train_proba[y_train == 1],
            bins=bins,
            label=f"f{fold_number} s (train)",
            density=True,
            histtype="step",
            weights=w_train[y_train == 1],
        )
        ax_hists.hist(
            train_proba[y_train == 0],
            bins=bins,
            label=f"f{fold_number} b (train)",
            density=True,
            histtype="step",
            weights=w_train[y_train == 0],
        )

        fpr, tpr, thresholds = roc_curve(y_test, test_proba, sample_weight=w_test)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax_rocs.plot(
            fpr, tpr, lw=1, alpha=0.45, label=f"fold {fold_number}, AUC = {roc_auc:0.3}"
        )

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


    ax_hists.legend(ncol=3)
    fig_hists.savefig("histograms.pdf")

    ax_rocs.legend(ncol=2)
    fig_rocs.savefig("roc.pdf")

    os.chdir(starting_dir)
    neg_roc_score = -1.0 * np.mean(aucs)
    return neg_roc_score

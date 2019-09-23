#!/usr/bin/env python


from __future__ import annotations

from pprint import pprint, pformat
from pathlib import PosixPath
import os
import glob
import json
import yaml

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm
import joblib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("pdf")

from tdub.train import prepare_from_root
from tdub.utils import quick_files


def gpmin_auc(
    data_dir: str,
    output_dir: Union[str, os.PathLike] = "_unnamed_optimization",
    n_calls: int = 15,
):
    """Hyperparameter optimization via gaussian process AUC minimization

    Parameters
    ----------
    data_dir: str
       path containing ROOT files
    output_dir: str or os.PathLike
       path to save optimization output
    n_calls:
       number of times to train during the minimization procedure
    """
    run_from_dir = os.getcwd()
    save_dir = PosixPath(output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    os.chdir(save_dir)

    qfiles = quick_files(f"{data_dir}")
    X, y, w, cols = prepare_from_root(qfiles["tW_DR"], qfiles["ttbar"], "2j2b")
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
    global ifit
    best_fit = 0
    best_auc = 0.0
    best_parameters = [{"teste": 1}]
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

        print(f"num_leaves: {num_leaves}")
        print(f"learning_rate: {learning_rate}")
        print(f"subsample_for_bin: {subsample_for_bin}")
        print(f"min_child_samples: {min_child_samples}")
        print(f"reg_alpha: {reg_alpha}")
        print(f"reg_lambda: {reg_lambda}")
        print(f"colsample_bytree: {colsample_bytree}")
        print(f"n_estimators: {n_estimators}")
        print(f"max_depth: {max_depth}")

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
            early_stopping_rounds=20,
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
            label="train bkg",
            density=True,
            histtype="step",
            weights=w_train[y_train == 0],
        )
        ax.hist(
            train_pred[y_train == 1],
            bins=bins,
            label="train sig",
            density=True,
            histtype="step",
            weights=w_train[y_train == 1],
        )
        ax.hist(
            pred[y_test == 0],
            bins=bins,
            label="test bkg",
            density=True,
            histtype="step",
            weights=w_test[y_test == 0],
        )
        ax.hist(
            pred[y_test == 1],
            bins=bins,
            label="test sig",
            density=True,
            histtype="step",
            weights=w_test[y_test == 1],
        )
        ax.legend()
        fig.savefig("histograms.pdf")
        plt.close(fig)

        curp = pformat(model.get_params())
        curp = eval(curp)

        with open("params.json", "w") as f:
            json.dump(curp, f)

        with open("auc.txt", "w") as f:
            print(f"{score}", file=f)

        os.chdir(curdir)

        if score > best_auc:
            best_parameters[0] = model.get_params()
            best_auc = score
            best_fit = ifit

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

    with open("best.txt", "w") as f:
        print(f"best: training {ifit}, auc: {best_auc}", file=f)
        print("", file=f)
        print("", file=f)
        print(f"best_fit: {best_fit}", file=f)
        print("", file=f)
        print("", file=f)
        print(pformat(best_parameters[0]), file=f)
        print("", file=f)
        print("", file=f)
        print(best_parameters[0], file=f)

    os.chdir(run_from_dir)
    return 0


if __name__ == "__main__":
    gpmin_auc("/var/phy/project/hep/atlas/users/drd25/data/wtloop/v29_20190913", n_calls=30)

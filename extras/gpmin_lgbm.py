#!/usr/bin/env python

from pprint import pprint
from pathlib import PosixPath
import os


from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm
import joblib

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("pdf")

from tdub import DataFramesInMemory
from tdub.regions import FSET_2j2b

# datadir = "/Users/ddavis/ATLAS/data"
datadir = "/home/drd25/ATLAS/analysis/run/pqs"

branches = list(set(FSET_2j2b) | {"weight_nominal"})

ddf_ttbar = dd.read_parquet(f"{datadir}/ttbar_r2j2b.parquet")[branches]
ddf_tW_DR = dd.read_parquet(f"{datadir}/tW_DR_r2j2b.parquet")[branches]

dfim_ttbar = DataFramesInMemory("ttbar", ddf_ttbar)
dfim_tW_DR = DataFramesInMemory("tW_DR", ddf_tW_DR)

w_ttbar = dfim_ttbar.weights.weight_nominal.to_numpy()
w_tW_DR = dfim_tW_DR.weights.weight_nominal.to_numpy()
w_ttbar[w_ttbar < 0] = 0.0
w_tW_DR[w_tW_DR < 0] = 0.0
w_tW_DR *= w_ttbar.sum() / w_tW_DR.sum()
w_tW_DR *= 1.0e5 / w_tW_DR.sum()
w_ttbar *= 1.0e5 / w_ttbar.sum()

y = np.concatenate([np.ones_like(w_tW_DR), np.zeros_like(w_ttbar)])
w = np.concatenate([w_tW_DR, w_ttbar])
X = np.concatenate([dfim_tW_DR.df.to_numpy(), dfim_ttbar.df.to_numpy()])

pprint(dfim_tW_DR.df.columns.to_list())

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, train_size=0.33, random_state=414, shuffle=True
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
        verbose=15,
        early_stopping_rounds=5,
        eval_sample_weight=[validation_w],
    )

    pred = fitted_model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, pred, sample_weight=w_test)

    train_pred = fitted_model.predict_proba(X_train)[:, 1]
    fig, ax = plt.subplots()
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(
        train_pred[y_train == 0],
        bins=bins,
        label="train bkg",
        density=True,
        histtype="step",
    )
    ax.hist(
        train_pred[y_train == 1],
        bins=bins,
        label="train sig",
        density=True,
        histtype="step",
    )
    ax.hist(pred[y_test == 0], bins=bins, label="test bkg", density=True, histtype="step")
    ax.hist(pred[y_test == 1], bins=bins, label="test sig", density=True, histtype="step")
    ax.legend()
    fig.savefig("histograms.pdf")
    plt.close(fig)
    os.chdir(curdir)

    if score > best_auc:
        best_parameters[0] = model.get_params()
        best_auc = score
        best_fit = ifit

    ifit += 1

    del model
    return -score


search_result = gp_minimize(
    func=afit, dimensions=dimensions, acq_func="EI", n_calls=50, x0=default_parameters
)

print()
print(best_auc)
print()

print()
print(best_parameters[0])
print()

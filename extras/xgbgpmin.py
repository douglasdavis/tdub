#!/usr/bin/env python

from pprint import pprint
from pathlib import PosixPath
import os


from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("pdf")

from tdub import DataFramesInMemory
from tdub.regions import FSET_2j2b

datadir = "/Users/ddavis/ATLAS/data"

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
    X, y, w, train_size=0.22, random_state=414, shuffle=True
)

validation_data = [(X_test, y_test)]
validation_w = w_test

dimensions = [
    Integer(low=3, high=7, name="max_depth"),
    Real(low=0.001, high=0.2, prior="log-uniform", name="learning_rate"),
    Integer(low=20, high=1000, name="n_estimators"),
    Real(low=0, high=0.9, prior="uniform", name="gamma"),
    Real(low=1, high=100, prior="uniform", name="min_child_weight"),
    Integer(low=0, high=10, name="max_delta_step"),
    Real(low=0, high=1, prior="uniform", name="reg_alpha"),
    Real(low=0, high=1, prior="uniform", name="reg_lambda"),
]

default_parameters = [3, 0.1, 100, 0, 1, 0, 0, 1]

best_auc = 0.0
best_parameters = [{"teste": 1}]
ifit = 0

@use_named_args(dimensions=dimensions)
def afit(
    max_depth,
    learning_rate,
    n_estimators,
    gamma,
    min_child_weight,
    max_delta_step,
    reg_alpha,
    reg_lambda,
):
    global ifit
    global best_auc
    global best_parameters

    print(f"max_depth: {max_depth}")
    print(f"learning_rate: {learning_rate}")
    print(f"n_estimators: {n_estimators}")
    print(f"gamma: {gamma}")
    print(f"min_child_weight: {min_child_weight}")
    print(f"max_delta_step: {max_delta_step}")
    print(f"reg_alpha: {reg_alpha}")
    print(f"reg_lambda: {reg_lambda}")

    curdir = os.getcwd()
    p = PosixPath(f"training_{ifit}")
    p.mkdir(exist_ok=False)
    os.chdir(p.resolve())

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_delta_step=max_delta_step,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=12,
        objective="binary:logistic",
        booster="gbtree",
    )
    fitted_model = model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=validation_data,
        eval_metric="auc",
        verbose=20,
        early_stopping_rounds=5,
        sample_weight_eval_set=[validation_w],
    )

    pred = fitted_model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, pred, sample_weight=w_test)

    train_pred = fitted_model.predict_proba(X_train)[:, 1]
    fig, ax = plt.subplots()
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(train_pred[y_train==0], bins=bins, label="train bkg", density=True, histtype='step')
    ax.hist(train_pred[y_train==1], bins=bins, label="train sig", density=True, histtype='step')
    ax.hist(pred[y_test==0], bins=bins, label="test bkg", density=True, histtype='step')
    ax.hist(pred[y_test==1], bins=bins, label="test sig", density=True, histtype='step')
    ax.legend()
    fig.savefig("histograms.pdf")
    plt.close(fig)
    os.chdir(curdir)


    ifit += 1

    if score > best_auc:
        best_parameters[0] = model.get_params()
        best_auc = score

    del model
    return -score


search_result = gp_minimize(
    func=afit, dimensions=dimensions, acq_func="EI", n_calls=15, x0=default_parameters
)

print()
print(best_accuracy)
print()

print()
print(best_parameters[0])
print()

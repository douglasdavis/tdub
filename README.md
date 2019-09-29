# tdub

[![Build Status](https://dev.azure.com/ddavis0485/tdub/_apis/build/status/douglasdavis.tdub?branchName=master)](https://dev.azure.com/ddavis0485/tdub/_build/latest?definitionId=3&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/tdub/badge/?version=latest)](https://tdub.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/tdub?color=teal)](https://pypi.org/project/tdub/)
[![Python Version](https://img.shields.io/pypi/pyversions/tdub)](https://pypi.org/project/tdub/)

---

`tdub` is a Python project for handling some mid- and down-stream
steps in the ATLAS Run 2 tW inclusive cross section analysis. The
project provides a simple command line interface for performing
standard analysis tasks including:

- generating plots from
  [`TRExFitter`](https://gitlab.cern.ch/TRExStats/TRExFitter/)
- BDT hyperparameter optimization.
- training BDT models on our Monte Carlo.
- applying trained BDT models to our data and Monte Carlo.

For potentially finer-grained tasks the API is fully documented. The
API mainly provides quick and easy access to Pythonic representations
(i.e. dataframes or NumPy arrays) of our datasets (which of course
originate from [ROOT](https://root.cern/) files).

# tdub

[![Actions Status](https://github.com/douglasdavis/tdub/workflows/Linux/macOS/badge.svg)](https://github.com/douglasdavis/tdub/actions)
[![Documentation Status](https://readthedocs.org/projects/tdub/badge/?version=latest)](https://tdub.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/tdub?color=teal)](https://pypi.org/project/tdub/)
[![Python Version](https://img.shields.io/pypi/pyversions/tdub)](https://pypi.org/project/tdub/)

`tdub` is a Python project for handling some downstsream steps in the
ATLAS Run 2 tW inclusive cross section analysis. The project provides
a simple command line interface for performing standard analysis tasks
including:

- BDT feature selection and hyperparameter optimization.
- training BDT models on our Monte Carlo.
- applying trained BDT models to our data and Monte Carlo.
- generating plots from various raw sources (our ROOT files and
  classifier training output).
- generating plots from the output of
  [`TRExFitter`](https://gitlab.cern.ch/TRExStats/TRExFitter/).

For potentially finer-grained tasks the API is fully documented. The
API mainly provides quick and easy access to pythonic representations
(i.e. dataframes or NumPy arrays) of our datasets (which of course
originate from [ROOT](https://root.cern/) files), modularized ML
tasks, and a set of utilities tailored for interacting with our
specific datasets.

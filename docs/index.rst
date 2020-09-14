tdub
====

|Actions Status| |Documentation Status| |PyPI| |Python Version|

``tdub`` is a Python project for handling some downstsream steps in
the ATLAS Run 2 :math:`tW` inclusive cross section analysis. The
project provides a simple command line interface for performing
standard analysis tasks including:

- BDT feature selection and hyperparameter optimization.
- Training BDT models on our Monte Carlo.
- Applying trained BDT models to our data and Monte Carlo.
- Generating plots from various raw sources (our ROOT files and
  Classifier training output).
- Generating plots from the output of TRExFitter_.

For potentially finer-grained tasks the API is fully documented. The API
mainly provides quick and easy access to pythonic representations
(i.e. dataframes or NumPy arrays) of our datasets (which of course
originate from ROOT_ files), modularized ML
tasks, and a set of utilities tailored for interacting with our specific
datasets.

.. _TRExFitter: https://gitlab.cern.ch/TRExStats/TRExFitter/
.. _ROOT: https://root.cern/

.. |Actions Status| image:: https://github.com/douglasdavis/tdub/workflows/Linux/macOS/badge.svg
   :target: https://github.com/douglasdavis/tdub/actions
.. |Documentation Status| image:: https://readthedocs.org/projects/tdub/badge/?version=latest
   :target: https://tdub.readthedocs.io/
.. |PyPI| image:: https://img.shields.io/pypi/v/tdub?color=teal
   :target: https://pypi.org/project/tdub/
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/tdub
   :target: https://pypi.org/project/tdub/

.. toctree::
   :maxdepth: 1
   :caption: Command Line Interface

   cli

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_art.rst
   api_batch.rst
   api_config.rst
   api_data.rst
   api_frames.rst
   api_features.rst
   api_hist.rst
   api_math.rst
   api_ml_apply.rst
   api_ml_train.rst
   api_rex.rst
   api_root.rst

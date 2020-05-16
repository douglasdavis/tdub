====
tdub
====

``tdub`` is a Python project for handling some downstsream steps in
the ATLAS Run 2 :math:`tW` inclusive cross section analysis. The
project provides a simple command line interface for performing
standard analysis tasks including:

- BDT feature selection and hyperparameter optimization.
- training BDT models on our Monte Carlo.
- applying trained BDT models to our data and Monte Carlo.
- generating plots from various raw sources (our ROOT files and
  classifier training output).
- generating plots from the output of `TRExFitter
  <https://gitlab.cern.ch/TRExStats/TRExFitter/>`_.

For potentially finer-grained tasks the API is fully documented. The
API mainly provides quick and easy access to pythonic representations
(i.e. dataframes or NumPy arrays) of our datasets (which of course
originate from `ROOT <https://root.cern/>`_ files).

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

tdub
====

``tdub`` is a Python project for handling some downstsream steps in
the ATLAS Run 2 :math:`tW` inclusive cross section analysis. The
project provides a simple command line interface for performing
standard analysis tasks including:

- generating plots from the output of `TRExFitter
  <https://gitlab.cern.ch/TRExStats/TRExFitter/>`_.
- BDT hyperparameter optimization.
- training BDT models on our Monte Carlo.
- applying trained BDT models to our data and Monte Carlo.

For potentially finer-grained tasks the API is fully documented. The
API mainly provides quick and easy access to pythonic representations
(i.e. dataframes or NumPy arrays) of our datasets (which of course
originate from `ROOT <https://root.cern/>`_ files).

Navigation
----------

.. toctree::
   :maxdepth: 2

   cli.rst
   api.rst

Command Line Interface
----------------------

Top level CLI:

.. command-output:: tdub --help

regions2parquet
^^^^^^^^^^^^^^^

Turn a set of ROOT files into parquet output split into our standard
regions:

.. command-output:: tdub regions2parquet --help


stacks
^^^^^^

Generate matplotlib stacked histogram plots using TRExFitter output

.. command-output:: tdub stacks --help


pulls
^^^^^

Generate matplotlib pull plots using TRExFitter output

.. command-output:: tdub pulls --help

gpmin
^^^^^

Run a round of hyperparameter optimiziation using Gaussian Processes

.. command-output:: tdub gpmin --help

fold
^^^^

Run a :math:`k`-fold cross validation training

.. command-output:: tdub fold --help

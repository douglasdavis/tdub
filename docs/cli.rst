Command Line Interface
----------------------

Top level CLI:

.. command-output:: tdub --help

apply-gennpy
^^^^^^^^^^^^

Apply BDT folded training models to file(s), save model response to .npy file(s)

.. command-output:: tdub apply-gennpy --help

fsel-prepare
^^^^^^^^^^^^

Prepare for feature selection by creating dedicated parquet files

.. command-output:: tdub fsel-prepare --help

fsel-execute
^^^^^^^^^^^^

Execute a round of feature selection on a particular setup

.. command-output:: tdub fsel-execute --help

rex-pulls
^^^^^^^^^

Generate matplotlib pull plots using TRExFitter output

.. command-output:: tdub rex-pulls --help

rex-stacks
^^^^^^^^^^

Generate matplotlib stacked histogram plots using TRExFitter output

.. command-output:: tdub rex-stacks --help

train-fold
^^^^^^^^^^

Run a :math:`k`-fold cross validation training

.. command-output:: tdub train-fold --help

train-optimize
^^^^^^^^^^^^^^

Run a round of hyperparameter optimiziation using Gaussian Processes

.. command-output:: tdub train-optimize --help

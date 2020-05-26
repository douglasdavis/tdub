"""Analysis configuration module.

tdub is a Python library for physics analysis. Naturally some
properties of the analysis need to be easily modifiable for various
studies. This module houses a handful of variables that can be
modified simply by importing the module.

For example, we can call :py:func:`tdub.data.features_for` and expect
different results without changing the API usage, just changing the
configuration module ``FEATURESET_foo`` constants::

  >>> from tdub.data import features_for
  >>> features_for("2j2b")
  ['mass_lep1jet1', 'mass_lep2jet1', 'pT_jet2', ...]
  >>> import tdub.config
  >>> tdub.config.FEATURESET_2j2b = ["pT_jet1", "met"]
  >>> features_for("2j2b")
  ['pT_jet1', 'met']

Similarly, we can modify the selection via this module::

  >>> from tdub.data import selection_for
  >>> selection_for("2j2b")
  '(reg2j2b == True) & (OS == True)'
  >>> import tdub.config
  >>> tdub.config.SELECTION_2j2b = "(reg2j2b == True) & (OS == True) & (mass_lep1jet1 < 155)"
  >>> selection_for("2j2b")
  '(reg2j2b == True) & (OS == True) & (mass_lep1jet1 < 155)'

"""


SELECTION_1j1b = "(reg1j1b == True) & (OS == True)"
"""
str: The numexpr selection string for the 1j1b region.
"""


SELECTION_2j1b = "(reg2j1b == True) & (OS == True)"
"""
str: The numexpr selection string for the 2j1b region.
"""


SELECTION_2j2b = "(reg2j2b == True) & (OS == True)"
"""
str: The numexpr selection string for the 2j2b region.
"""


FEATURESET_1j1b_TMVA = sorted(
    [
        "pTsys_lep1lep2jet1met",
        "mass_lep2jet1",
        "mass_lep1jet1",
        "pTsys_lep1lep2",
        "deltaR_lep2_jet1",
        "nsoftjets",
        "deltaR_lep1_lep2",
        "deltapT_lep1_jet1",
        "mT_lep2met",
        "nsoftbjets",
        "cent_lep1lep2",
        "pTsys_lep1lep2jet1",
    ],
    key=str.lower,
)
"""
list(str): List of features we use for classifiers in the 1j1b region.
"""


FEATURESET_2j1b_TMVA = sorted(
    [
        "mass_lep1jet2",
        "psuedoContTagBin_jet1",
        "mass_lep1jet1",
        "mass_lep2jet1",
        "mass_lep2jet2",
        "pTsys_lep1lep2jet1jet2met",
        "psuedoContTagBin_jet2",
        "pT_jet2",
    ],
    key=str.lower,
)
"""
list(str): List of features we use for classifiers in the 2j1b region.
"""


FEATURESET_2j2b_TMVA = sorted(
    [
        "mass_lep1jet2",
        "mass_lep1jet1",
        "deltaR_lep1_jet1",
        "mass_lep2jet1",
        "pTsys_lep1lep2met",
        "pT_jet2",
        "mass_lep2jet2",
    ],
    key=str.lower,
)
"""
list(str): List of features we use for classifiers in the 2j2b region.
"""


FEATURESET_1j1b = sorted(
    [
        "cent_lep1lep2",
        "deltapT_lep1_lep2",
        "mass_lep1jet1",
        "mass_lep2jet1",
        "mass_lep2jet1met",
        "mT_jet1met",
        # "nsoftbjets",
        # "nsoftjets",
        "pT_jetS1",
        "pTsys_jet1met",
        "pTsys_lep1lep2",
        "pTsys_lep1lep2jet1met",
    ],
    key=str.lower,
)
"""
list(str): List of features we use for classifiers in the 1j1b region.
"""


FEATURESET_2j1b = sorted(
    [
        # "deltaR_lep1lep2_jet1jet2met",
        # "HT_jet1jet2",
        # "mass_lep2jet1",
        # "pT_jet2",
        "pTsys_lep1lep2",
        "mass_lep1jet1",
        "mass_lep1jet2",
        # "mass_lep2jet2",
        # "deltaR_lep1_jet1",
        "deltaR_lep2_jet1",
        # "psuedoContTagBin_jet1",
        # "psuedoContTagBin_jet2",
        "pTsys_lep1lep2jet1met",
        "pTsys_lep1lep2jet1jet2met",
        "HTratio_lep1lep2_lep1lep2jet1met",
        "HTratio_lep1lep2_lep1lep2jet1jet2met"
        # "pTHTratio_lep1lep2jet1met",
        # "pTHTratio_lep1lep2jet1jet2met",
    ],
    key=str.lower,
)
"""
list(str): List of features we use for classifiers in the 2j1b region.
"""


FEATURESET_2j2b = sorted(
    [
        # "deltaR_jet1_jet2",
        "mass_lep1jet1",
        "mass_lep1jet2",
        "mass_lep2jet1",
        "mass_lep2jet2",
        "pT_jet2",
        "pTsys_jet1jet2",
        "pTsys_lep1lep2met",
    ],
    key=str.lower,
)
"""
list(str): List of features we use for classifiers in the 2j2b region.
"""


AVOID_IN_CLF = sorted(
    [
        "tmva_bdt_response",
        "phi_lep1",
        "phi_lep2",
        "phi_jet1",
        "phi_jet2",
        "eta_met",
        "eta_jetL1",
        "eta_jetS1",
        "sumet",
        "mass_jet1",
        "mass_jet2",
        "mass_jetF",
        "mass_jetL1",
        "mass_jetS1",
        "E_jetL1",
        "E_jetS1",
        "E_jet1",
        "E_jet2",
        "pT_lep3",
        "pT_jetL1",
        "nbjets",
        "njets",
    ],
    key=str.lower,
)
"""
list(str): List of features to avoid in classifiers.
"""


AVOID_IN_CLF_1j1b = sorted(["_nothing"])
"""
list(str): List of features to avoid specifically in 1j1b classifiers.
"""


AVOID_IN_CLF_2j1b = sorted(
    ["HT_jet1jet2", "deltaR_lep1lep2_jet1jet2met", "mass_lep2jet1", "pT_jet2"]
)
"""
list(str): List of features to avoid specifically in 2j1b classifiers.
"""


AVOID_IN_CLF_2j2b = sorted(["deltaR_jet1_jet2"])
"""
list(str): List of features to avoid specifically in 2j2b classifiers.
"""


DEFAULT_SCAN_PARAMETERS = {
    "max_depth": [4, 5, 6, 7, 8],
    "num_leaves": [20, 31, 64],
    "learning_rate": [0.07, 0.1, 0.2],
    "min_child_samples": [50, 60, 80, 120, 180],
    "n_estimators": [250],
}
"""
dict(str, list): The default grid to perform a parameter scan.
"""


PLOTTING_META_TABLE = None
"""
dict, optional: Plotting metadata table.
"""

PLOTTING_LOGY = []
"""
list(str): Plots (defined as TRExFitter Regions) to use log scale.
"""

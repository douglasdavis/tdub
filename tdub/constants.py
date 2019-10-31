"""
a module to house some constants
"""

SELECTION_1j1b = "(reg1j1b == True) & (OS == True)"
"""
str: The pandas flavor selection string for the 1j1b region
"""


SELECTION_2j1b = "(reg2j1b == True) & (OS == True)"
"""
str: The pandas flavor selection string for the 2j1b region
"""


SELECTION_2j2b = "(reg2j2b == True) & (OS == True)"
"""
str: The pandas flavor selection string for the 2j2b region
"""


FEATURESET_1j1b = sorted(
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
list(str): list of features we use for classifiers in the 1j1b region
"""


FEATURESET_2j1b = sorted(
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
list(str): list of features we use for classifiers in the 2j1b region
"""


FEATURESET_2j2b = sorted(
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
list(str): list of features we use for classifiers in the 2j2b region
"""


AVOID_IN_CLF = sorted(
    [
        "bdt_response",
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
list(str): list of features to avoid in classifiers
"""
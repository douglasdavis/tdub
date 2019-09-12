from enum import Enum

__all__ = [
    "Region",
    "SEL_1j1b",
    "SEL_2j1b",
    "SEL_2j2b",
    "FSET_1j1b",
    "FSET_2j1b",
    "FSET_2j2b",
]


class Region(Enum):
    """A simple enum class for easily using region information

    Attributes
    ----------
    r1j1b
        Our ``1j1b`` region
    r1j1b
        Our ``2j1b`` region
    r2j1b = 1
        Our ``2j2b`` region

    Examples
    --------

    Using this enum for grabing the ``2j2b`` region from a set of
    files:

    >>> from tdub.regions import Region
    >>> from tdub.frames import specific_dataframe
    >>> sdf = specific_dataframe(files, Region.r2j2b)

    """
    r1j1b = 0
    r2j1b = 1
    r2j2b = 2


SEL_1j1b = "(reg1j1b == True) & (OS == True)"
SEL_2j1b = "(reg2j1b == True) & (OS == True)"
SEL_2j2b = "(reg2j2b == True) & (OS == True)"

FSET_1j1b = [
    "pTsys_lep1lep2jet1met",
    "mass_lep2jet1",
    "mass_lep1jet1",
    "pTsys_lep1lep2",
    "deltaR_lep2_jet1",
    "nloosejets",
    "deltaR_lep1_lep2",
    "deltapT_lep1_jet1",
    "mT_lep2met",
    "nloosebjets",
    "cent_lep1lep2",
    "pTsys_lep1lep2jet1",
]

FSET_2j1b = [
    "mass_lep1jet2",
    "psuedoContTagBin_jet1",
    "mass_lep1jet1",
    "mass_lep2jet1",
    "mass_lep2jet2",
    "pTsys_lep1lep2jet1jet2met",
    "psuedoContTagBin_jet2",
    "pT_jet2",
]

FSET_2j2b = [
    "mass_lep1jet2",
    "mass_lep1jet1",
    "deltaR_lep1_jet1",
    "mass_lep2jet1",
    "pTsys_lep1lep2met",
    "pT_jet2",
    "mass_lep2jet2",
]

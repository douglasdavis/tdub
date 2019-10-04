from __future__ import annotations

from enum import Enum

__all__ = [
    "Region",
    "SELECTION_1j1b",
    "SELECTION_2j1b",
    "SELECTION_2j2b",
    "SELECTIONS",
    "FEATURESET_1j1b",
    "FEATURESET_2j1b",
    "FEATURESET_2j2b",
    "FEATURESETS",
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

    @staticmethod
    def from_str(s: str) -> Region:
        """get enum value for the given string

        This function supports three ways to define a region; prefixed
        with "r", prefixed with "reg", or no prefix at all. For
        example, ``Region.r2j2b`` can be retrieved like so:

        - ``Region.from_str("r2j2b")``
        - ``Region.from_str("reg2j2b")``
        - ``Region.from_str("2j2b")``

        Parameters
        ----------
        s : str
           string representation of the desired region

        Returns
        -------
        Region
           the enum version

        Examples
        --------

        >>> from tdub.regions import Region
        >>> Region.from_str("1j1b")
        <Region.r1j1b: 0>

        """
        if s.startswith("reg"):
            rsuff = s.split("reg")[-1]
            return Region.from_str(rsuff)
        elif s.startswith("r"):
            return Region[s]
        else:
            if s == "2j2b":
                return Region.r2j2b
            elif s == "2j1b":
                return Region.r2j1b
            elif s == "1j1b":
                return Region.r1j1b
            else:
                raise ValueError(f"{s} doesn't correspond to a Region")

    def __str__(self) -> str:
        return self.name[1:]


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


SELECTIONS = {
    Region.r1j1b: SELECTION_1j1b,
    Region.r2j1b: SELECTION_2j1b,
    Region.r2j2b: SELECTION_2j2b,
}
"""
dict(Region, str): key-value pairs for regions to their selection string
"""


FEATURESET_1j1b = sorted(
    [
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
    ]
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
    ]
)
"""
list(str): list of features we use for classifiers in the 2j2b region
"""


FEATURESETS = {
    Region.r1j1b: FEATURESET_1j1b,
    Region.r2j1b: FEATURESET_2j1b,
    Region.r2j2b: FEATURESET_2j2b,
}
"""
dict(Region, list(str)): key-value pairs for regions to their feature set
"""

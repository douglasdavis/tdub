import pytest

from tdub.utils import Region

from tdub.branches import (
    categorize_branches,
    get_selection,
    minimal_branches,
    numexpr_selection,
    root_selection,
)

from tdub.constants import (
    FEATURESET_1j1b,
    FEATURESET_2j1b,
    FEATURESET_2j2b,
    SELECTION_1j1b,
    SELECTION_2j1b,
    SELECTION_2j2b,
    AVOID_IN_CLF_1j1b,
    AVOID_IN_CLF_2j1b,
    AVOID_IN_CLF_2j2b,
)

def test_categorize_branches():
    branches = [
        "pT_lep1",
        "pT_lep2",
        "weight_nominal",
        "weight_sys_jvt",
        "reg2j2b",
        "reg1j1b",
        "OS",
        "elmu",
    ]
    cb = categorize_branches(branches)
    assert cb["meta"] == sorted(["OS", "reg1j1b", "reg2j2b", "elmu"], key=str.lower)
    assert cb["kinematics"] == sorted(["pT_lep1", "pT_lep2"], key=str.lower)
    assert cb["weights"] == sorted(["weight_nominal", "weight_sys_jvt"], key=str.lower)

    cb = categorize_branches("tests/test_data/testfile1.root")
    assert "reg2j1b" in cb["meta"]
    assert "pT_lep1" not in cb["meta"]
    assert "pT_lep1" not in cb["weights"]
    assert "pT_lep2" in cb["kinematics"]

    from pathlib import PosixPath

    cb = categorize_branches(PosixPath("tests/test_data/testfile2.root"))
    assert "reg2j1b" in cb["meta"]
    assert "pT_lep1" not in cb["meta"]
    assert "pT_lep1" not in cb["weights"]
    assert "pT_lep2" in cb["kinematics"]

def test_get_selection():
    assert get_selection("reg2j1b") == SELECTION_2j1b
    assert get_selection("2j2b") == SELECTION_2j2b
    assert get_selection(Region.r1j1b) == SELECTION_1j1b
    assert get_selection("r2j1b") == SELECTION_2j1b
    assert get_selection(Region.r2j2b) == SELECTION_2j2b
    assert get_selection("reg1j1b") == SELECTION_1j1b
    assert get_selection("1j1b") == SELECTION_1j1b


def test_numexpr_selection():
    sel = "reg1j1b == true && OS == true && mass_lep1jet1 < 155"
    newsel = numexpr_selection(sel)
    assert newsel == "(reg1j1b == True) & (OS == True) & (mass_lep1jet1 < 155)"
    sel = "(reg1j1b == True) & (OS == True) & (mass_lep1jet1 < 155)"
    newsel = numexpr_selection(sel)
    assert newsel == "(reg1j1b == True) & (OS == True) & (mass_lep1jet1 < 155)"


def test_root_selection():
    sel = "(reg1j1b == True) & (OS == True) & (mass_lep1jet1 < 155)"
    newsel = root_selection(sel)
    assert "(reg1j1b == true) && (OS == true) && (mass_lep1jet1 < 155)"
    sel = "(reg1j1b == true) && (OS == true) && (mass_lep1jet1 < 155)"
    newsel = root_selection("(reg1j1b == true) && (OS == true) && (mass_lep1jet1 < 155)")
    assert newsel == "(reg1j1b == true) && (OS == true) && (mass_lep1jet1 < 155)"


def test_minimal_branches():
    sel = "(reg1j1b == True) & (OS == True) & (mass_lep1jet1 < 155)"
    varsin = set(("reg1j1b", "OS", "mass_lep1jet1"))
    assert minimal_branches(sel) == varsin
    sel = "reg1j1b == true && OS == true && mass_lep1jet1 < 155"
    varsin = set(("reg1j1b", "OS", "mass_lep1jet1"))
    assert minimal_branches(sel) == varsin

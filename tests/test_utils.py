from tdub.utils import (
    SampleInfo,
    Region,
    categorize_branches,
    get_features,
    get_branches,
    FEATURESET_1j1b,
    FEATURESET_2j1b,
    FEATURESET_2j2b,
)

import pytest


def test_sample_info():
    s1 = "ttbar_410472_AFII_MC16d_nominal.root"
    s2 = "ttbar_410472_AFII_MC16e_EG_SCALE_ALL__1up"
    s3 = "tW_DS_410657_FS_MC16d_EG_RESOLUTION_ALL__1down"
    s4 = "Diboson_364255_FS_MC16e_nominal.root"
    s5 = "Data17_data17_Data_Data_nominal.root"
    s6 = "MCNP_Zjets_364137_FS_MC16a_nominal.root"
    s7 = "tW_DR_410648_AFII_MC16d_nominal.bdtresp.npy"

    si1 = SampleInfo(s1)
    si2 = SampleInfo(s2)
    si3 = SampleInfo(s3)
    si4 = SampleInfo(s4)
    si5 = SampleInfo(s5)
    si6 = SampleInfo(s6)
    si7 = SampleInfo(s7)

    assert si1.phy_process == "ttbar"
    assert si1.dsid == 410472
    assert si1.sim_type == "AFII"
    assert si1.campaign == "MC16d"
    assert si1.tree == "nominal"

    assert si2.phy_process == "ttbar"
    assert si2.dsid == 410472
    assert si2.sim_type == "AFII"
    assert si2.campaign == "MC16e"
    assert si2.tree == "EG_SCALE_ALL__1up"

    assert si3.phy_process == "tW_DS"
    assert si3.dsid == 410657
    assert si3.sim_type == "FS"
    assert si3.campaign == "MC16d"
    assert si3.tree == "EG_RESOLUTION_ALL__1down"

    assert si4.phy_process == "Diboson"
    assert si4.dsid == 364255
    assert si4.sim_type == "FS"
    assert si4.campaign == "MC16e"
    assert si4.tree == "nominal"

    assert si5.phy_process == "Data"
    assert si5.dsid == 0
    assert si5.sim_type == "Data"
    assert si5.campaign == "Data"
    assert si5.tree == "nominal"

    assert si6.phy_process == "MCNP"
    assert si6.dsid == 364137
    assert si6.sim_type == "FS"
    assert si6.campaign == "MC16a"
    assert si6.tree == "nominal"

    assert si7.phy_process == "tW_DR"
    assert si7.dsid == 410648
    assert si7.sim_type == "AFII"
    assert si7.campaign == "MC16d"
    assert si7.tree == "nominal"


def test_bad_sample_info():
    bad = "tW_DR_410_bad"
    with pytest.raises(ValueError) as err:
        sib = SampleInfo(bad)
    assert str(err.value) == "tW_DR_410_bad cannot be parsed by SampleInfo regex"


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


def test_Region_from_str():
    assert Region.from_str("2j2b") == Region.r2j2b
    assert Region.from_str("1j1b") == Region.r1j1b
    assert Region.from_str("2j1b") == Region.r2j1b

    assert Region.from_str("r2j2b") == Region.r2j2b
    assert Region.from_str("r1j1b") == Region.r1j1b
    assert Region.from_str("r2j1b") == Region.r2j1b

    assert Region.from_str("reg2j2b") == Region.r2j2b
    assert Region.from_str("reg1j1b") == Region.r1j1b
    assert Region.from_str("reg2j1b") == Region.r2j1b


def test_get_features():
    assert get_features("reg2j1b") == FEATURESET_2j1b
    assert get_features("2j2b") == FEATURESET_2j2b
    assert get_features(Region.r1j1b) == FEATURESET_1j1b
    assert get_features("r2j1b") == FEATURESET_2j1b
    assert get_features(Region.r2j2b) == FEATURESET_2j2b
    assert get_features("reg1j1b") == FEATURESET_1j1b
    assert get_features("1j1b") == FEATURESET_1j1b

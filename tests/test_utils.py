import pytest

from tdub.utils import (
    SampleInfo,
    Region,
    files_for_tree,
    get_features,
    get_avoids,
)

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
        SampleInfo(bad)
    assert str(err.value) == "tW_DR_410_bad cannot be parsed by SampleInfo regex"


def test_bad_files_for_tree():
    with pytest.raises(ValueError) as err:
        files_for_tree("a", "b", "c")
    assert (
        str(err.value)
        == "bad sample_prefix 'b', must be one of: ['tW_DR', 'tW_DS', 'ttbar']"
    )


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


def test_get_avoids():
    assert get_avoids("reg1j1b") == AVOID_IN_CLF_1j1b
    assert get_avoids("r1j1b") == AVOID_IN_CLF_1j1b
    assert get_avoids("1j1b") == AVOID_IN_CLF_1j1b
    assert get_avoids(Region.r1j1b) == AVOID_IN_CLF_1j1b
    assert get_avoids("reg2j1b") == AVOID_IN_CLF_2j1b
    assert get_avoids("r2j1b") == AVOID_IN_CLF_2j1b
    assert get_avoids("2j1b") == AVOID_IN_CLF_2j1b
    assert get_avoids(Region.r2j1b) == AVOID_IN_CLF_2j1b
    assert get_avoids("reg2j2b") == AVOID_IN_CLF_2j2b
    assert get_avoids("r2j2b") == AVOID_IN_CLF_2j2b
    assert get_avoids("2j2b") == AVOID_IN_CLF_2j2b
    assert get_avoids(Region.r2j2b) == AVOID_IN_CLF_2j2b

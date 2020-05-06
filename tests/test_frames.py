from pathlib import PosixPath

from tdub.constants import AVOID_IN_CLF
from tdub.frames import iterative_selection, drop_avoid
from tdub.branches import minimal_branches

test_file_root = PosixPath(__file__).parent / "test_data"


def test_exclude_avoids():
    files = [
        str(test_file_root / "testfile1.root"),
        str(test_file_root / "testfile2.root"),
        str(test_file_root / "testfile3.root"),
    ]
    df = iterative_selection(files, "(reg1j1b == True)", exclude_avoids=True)
    cols = set(df.columns)
    avoid = set(AVOID_IN_CLF)
    assert len(cols & avoid) == 0

    df = iterative_selection(
        files,
        "(reg1j1b == True)",
        exclude_avoids=True,
        keep_category="kinematics",
    )
    cols = set(df.columns)
    avoid = set(AVOID_IN_CLF)
    assert len(cols & avoid) == 0


def test_drop_avoid():
    files = [
        str(test_file_root / "testfile1.root"),
        str(test_file_root / "testfile2.root"),
        str(test_file_root / "testfile3.root"),
    ]
    df = iterative_selection(files, "(reg1j1b == True)")
    df.drop_avoid()
    avoid = set(AVOID_IN_CLF)
    cols = set(df.columns)
    assert len(cols & avoid) == 0


def test_drop_jet2():
    files = [str(test_file_root / "testfile1.root"), str(test_file_root / "testfile3.root")]
    df = iterative_selection(files, "(OS == True)")
    j2s = [col for col in df.columns if "jet2" in col]
    df.drop_jet2()
    for j in j2s:
        assert j not in df.columns


def test_selection_augmented():
    files = [str(test_file_root / "testfile1.root"), str(test_file_root / "testfile3.root")]
    df = iterative_selection(files, "(OS == True) & (reg1j1b == True) & (mass_lep1jet1 < 155)")
    sel_vars = set(minimal_branches(df.selection_used))
    manual = {"OS", "reg1j1b", "mass_lep1jet1"}
    assert sel_vars == manual
    assert df.selection_used == "(OS == True) & (reg1j1b == True) & (mass_lep1jet1 < 155)"


def test_selection_strings():
    files = [str(test_file_root / "testfile1.root"), str(test_file_root / "testfile3.root")]
    root_sel1 = "OS == 1 && reg2j2b == 1 && mass_lep1jet1 < 155"
    nume_sel1 = "(OS == 1) & (reg2j2b == 1) & (mass_lep1jet1 < 155)"
    root_sel2 = "OS == true && reg2j2b == true && mass_lep1jet1 < 155"
    nume_sel2 = "(OS == True) & (reg2j2b == True) & (mass_lep1jet1 < 155)"
    df_r_sel1 = iterative_selection(files, root_sel1)
    df_r_sel2 = iterative_selection(files, root_sel2)
    df_n_sel1 = iterative_selection(files, nume_sel1)
    df_n_sel2 = iterative_selection(files, nume_sel2)
    assert df_r_sel1.equals(df_r_sel2)
    assert df_r_sel1.equals(df_n_sel1)
    assert df_r_sel1.equals(df_n_sel2)

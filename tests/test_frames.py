from pathlib import PosixPath

from tdub.frames import iterative_selection, drop_avoid
from tdub.utils import AVOID_IN_CLF

test_file_root = PosixPath(__file__).parent / "test_data"


def test_ignore_avoid():
    files = [
        str(test_file_root / "testfile1.root"),
        str(test_file_root / "testfile2.root"),
        str(test_file_root / "testfile3.root"),
    ]
    df = iterative_selection(files, "(reg1j1b == True)", ignore_avoid=True, concat=True)
    cols = set(df.columns)
    avoid = set(AVOID_IN_CLF)
    assert len(cols & avoid) == 0

    df = iterative_selection(
        files,
        "(reg1j1b == True)",
        ignore_avoid=True,
        concat=True,
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
    df = iterative_selection(files, "(reg1j1b == True)", concat=True)
    drop_avoid(df)
    avoid = set(AVOID_IN_CLF)
    cols = set(df.columns)
    assert len(cols & avoid) == 0

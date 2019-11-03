"""
Module to help running things on BNL machines
"""

from __future__ import annotations

# stdlib
import pathlib
import shutil

# tdub
from tdub.utils import SampleInfo


CONDOR_HEADER = """Universe        = vanilla
notification    = Error
notify_user     = ddavis@phy.duke.edu
x509userproxy   = $ENV(X509_USER_PROXY)
GetEnv          = True
Executable      = {tdub_exe_path}
Output          = job.out.apply-gennpy.$(cluster).$(process)
Error           = job.err.apply-gennpy.$(cluster).$(process)
Log             = /tmp/ddavis/log.$(cluster).$(process)
request_memory  = 2.0G
"""


def get_tdub_exe() -> str:
    """get the tdub executable

    Returns
    -------
    str
       full path of the tdub executable

    """
    return shutil.which("tdub")


def parse_samples(bnl_path: Union[str, os.PathLike]) -> List[str]:
    """get a list of all ROOT samples in a directory on BNL

    Parameters
    ----------
    bnl_path : str or os.PathLike
       the

    Returns
    -------
    list(str)
       all sample in the directory
    """
    path = pathlib.PosixPath(bnl_path).resolve()
    return [p for p in path.iterdir() if (p.is_file() and p.suffix == ".root")]


def gen_submit_script(
    input_dir: Union[str, os.PathLike],
    fold_dirs: List[Union[str, os.PathLike]],
    output_dir: Union[str, os.PathLike],
    arr_name: str = "bdt_res",
    script_name: str = "apply-gennpy.condor.submit",
) -> None:
    """generate a condor submission script

    Parameters
    ----------
    input_dir : str or os.PathLike
       directory containing ROOT files
    fold_dirs : list(str) or list(os.PathLike)
       list of fold result directories
    output_dir : str or os.PathLike
       directory to store output .npy files
    arr_name : str
       name for the calculated result array
    script_name : str
       name for the output submissions script

    """
    output_script = pathlib.PosixPath(script_name)
    header = CONDOR_HEADER.format(tdub_exe_path=get_tdub_exe())
    folds = [str(pathlib.PosixPath(fold).resolve()) for fold in fold_dirs]
    folds = " ".join(folds)
    out = pathlib.PosixPath(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    outdir = str(out.resolve())

    with output_script.open("w") as f:
        print(header, file=f)
        opt = "apply-gennpy"
        for sample in parse_samples(input_dir):
            line = f"{opt} --single-file {sample.resolve()} -f {folds} -n {arr_name} -o {outdir}"
            print(f"Arguments = {line}\nQueue\n\n", file=f)

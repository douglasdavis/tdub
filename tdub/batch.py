"""
Module to help running batch jobs
"""

# stdlib
import logging
import pathlib
import os
from typing import List, Union, Optional

log = logging.getLogger(__name__)

BNL_CONDOR_HEADER = """
Universe        = vanilla
notification    = Error
notify_user     = ddavis@phy.duke.edu
GetEnv          = True
Executable      = {tdub_exe_path}
Output          = logs/job.out.apply-gennpy.$(cluster).$(process)
Error           = logs/job.err.apply-gennpy.$(cluster).$(process)
Log             = /tmp/ddavis/log.$(cluster).$(process)
request_memory  = 2.0G
"""


def parse_samples(usatlasdata_path: Union[str, os.PathLike]) -> List[pathlib.Path]:
    """get a list of all ROOT samples in a directory on BNL

    Parameters
    ----------
    usatlasdata_path : str or os.PathLike
       the path to the datasets inside the ``usatlasdata`` area.

    Returns
    -------
    list(str)
       all sample in the directory

    Examples
    --------

    >>> from tdub.batch import parse_samples
    >>> samples = parse_samples("/path/to/usaltasdata/some_root_files")

    """
    log.info(f"parsing samples in {usatlasdata_path}")
    path = pathlib.PosixPath(usatlasdata_path).resolve()
    return [p for p in path.iterdir() if (p.is_file() and p.suffix == ".root")]


def gen_apply_npy_script(
    exe: str,
    input_dir: Union[str, os.PathLike],
    fold_dirs: List[Union[str, os.PathLike]],
    output_dir: Union[str, os.PathLike],
    arr_name: str = "bdt_res",
    script_name: Optional[str] = None,
) -> None:
    """generate a condor submission script

    Parameters
    ----------
    exe : str
       executable string
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

    Examples
    --------

    >>> from tdub.batch import gen_submit_script
    >>> gen_submit_script(
    ...    "/path/to/usaltasdata/some_root_files",
    ...    ["fold1", "fold2", "fold3"],
    ...    "/path/to/npy_output_dir",
    ...    "bdt_res",
    ...    "my.condor.sub.script",
    ... )
    >>> exit()
    $ condor_submit my.condor.sub.script

    """
    if script_name is None:
        script_name = "apply-gen-npy.condor.sub"
    log_dir = pathlib.PosixPath(os.getcwd()) / "logs"
    log_dir.mkdir(exist_ok=True)
    output_script = pathlib.PosixPath(script_name)
    header = BNL_CONDOR_HEADER.format(tdub_exe_path=exe)
    folds = " -f ".join([str(pathlib.PosixPath(fold).resolve()) for fold in fold_dirs])
    action = "apply-gen-npy"
    outdir = pathlib.PosixPath(output_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    with output_script.open("w") as f:
        print(header, file=f)
        for sample in parse_samples(input_dir):
            opts = f"--single {sample.resolve()} -f {folds} -n {arr_name} -o {outdir.resolve()}"
            print(f"Arguments = {action} {opts}\nQueue\n\n", file=f)
    log.info(f"generated condor submission script {output_script}")

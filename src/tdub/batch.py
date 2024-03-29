"""Module to help running batch jobs."""

from __future__ import annotations

# stdlib
import logging
import os
import shutil
import subprocess
from typing import TextIO, Union
from pathlib import PosixPath

log = logging.getLogger(__name__)

BNL_CONDOR_PREAMBLE = """## -*- dear emacs, mode: conf -*-
## Condor Submission; generated by tdub.batch

Universe        = {universe}
notification    = {notification}
notify_user     = {email}
Executable      = {exe}
Output          = {workspace}/out/$(cluster).$(process)
Error           = {workspace}/err/$(cluster).$(process)
Log             = {workspace}/log/$(cluster).$(process)
request_memory  = {memory}
GetEnv          = {getenv}
"""


def create_condor_workspace(
    name: str | os.PathLike, overwrite: bool = False
) -> PosixPath:
    """Create a condor workspace given a name.

    This will create a new directory containing `log`, `out`, and
    `err` directories inside. The `workspace` argument to the
    :py:func:`~condor_preamble` function assumes creation of a workspace
    via this function.

    Missing parent directories will always be created.

    Parameters
    ----------
    name : str or os.PathLike
        the desired filesystem path for the workspace
    overwrite: bool
        if True, an existing workspace will be overwritten

    Raises
    ------
    OSError
        if the filesystem path exists and exist_ok is False

    Returns
    -------
    pathlib.PosixPath
        filesystem path to the workspace

    Examples
    --------
    >>> import tdub.batch as tb
    >>> import shutil
    >>> ws = tb.create_condor_workspace("./some/ws")
    >>> with open(ws / "condor.sub", "w") as f:
    ...     preamble = tb.condor_preamble(ws, shutil.which("tdub"), to_file=f)
    ...     tb.add_condor_arguments("train-single ......", f)

    """
    ws = PosixPath(name).resolve()
    if overwrite and ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(exist_ok=False, parents=True)
    (ws / "log").mkdir()
    (ws / "err").mkdir()
    (ws / "out").mkdir()
    return ws


def condor_preamble(
    workspace: str | os.PathLike,
    exe: str | os.PathLike,
    universe: str = "vanilla",
    memory: str = "2GB",
    email: str = "ddavis@phy.duke.edu",
    notification: str = "Error",
    getenv: str = "True",
    to_file: TextIO = None,
    **kwargs,
) -> str:
    """Create the preamble of a condor submission script.

    Extra kwargs create additional preamble entries. See the HTCondor
    documentation for more details on all parameters.

    Parameters
    ----------
    workspace : str or os.PathLike
        the filesystem directry where the workspace is
    exe : str or os.PathLike
        the path of the executable that condor will run
    universe : str
        the HTCondor universe
    memory : str
        the requested memory
    email : str
        the email to send updates to (if any)
    notification : str
        the condor notification argument
    to_file : typing.TextIO, optional
        if not None, write the string to file

    Returns
    -------
    str
        the submission script preamble

    Examples
    --------
    >>> import tdub.batch as tb
    >>> import shutil
    >>> ws = tb.create_condor_workspace("./some/ws")
    >>> with open(ws / "condor.sub", "w") as f:
    ...     preamble = tb.condor_preamble(ws, shutil.which("tdub"), to_file=f)
    ...     tb.add_condor_arguments("train-single ......", f)

    """
    res = BNL_CONDOR_PREAMBLE.format(
        universe=universe,
        workspace=os.path.abspath(workspace),
        exe=exe,
        memory=memory,
        email=email,
        notification=notification,
        getenv=getenv,
    )
    for k, v in kwargs.items():
        res += f"{k:<15} = {v}\n"
    if to_file is not None:
        print(res, file=to_file)
    return res


def add_condor_arguments(arguments: str, to_file: TextIO) -> None:
    """Add an arguments line to a condor submission script.

    the `arguments` argument is prefixed with `"Arguments = "` and
    written to `to_file`.

    Parameters
    ----------
    arguments : str
        the arguments line
    to_file : typing.TextIO
        the open file stream

    Examples
    --------
    >>> import tdub.batch as tb
    >>> import shutil
    >>> ws = tb.create_condor_workspace("./some/ws")
    >>> with open(ws / "condor.sub", "w") as f:
    ...     preamble = tb.condor_preamble(ws, shutil.which("tdub"), to_file=f)
    ...     tb.add_condor_arguments("train-single ......", f)

    """
    to_file.write("\n")
    to_file.write(f"Arguments = {arguments}\n")
    to_file.write("Queue\n")


def condor_submit(workspace: str | os.PathLike) -> None:
    """Execute condor_submit on the condor.sub file in a workspace.

    Parameters
    ----------
    workspace : str or os.PathLike
        the workspace containing the condor.sub file

    """
    ws = PosixPath(workspace).resolve()
    proc = subprocess.Popen(
        ["condor_submit", str(ws / "condor.sub")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate()
    try:
        log_out = out.decode("utf-8")
    except AttributeError:
        pass
    log.info(log_out)

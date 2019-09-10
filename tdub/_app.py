import argparse
import logging

import dask
from dask.distributed import Client, Lock
from dask.utils import SerializableLock
from dask.dataframe import to_parquet

from tdub.frames import stdregion_dataframes
from tdub.art import run_pulls, run_stacks


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(prog="tdub", description="tee-double-you CLI")
    subparsers = parser.add_subparsers(dest="action", help="Action")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--debug", action="store_true", help="set logging level to debug")

    regions2parquet = subparsers.add_parser("regions2parquet", help="create parquet output for individual regions", parents=[common_parser])
    regions2parquet.add_argument("files", type=str, nargs="+", help="input ROOT files")
    regions2parquet.add_argument("prefix", type=str, help="output file name prefix")
    regions2parquet.add_argument("-b","--branches", type=str, nargs="+", default=None, help="Branches")
    regions2parquet.add_argument("-t","--tree-name", type=str, default="WtLoop_nominal", help="ROOT tree name")

    stacks = subparsers.add_parser("stacks", help="create matplotlib stack plots from TRExFitter output", parents=[common_parser])
    stacks.add_argument("workspace", type=str, help="TRExFitter workspace")
    stacks.add_argument("-o", "--out-dir", type=str, help="output directory for plots")
    stacks.add_argument("--lumi", type=str, default="139", help="Integrated lumi. for text")
    stacks.add_argument("--do-postfit", action="store_true", help="produce post fit plots as well")
    stacks.add_argument("--skip-regions", type=str, default=None, help="skip regions based on regex")
    stacks.add_argument("--band-style", type=str, choices=["hatch", "shade"], default="hatch", help="band art")
    stacks.add_argument("--legend-ncol", type=int, choices=[1, 2], default=1, help="number of legend columns")

    pulls = subparsers.add_parser("pulls", help="create matplotlib pull plots from TRExFitter output", parents=[common_parser])
    pulls.add_argument("workspace", type=str, help="TRExFitter workspace")
    pulls.add_argument("config", type=str, help="TRExFitter config")
    pulls.add_argument("-o", "--out-dir", type=str, help="output directory")
    pulls.add_argument("--no-text", action="store_true", help="don't print values on plots")

    # fmt: on
    return (parser.parse_args(), parser)


def _parquet_regions(args, log):
    import numexpr
    numexpr.set_num_threads(1)
    frames = stdregion_dataframes(args.files, args.tree_name, args.branches)
    log.info("Executing queries:")
    for k, v in frames.items():
        log.info(f"  - {v.name}: {v.selection}")
    for name, frame in frames.items():
        output_name = f"{args.prefix}_{name}.parquet"
        log.info(f"saving one at a time ({output_name})")
        to_parquet(frame.df, output_name, engine="auto")
    return 0


def cli():
    args, parser = parse_args()
    if args.action is None:
        parser.print_help()
        return 0

    # fmt: off
    logging.basicConfig(level=logging.INFO, format="{:20}  %(levelname)s  %(message)s".format("[%(name)s]"))
    logging.addLevelName(logging.WARNING, "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)))
    logging.addLevelName(logging.ERROR, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)))
    logging.addLevelName(logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)))
    logging.addLevelName(logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)))
    log = logging.getLogger("tdub.cli")
    # fmt: on

    if args.action == "regions2parquet":
        return _parquet_regions(args, log)
    elif args.action == "stacks":
        return run_stacks(args)
    elif args.action == "pulls":
        return run_pulls(args)
    else:
        parser.print_help()

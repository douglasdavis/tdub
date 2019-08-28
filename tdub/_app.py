import argparse
import dask
from tdub.frames import stdregion_dataframes
from dask.distributed import Client, Lock
from dask.utils import SerializableLock
import logging


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
        frame.df.to_parquet(output_name, f"/{args.tree_name}")
    return 0


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(prog="tdub", description="tee-double-you CLI")
    subparsers = parser.add_subparsers(dest="action", help="Action")

    regions2parquet = subparsers.add_parser("regions2parquet", help="generate parquet output for individual regions")
    regions2parquet.add_argument("files", type=str, nargs="+", help="input ROOT files")
    regions2parquet.add_argument("prefix", type=str, help="output file name prefix")
    regions2parquet.add_argument("-b","--branches", type=str, nargs="+", default=None, help="Branches")
    regions2parquet.add_argument("-t","--tree-name", type=str, default="WtLoop_nominal", help="ROOT tree name")
    # fmt: on
    return (parser.parse_args(), parser)


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
    else:
        parser.print_help()

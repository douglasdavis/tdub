import argparse
import dask
from tdub.frames import stdregion_dataframes
from dask.distributed import Client, Lock
from dask.utils import SerializableLock
import logging


def _h5_regions(args, log):
    frames = stdregion_dataframes(args.files, args.tree_name, args.branches)
    computes = []
    for name, frame in frames.items():
        output_name = f"{args.prefix}_{name}.h5"
        if args.delay:
            computes.append(frame.to_hdf(output_name, f"/{args.tree_name}", compute=False))
        else:
            log.info(f"generating undelayed {output_name}")
            frame.to_hdf(output_name, f"/{args.tree_name}")
    if args.delay:
        dask.compute(*computes)
    return 0


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(prog="tdub", description="tee-double-you CLI")
    subparsers = parser.add_subparsers(dest="action", help="Action")

    regions2hdf = subparsers.add_parser("regions2hdf", help="generate HDF5 files for individual regions")
    regions2hdf.add_argument("files", type=str, nargs="+", help="input ROOT files")
    regions2hdf.add_argument("prefix", type=str, help="output file name prefix")
    regions2hdf.add_argument("-b","--branches", type=str, nargs="+", default=None, help="Branches")
    regions2hdf.add_argument("-d","--delay", action="store_true", help="delay to_hdf calls (unstable)")
    regions2hdf.add_argument("-t","--tree-name", type=str, default="WtLoop_nominal", help="ROOT tree name")
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

    if args.action == "regions2hdf":
        return _h5_regions(args, log)
    else:
        parser.print_help()

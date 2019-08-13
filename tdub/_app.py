import argparse
import dask
from tdub.data import selected_dataframes
from dask.distributed import Client, Lock
from dask.utils import SerializableLock


def _h5_regions(args):
    selections = {
        "reg1j1b": "(reg1j1b == True) & (OS == True)",
        "reg2j1b": "(reg2j1b == True) & (OS == True)",
        "reg2j2b": "(reg2j2b == True) & (OS == True)",
        "reg3j": "(reg3j == True) & (OS == True)",
    }
    frames = selected_dataframes(args.files, selections=selections)
    computes = []
    for name, frame in frames.items():
        output_name = f"{args.prefix}_{name}.h5"
        #computes.append(frame.to_hdf(output_name, f"/{args.prefix}", compute=False))
        frame.to_hdf(output_name, f"/{args.prefix}")
    #if args.save_graph:
    #    dask.visualize(computes, format="png", filename=f"dask-graph_{args.prefix}")
    #dask.compute(computes)
    return 0


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(prog="tdub", description="tee-double-you CLI")
    subparsers = parser.add_subparsers(dest="action", help="Action")

    h5regions = subparsers.add_parser("h5regions", help="generate HDF5 files for individual regions")
    h5regions.add_argument("files", type=str, nargs="+", help="input ROOT files")
    h5regions.add_argument("prefix", type=str, help="output file name prefix")
    #h5regions.add_argument("--save-graph", action="store_true", help="save dask computational graph as [prefix].png")
    #h5regions.add_argument("--dry", action="store_true", help="do not compute, only save graph")
    h5regions.set_defaults(func=_h5_regions)
    # fmt: on
    return (parser.parse_args(), parser)


def cli():
    args, parser = parse_args()
    if args.action is None:
        parser.print_help()
        return 0

    # fmt: off
    import logging
    logging.basicConfig(level=logging.INFO, format="{:20}  %(levelname)s  %(message)s".format("[%(name)s]"))
    logging.addLevelName(logging.WARNING, "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)))
    logging.addLevelName(logging.ERROR, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)))
    logging.addLevelName(logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)))
    logging.addLevelName(logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)))
    # fmt: on

    args.func(args)
    return 0

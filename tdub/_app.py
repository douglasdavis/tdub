"""
Command line application
"""

# stdlib
import argparse
import json
import logging
import pathlib

# tdub
from tdub import setup_logging


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(prog="tdub", description="tdub CLI")
    subparsers = parser.add_subparsers(dest="action", help="Action")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--debug", action="store_true", help="set logging level to debug")

    applygennpy = subparsers.add_parser("apply-gennpy", help="Calculate samples BDT response and save to .npy file")
    applygennpy.add_argument("--single-file", type=str, help="input ROOT file")
    applygennpy.add_argument("--all-in-dir", type=str, help="Process all files in a directory")
    applygennpy.add_argument("-f", "--folds", type=str, nargs="+", help="directories with outputs from folded trainings", required=True)
    applygennpy.add_argument("-n", "--arr-name", type=str, default="pbdt_response", help="name for the array")
    applygennpy.add_argument("-o", "--out-dir", type=str, help="save output to a specific directory")

    rexpulls = subparsers.add_parser("rex-pulls", help="create matplotlib pull plots from TRExFitter output", parents=[common_parser])
    rexpulls.add_argument("workspace", type=str, help="TRExFitter workspace")
    rexpulls.add_argument("config", type=str, help="TRExFitter config")
    rexpulls.add_argument("-o", "--out-dir", type=str, help="output directory")
    rexpulls.add_argument("--no-text", action="store_true", help="don't print values on plots")

    rexstacks = subparsers.add_parser("rex-stacks", help="create matplotlib stack plots from TRExFitter output", parents=[common_parser])
    rexstacks.add_argument("workspace", type=str, help="TRExFitter workspace")
    rexstacks.add_argument("-o", "--out-dir", type=str, help="output directory for plots")
    rexstacks.add_argument("--lumi", type=str, default="139", help="Integrated lumi. for text")
    rexstacks.add_argument("--do-postfit", action="store_true", help="produce post fit plots as well")
    rexstacks.add_argument("--skip-regions", type=str, default=None, help="skip regions based on regex")
    rexstacks.add_argument("--band-style", type=str, choices=["hatch", "shade"], default="hatch", help="systematic band style")
    rexstacks.add_argument("--legend-ncol", type=int, choices=[1, 2], default=1, help="number of legend columns")

    trainfold = subparsers.add_parser("train-fold", help="Perform a folded training")
    trainfold.add_argument("optimdir", type=str, help="directory containing optimization information")
    trainfold.add_argument("datadir", type=str, help="Directory with ROOT files")
    trainfold.add_argument("-o", "--out-dir", type=str, default="_folded", help="output directory for saving optimizatin results")
    trainfold.add_argument("-s", "--seed", type=int, default=414, help="random seed for folding")
    trainfold.add_argument("-n", "--n-splits", type=int, default=3, help="number of splits for folding")

    trainoptimize = subparsers.add_parser("train-optimize", help="Gaussian processes minimization for HP optimization")
    trainoptimize.add_argument("region", type=str, help="Region to train")
    trainoptimize.add_argument("nlomethod", type=str, help="NLO method samples to use", choices=["DR", "DS", "Both"])
    trainoptimize.add_argument("datadir", type=str, help="Directory with ROOT files")
    trainoptimize.add_argument("-o", "--out-dir", type=str, default="_optim", help="output directory for saving optimizatin results")
    trainoptimize.add_argument("-n", "--n-calls", type=int, default=15, help="number of calls for the optimization procedure")
    trainoptimize.add_argument("-r", "--esr", type=int, default=20, help="early stopping rounds for the training")

    # fmt: on
    return (parser.parse_args(), parser)


def _optimize(args):
    from tdub.train import gp_minimize_auc

    return gp_minimize_auc(
        args.region,
        args.nlomethod,
        args.datadir,
        output_dir=args.out_dir,
        n_calls=args.n_calls,
        esr=args.esr,
    )


def _foldedtraining(args):
    from tdub.train import folded_training, prepare_from_root
    from tdub.utils import quick_files

    with open(f"{args.optimdir}/summary.json", "r") as f:
        summary = json.load(f)
    nlo_method = summary["nlo_method"]
    qfiles = quick_files(f"{args.datadir}")
    if nlo_method == "DR":
        tW_files = qfiles["tW_DR"]
    elif nlo_method == "DS":
        tW_files = qfiles["tW_DS"]
    elif nlo_method == "Both":
        tW_files = qfiles["tW_DR"] + qfiles["tW_DS"]
        tW_files.sort()
    else:
        raise ValueError("nlo_method must be 'DR' or 'DS' or 'Both'")

    X, y, w, cols = prepare_from_root(tW_files, qfiles["ttbar"], summary["region"])
    folded_training(
        X,
        y,
        w,
        cols,
        summary["best_params"],
        {"verbose": 20},
        args.out_dir,
        summary["region"],
        kfold_kw={"n_splits": args.n_splits, "shuffle": True, "random_state": args.seed},
    )
    return 0


def _pred2npy(args):
    from tdub.apply import FoldedResult, generate_npy
    from tdub.frames import conservative_dataframe
    from tdub.utils import SampleInfo

    frs = [FoldedResult(p) for p in args.folds]

    if args.out_dir:
        outdir = pathlib.PosixPath(args.out_dir)
    else:
        outdir = pathlib.PosixPath(".")

    def process_sample(sample_name):
        stem = pathlib.PosixPath(sample_name).stem
        sampinfo = SampleInfo(stem)
        tree = f"WtLoop_{sampinfo.tree}"
        df = conservative_dataframe(sample_name, tree=tree)
        npyfilename = outdir / f"{stem}.{args.arr_name}.npy"
        generate_npy(frs, df, npyfilename)

    if args.single_file is not None and args.all_in_dir is not None:
        raise ValueError("--single-file and --all-in-dir cannot be used together")

    if args.single_file is not None:
        process_sample(args.single_file)
        return 0
    elif args.all_in_dir is not None:
        for child in pathlib.PosixPath(args.all_in_dir).iterdir():
            if child.suffix == ".root":
                process_sample(str(child.resolve()))
        return 0


def cli():
    args, parser = parse_args()
    if args.action is None:
        parser.print_help()
        return 0

    setup_logging()
    log = logging.getLogger("tdub.cli")

    # fmt: off
    if args.action == "rex-stacks":
        from tdub.rex_art import run_stacks
        return run_stacks(args)
    elif args.action == "rex-pulls":
        from tdub.rex_art import run_pulls
        return run_pulls(args)
    elif args.action == "train-optimize":
        return _optimize(args)
    elif args.action == "train-fold":
        return _foldedtraining(args)
    elif args.action == "apply-gennpy":
        return _pred2npy(args)
    else:
        parser.print_help()
    # fmt: on

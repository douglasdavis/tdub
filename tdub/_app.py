"""
Command line application
"""

# stdlib
import argparse
import json
import pathlib
import logging

# external
import yaml

# tdub
from tdub.apply import FoldedResult, generate_npy
from tdub.batch import gen_apply_npy_script
from tdub.features import create_parquet_files, prepare_from_parquet, FeatureSelector
from tdub.frames import raw_dataframe, drop_cols
from tdub.rex_art import run_stacks, run_pulls
from tdub.utils import SampleInfo, quick_files
from tdub import setup_logging


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(prog="tdub", description="tdub CLI")
    subparsers = parser.add_subparsers(dest="action", help="")
    subparsers.metavar = "action           "
    common_parser = argparse.ArgumentParser(add_help=False, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=60))
    common_parser.add_argument("--debug", action="store_true", help="set logging level to debug")

    applygennpy = subparsers.add_parser("apply-gennpy", help="Calculate samples BDT response and save to .npy file", parents=[common_parser])
    applygennpy.add_argument("--bnl-dir", type=str, help="directory on BNL parse for generating a condor submission script")
    applygennpy.add_argument("--single-file", type=str, help="input ROOT file")
    applygennpy.add_argument("--all-in-dir", type=str, help="Process all files in a directory")
    applygennpy.add_argument("-f", "--folds", type=str, nargs="+", help="directories with outputs from folded trainings", required=True)
    applygennpy.add_argument("-n", "--arr-name", type=str, default="pbdt_response", help="name for the array")
    applygennpy.add_argument("-o", "--out-dir", type=str, help="save output to a specific directory")
    applygennpy.add_argument("--bnl-script-name", type=str, help="name for output condor submission script when using '--bnl-dir")
    applygennpy.add_argument("--ignore-main", action="store_true", help="skip 'main' samples (ttbar & tW_(DR,DS) nominal FS)")

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

    fselprepare = subparsers.add_parser("fsel-prepare", help="Prepare a set of parquet files for feature selection", parents=[common_parser])
    fselprepare.add_argument("-i", "--in", dest="indir", type=str, required=True, help="Directory containing ROOT files")
    fselprepare.add_argument("-o", "--out", dest="outdir", type=str, required=True, help="Output directory to save parquet files")
    fselprepare.add_argument("--entrysteps", type=str, required=False, help="entrysteps argument for create_parquet_files function")
    fselprepare.add_argument("--use-campaign-weight", action="store_true", help="propagate weight_campaign to weight_nominal")

    fselexecute = subparsers.add_parser("fsel-execute", help="Execute a round of feature selection", parents=[common_parser])
    fselexecute.add_argument("-i", "--in-pqdir", type=str, help="Directory containg the parquet files")
    fselexecute.add_argument("-n", "--nlo-method", type=str, choices=["DR", "DS"], required=True, help="tW NLO sample")
    fselexecute.add_argument("-r", "--region", type=str, choices=["1j1b", "2j1b", "2j2b"], required=True, help="Region to process")
    fselexecute.add_argument("-t", "--type", type=str, dest="itype", choices=["split", "gain"], required=True, help="importance type")
    fselexecute.add_argument("-o", "--out", type=str, required=False, default="_auto", help="Output directory to save selection result (prepended with 'fsel_result.')")
    fselexecute.add_argument("--corr-threshold", dest="corrt", type=float, default=0.85, help="Correlation threshold")
    fselexecute.add_argument("--importance-n-fits", dest="nfits", type=int, default=5, help="Number of fitting rounds for importance calc.")
    fselexecute.add_argument("--max-features", dest="maxf", type=int, default=25, help="maximum number of features to test iteratively")
    fselexecute.add_argument("--exclude", type=str, nargs="+", help="features to exclude from consideration")
    fselexecute.add_argument("--weight-mean", type=float, default=1.0, help="scale weights such that the mean is this value")
    fselexecute.add_argument("--test-case", type=int, required=False, help="when testing, get a subset of each sample of this size")
    fselexecute.add_argument("--ttbar-frac", type=float, required=False, help="use a fraction of the ttbar sample (can't be used with --test-case")

    # fmt: on
    return (parser.parse_args(), parser)


def _fselprepare(args):
    create_parquet_files(args.indir, args.outdir, args.entrysteps, args.use_campaign_weight)


def _fselexecute(args):
    if args.out == "_auto":
        name = f"{args.nlo_method}_{args.region}_{args.itype}"
    else:
        name = args.out

    full_df, full_labels, full_weights = prepare_from_parquet(
        args.in_pqdir,
        region=args.region,
        nlo_method=args.nlo_method,
        weight_mean=args.weight_mean,
        test_case_size=args.test_case,
        ttbar_frac=args.ttbar_frac,
    )
    if args.exclude:
        drop_cols(full_df, *args.exclude)

    fs = FeatureSelector(
        df=full_df,
        labels=full_labels,
        weights=full_weights,
        importance_type=args.itype,
        name=name,
    )
    # fs.check_for_uniques()
    fs.check_importances(n_fits=args.nfits)
    fs.check_collinearity(threshold=args.corrt)
    fs.check_candidates(n=args.maxf)
    fs.check_iterative_remove_aucs(max_features=args.maxf)
    fs.check_iterative_add_aucs(max_features=args.maxf)
    fs.save_result()


def _pred2npy(args):
    n_opts = 0
    if args.single_file is not None:
        n_opts += 1
    if args.all_in_dir is not None:
        n_opts += 1
    if args.bnl_dir is not None:
        n_opts += 1
    if n_opts != 1:
        raise ValueError(
            "must choose one (and only one) of '--single-file', '--all-in-dir', '--bnl-dir'"
        )

    if args.out_dir:
        outdir = pathlib.PosixPath(args.out_dir)
    else:
        outdir = pathlib.PosixPath(".")

    if args.bnl_dir is not None:
        gen_apply_npy_script(
            args.bnl_dir, args.folds, outdir, args.arr_name, args.bnl_script_name
        )
        return 0

    frs = [FoldedResult(p) for p in args.folds]
    necessary_branches = ["OS", "elmu", "reg1j1b", "reg2j1b", "reg2j2b"]
    for fold in frs:
        necessary_branches += fold.features
    necessary_branches = sorted(set(necessary_branches), key=str.lower)
    log = logging.getLogger("tdub.apply")
    log.info("Loading necessary branches:")
    for nb in necessary_branches:
        log.info(f" - {nb}")

    def process_sample(sample_name):
        stem = pathlib.PosixPath(sample_name).stem
        sampinfo = SampleInfo(stem)
        tree = f"WtLoop_{sampinfo.tree}"
        df = raw_dataframe(sample_name, tree=tree, branches=necessary_branches)
        npyfilename = outdir / f"{stem}.{args.arr_name}.npy"
        generate_npy(frs, df, npyfilename)

    main_samples_pfxs = (
        "tW_DR_410648_FS",
        "tW_DR_410649_FS",
        "tW_DS_410656_FS",
        "tW_DS_410657_FS",
        "ttbar_410472_FS",
    )

    if args.single_file is not None:
        process_sample(args.single_file)
        return 0
    elif args.all_in_dir is not None:
        for child in pathlib.PosixPath(args.all_in_dir).iterdir():
            if args.ignore_main and child.name.startswith(main_samples_pfxs):
                continue
            if child.suffix == ".root":
                process_sample(str(child.resolve()))
        return 0


def cli():
    args, parser = parse_args()
    if args.action is None:
        parser.print_help()
        return 0

    setup_logging()

    for name in logging.root.manager.loggerDict:
        if args.debug:
            logging.getLogger(name).setLevel(logging.DEBUG)
        else:
            logging.getLogger(name).setLevel(logging.INFO)

    # fmt: off
    if args.action == "rex-stacks":
        return run_stacks(args)
    elif args.action == "rex-pulls":
        return run_pulls(args)
    elif args.action == "train-optimize":
        return _optimize(args)
    elif args.action == "train-fold":
        return _foldedtraining(args)
    elif args.action == "apply-gennpy":
        return _pred2npy(args)
    elif args.action == "fsel-prepare":
        return _fselprepare(args)
    elif args.action == "fsel-execute":
        return _fselexecute(args)
    else:
        parser.print_help()
    # fmt: on

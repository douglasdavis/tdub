"""
tdub command line interface
"""

# stdlib
import json
import logging
import os
import shutil
from pathlib import PosixPath

# third party
import click

# tdub
from tdub import setup_logging

setup_logging()
log = logging.getLogger("tdub-cli")


@click.group(context_settings=dict(max_content_width=92))
def cli():
    pass


@cli.command("train-single", context_settings=dict(max_content_width=92))
@click.option(
    "-d",
    "--datadir",
    type=click.Path(exists=True),
    help="directory containing data files",
    required=True,
)
@click.option("-r", "--region", type=str, required=True, help="the region to train on")
@click.option("-o", "--outdir", type=str, required=True, help="output directory name")
@click.option(
    "-n",
    "--nlo-method",
    type=str,
    default="DR",
    help="tW simluation NLO method",
    show_default=True,
)
@click.option(
    "-x", "--override-selection", type=str, help="override selection with contents of file"
)
@click.option("-t", "--use-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("-i", "--ignore-list", type=str, help="variable ignore list file")
@click.option("-e", "--early-stop", type=int, help="number of early stopping rounds")
@click.option("-u", "--use-dilep", is_flag=True, help="train with dilepton samples")
@click.option(
    "--learning-rate", type=float, required=True, help="learning_rate model parameter"
)
@click.option("--num-leaves", type=int, required=True, help="num_leaves model parameter")
@click.option(
    "--min-child-samples", type=int, required=True, help="min_child_samples model parameter"
)
@click.option("--max-depth", type=int, required=True, help="max_depth model parameter")
@click.option(
    "--n-estimators", type=int, required=True, help="n_estimators model parameter"
)
def single(
    datadir,
    region,
    outdir,
    nlo_method,
    override_selection,
    use_tptrw,
    ignore_list,
    early_stop,
    use_dilep,
    learning_rate,
    num_leaves,
    min_child_samples,
    max_depth,
    n_estimators,
):
    """Execute a single training round."""
    # fmt: on
    from tdub.train import single_training, prepare_from_root
    from tdub.utils import get_avoids, quick_files
    from tdub.frames import drop_cols

    qf = quick_files(datadir)
    sig_files = qf[f"tW_{nlo_method}"] if use_dilep else qf[f"tW_{nlo_method}_inc"]
    bkg_files = qf["ttbar"] if use_dilep else qf["ttbar_inc"]
    override_sel = override_selection
    if override_sel:
        override_sel = PosixPath(override_sel).read_text().strip()
    df, y, w = prepare_from_root(
        sig_files,
        bkg_files,
        region,
        weight_mean=1.0,
        override_selection=override_sel,
        use_tptrw=use_tptrw,
    )
    drop_cols(df, *get_avoids(region))
    if ignore_list:
        drops = PosixPath(ignore_list).read_text().strip().split()
        drop_cols(df, *drops)
    params = dict(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        max_depth=max_depth,
        n_estimators=n_estimators,
    )
    extra_sum = {"region": region, "nlo_method": nlo_method}
    single_training(
        df,
        y,
        w,
        params,
        outdir,
        early_stopping_rounds=early_stop,
        extra_summary_entries=extra_sum,
    )
    return 0


@cli.command("train-scan", context_settings=dict(max_content_width=140))
@click.argument("config", type=click.Path(exists=True, resolve_path=True))
@click.argument("datadir", type=click.Path(exists=True, resolve_path=True))
@click.argument("workspace", type=click.Path(exists=False))
@click.option("-r", "--region", type=str, required=True, help="the region to train on")
@click.option(
    "-n",
    "--nlo-method",
    type=str,
    default="DR",
    help="tW simluation NLO method",
    show_default=True,
)
@click.option(
    "-x", "--override-selection", type=str, help="override selection with contents of file"
)
@click.option("-t", "--use-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("-i", "--ignore-list", type=str, help="variable ignore list file")
@click.option("-e", "--early-stop", type=int, help="number of early stopping rounds")
@click.option("-u", "--use-dilep", is_flag=True, help="train with dilepton samples")
def scan(
    config,
    datadir,
    workspace,
    region,
    nlo_method,
    override_selection,
    use_tptrw,
    ignore_list,
    early_stop,
    use_dilep,
):
    """Perform a parameter scan via condor jobs.

    The scan parameters are defined in the CONFIG file, and the data
    to use is in the DATADIR. All relevant output will be saved to
    the WORKSPACE directory. Example:

    $ tdub train-scan grid.json /data/path scan_2j1b -r 2j1b -e 10

    """
    with open(config, "r") as f:
        pd = json.load(f)

    from tdub.batch import create_condor_workspace, condor_header, add_condor_arguments

    ws = create_condor_workspace(workspace, exist_ok=False)
    (ws / "res").mkdir()

    runs = []
    i = 0
    override_sel = override_selection
    if override_sel is None:
        override_sel = "_NONE"
    else:
        override_sel = str(PosixPath(override_sel).resolve())
    if ignore_list is None:
        ignore_list = "_NONE"
    else:
        ignore_list = str(PosixPath(ignore_list).resolve())

    for max_depth in pd.get("max_depth"):
        for num_leaves in pd.get("num_leaves"):
            for n_estimators in pd.get("n_estimators"):
                for learning_rate in pd.get("learning_rate"):
                    for min_child_samples in pd.get("min_child_samples"):
                        suffix = "{}-{}-{}-{}-{}".format(
                            learning_rate,
                            num_leaves,
                            n_estimators,
                            min_child_samples,
                            max_depth,
                        )
                        arglist = (
                            "{}"
                            "{}"
                            "-d {} "
                            "-o {}/res/{:04d}_{} "
                            "-r {} "
                            "-n {} "
                            "-x {} "
                            "-i {} "
                            "--learning-rate {} "
                            "--num-leaves {} "
                            "--n-estimators {} "
                            "--min-child-samples {} "
                            "--max-depth {} "
                            "--early-stop {} "
                        ).format(
                            "-t " if use_tptrw else "",
                            "-u " if use_dilep else "",
                            datadir,
                            ws,
                            i,
                            suffix,
                            region,
                            nlo_method,
                            override_sel,
                            ignore_list,
                            learning_rate,
                            num_leaves,
                            n_estimators,
                            min_child_samples,
                            max_depth,
                            early_stop,
                        )
                        arglist = arglist.replace("-x _NONE ", "")
                        arglist = arglist.replace("-i _NONE ", "")
                        runs.append(arglist)
                        i += 1
    log.info(f"prepared {len(runs)} jobs for submission")
    with (ws / "scan.condor.sub").open("w") as f:
        condor_header(ws, shutil.which("tdub"), memory="2GB", file=f)
        for run in runs:
            add_condor_arguments(f"train-single {run}", f)

    with (ws / "run.sh").open("w") as f:
        print("#!/bin/bash\n\n", file=f)
        for run in runs:
            print(f"tdub train-single {run}\n", file=f)
    os.chmod(ws / "run.sh", 0o755)

    return 0


@cli.command("train-check", context_settings=dict(max_content_width=92))
@click.argument("workspace", type=click.Path(exists=True))
@click.option("-p", "--print-top", is_flag=True, help="Print the top results")
@click.option(
    "-n",
    "--n-res",
    type=int,
    default=10,
    help="Number of top results to print",
    show_default=True,
)
def check(workspace, print_top, n_res):
    """Check the results of a parameter scan WORKSPACE."""
    from tdub.train import SingleTrainingResult
    import shutil

    results = []
    top_dir = PosixPath(workspace)
    resdirs = (top_dir / "res").iterdir()
    for resdir in resdirs:
        if resdir.name == "logs" or not resdir.is_dir():
            continue
        summary_file = resdir / "summary.json"
        if not summary_file.exists():
            log.warn("no summary file for %s" % str(resdir))
        with summary_file.open("r") as f:
            summary = json.load(f)
            if summary["bad_ks"]:
                continue
            res = SingleTrainingResult(**summary)
            res.workspace = resdir
            res.summary = summary
            results.append(res)

    auc_sr = sorted(results, key=lambda r: -r.auc)
    ks_pvalue_sr = sorted(results, key=lambda r: -r.ks_pvalue_sig)
    max_auc_rounded = str(round(auc_sr[0].auc, 2))

    potentials = []
    for res in ks_pvalue_sr:
        curauc = str(round(res.auc, 2))
        if curauc == max_auc_rounded and res.ks_pvalue_bkg > 0.95:
            potentials.append(res)
        if len(potentials) > 15:
            break

    for result in potentials:
        print(result)

    best_res = potentials[0]
    if os.path.exists(top_dir / "best"):
        shutil.rmtree(top_dir / "best")
    shutil.copytree(best_res.workspace, top_dir / "best")
    print(best_res.workspace.name)
    print(best_res.summary)

    return 0


@cli.command("train-fold", context_settings=dict(max_content_width=92))
@click.argument("scandir", type=click.Path(exists=True))
@click.argument("datadir", type=click.Path(exists=True))
@click.option("-t", "--use-tptrw", is_flag=True, help="use top pt reweighting")
@click.option(
    "-r",
    "--random-seed",
    type=int,
    default=414,
    help="random seed for folding",
    show_default=True,
)
@click.option(
    "-n",
    "--n-splits",
    type=int,
    default=3,
    help="number of splits for folding",
    show_default=True,
)
def fold(scandir, datadir, use_tptrw, random_seed, n_splits):
    """Perform a folded training based on a hyperparameter scan result."""
    from tdub.train import folded_training, prepare_from_root
    from tdub.utils import quick_files

    scandir = PosixPath(scandir).resolve()
    summary_file = scandir / "best" / "summary.json"
    outdir = scandir / "foldres"
    if outdir.exists():
        log.warn(f"fold result already exists for {scandir}, exiting")
        return 0
    summary = None
    with summary_file.open("r") as f:
        summary = json.load(f)
    nlo_method = summary["nlo_method"]
    best_iteration = summary["best_iteration"]
    if best_iteration > 0:
        summary["all_params"]["n_estimators"] = best_iteration
    region = summary["region"]
    branches = summary["features"]
    selection = summary["selection_used"]
    qf = quick_files(datadir)
    df, y, w = prepare_from_root(
        qf[f"tW_{nlo_method}"],
        qf["ttbar"],
        region,
        override_selection=selection,
        branches=branches,
        weight_mean=1.0,
        use_tptrw=use_tptrw,
    )
    folded_training(
        df,
        y,
        w,
        summary["all_params"],
        {"verbose": 10},
        str(outdir),
        summary["region"],
        kfold_kw={"n_splits": n_splits, "shuffle": True, "random_state": random_seed,},
    )
    return 0


@cli.command("apply-single", context_settings=dict(max_content_width=92))
@click.argument("infile", type=click.Path(exists=True))
@click.option(
    "-n",
    "--arr-name",
    type=str,
    help="name array name (eventual TTree branch name)",
    required=True,
)
@click.option("-o", "--outdir", type=str, help="save output to directory", required=True)
@click.option(
    "-f",
    "--fold-results",
    type=click.Path(exists=True),
    multiple=True,
    help="fold output directories",
)
@click.option(
    "-s",
    "--single-results",
    type=click.Path(exists=True),
    multiple=True,
    help="single result dirs",
)
def apply_single(infile, arr_name, outdir, fold_results=None, single_results=None):
    """Generate BDT response array for INFILE and save to .npy file.

    We generate the .npy files using either single training results
    (-s flag) or folded training results (-f flag).

    """
    if len(single_results) > 0 and len(fold_results) > 0:
        raise ValueError("Cannot use -f and -s together with apply-single")

    from tdub.apply import get_result_array, FoldedResult, SingleResult
    from tdub.utils import SampleInfo, minimal_branches
    from tdub.frames import raw_dataframe
    import numpy as np

    outdir = PosixPath(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    trs = None
    if len(fold_results) > 0:
        trs = [FoldedResult(p) for p in fold_results]
    elif len(single_results) > 0:
        trs = [SingleResult(p) for p in single_results]
    else:
        raise ValueError("-f or -s required")

    necessary_branches = ["OS", "elmu", "reg2j1b", "reg2j2b", "reg1j1b"]
    for res in trs:
        necessary_branches += res.features
        necessary_branches += minimal_branches(res.selection_used)
    necessary_branches = sorted(set(necessary_branches), key=str.lower)

    log.info("Loading necessary branches:")
    for nb in necessary_branches:
        log.info(f" - {nb}")

    stem = PosixPath(infile).stem
    sampinfo = SampleInfo(stem)
    tree = f"WtLoop_{sampinfo.tree}"
    df = raw_dataframe(infile, tree=tree, branches=necessary_branches)
    npyfilename = outdir / f"{stem}.{arr_name}.npy"
    result_arr = get_result_array(trs, df)
    np.save(npyfilename, result_arr)


@cli.command("soverb", context_settings=dict(max_content_width=92))
@click.argument("datadir", type=click.Path(exists=True))
@click.argument("selections", type=click.Path(exists=True))
@click.option("-t", "--use-tptrw", is_flag=True, help="use top pt reweighting")
def soverb(datadir, selections, use_tptrw):
    """Get signal over background using data in DATADIR and a SELECTIONS file.

    the format of the JSON entries should be "region": "numexpr selection".

    Example SELECTIONS file:

    \b
    {
        "reg1j1b" : "(mass_lep1lep2 < 150) & (mass_lep2jet1 < 150)",
        "reg1j1b" : "(mass_jet1jet2 < 150) & (mass_lep2jet1 < 120)",
        "reg2j2b" : "(met < 120)"
    }

    """
    from tdub.frames import raw_dataframe, apply_weight_tptrw, satisfying_selection
    from tdub.utils import quick_files, minimal_branches

    with open(selections, "r") as f:
        selections = json.load(f)

    necessary_branches = set()
    for selection, query in selections.items():
        necessary_branches |= minimal_branches(query)
    necessary_branches = list(necessary_branches) + ["weight_tptrw_tool"]

    qf = quick_files(datadir)
    bkg = qf["ttbar"] + qf["Diboson"] + qf["Zjets"] + qf["MCNP"]
    sig = qf["tW_DR"]

    sig_df = raw_dataframe(sig, branches=necessary_branches)
    bkg_df = raw_dataframe(bkg, branches=necessary_branches, entrysteps="1GB")
    apply_weight_tptrw(bkg_df)

    for sel, query in selections.items():
        s_df, b_df = satisfying_selection(sig_df, bkg_df, selection=query)
        print(sel, s_df["weight_nominal"].sum() / b_df["weight_nominal"].sum())


def run_cli():
    import tdub.constants

    tdub.constants.AVOID_IN_CLF_1j1b = []
    tdub.constants.AVOID_IN_CLF_2j1b = []
    tdub.constants.AVOID_IN_CLF_2j2b = []
    cli()


if __name__ == "__main__":
    run_cli()

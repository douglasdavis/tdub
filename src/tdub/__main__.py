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


@click.group(context_settings=dict(max_content_width=82))
def cli():
    pass


@cli.command("train-single", context_settings=dict(max_content_width=92))
@click.argument("datadir", type=click.Path(exists=True))
@click.argument("region", type=str)
@click.argument("outdir", type=click.Path())
@click.option("-p", "--pre-exec", type=click.Path(resolve_path=True), help="Python code to pre-execute")
@click.option("-n", "--nlo-method", type=str, default="DR", help="tW simluation NLO method", show_default=True)
@click.option("-x", "--override-selection", type=str, help="override selection with contents of file")
@click.option("-t", "--use-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("-i", "--ignore-list", type=str, help="variable ignore list file")
@click.option("-s", "--test-size", type=float, default=0.40, help="training test size", show_default=True)
@click.option("-e", "--early-stop", type=int, default=10, help="number of early stopping rounds", show_default=True)
@click.option("-d", "--use-dilep", is_flag=True, help="train with dilepton samples")
@click.option("-k", "--use-sklearn", is_flag=True, help="use sklearn instead of lgbm")
@click.option("--learning-rate", type=float, default=0.1, help="learning_rate model parameter", show_default=True)
@click.option("--num-leaves", type=int, default=31, help="num_leaves model parameter", show_default=True)
@click.option("--min-child-samples", type=int, default=50, help="min_child_samples model parameter", show_default=True)
@click.option("--max-depth", type=int, default=5, help="max_depth model parameter", show_default=True)
@click.option("--reg-lambda", type=float, default=0, help="lambda (L2) regularization", show_default=True)
def single(
    datadir,
    region,
    outdir,
    pre_exec,
    nlo_method,
    override_selection,
    use_tptrw,
    ignore_list,
    test_size,
    early_stop,
    use_dilep,
    use_sklearn,
    learning_rate,
    num_leaves,
    min_child_samples,
    max_depth,
    reg_lambda,
):
    """Execute a single training round."""
    if pre_exec is not None:
        exec(PosixPath(pre_exec).read_text())

    from tdub.ml_train import single_training, prepare_from_root
    from tdub.data import avoids_for, quick_files
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
    drop_cols(df, *avoids_for(region))
    if ignore_list:
        drops = PosixPath(ignore_list).read_text().strip().split()
        drop_cols(df, *drops)
    train_axes = dict(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
    )
    extra_sum = {"region": region, "nlo_method": nlo_method}
    single_training(
        df,
        y,
        w,
        train_axes,
        outdir,
        test_size=test_size,
        early_stopping_rounds=early_stop,
        extra_summary_entries=extra_sum,
        use_sklearn=use_sklearn,
    )
    return 0


@cli.command("train-scan", context_settings=dict(max_content_width=140))
@click.argument("datadir", type=click.Path(exists=True, resolve_path=True))
@click.argument("region", type=str)
@click.argument("workspace", type=click.Path(exists=False))
@click.option("-p", "--pre-exec", type=click.Path(resolve_path=True), help="Python code to pre-execute")
@click.option("-n", "--nlo-method", type=str, default="DR", help="tW simluation NLO method", show_default=True)
@click.option("-e", "--early-stop", type=int, default=10, help="number of early stopping rounds", show_default=True)
@click.option("-x", "--override-selection", type=str, help="override selection with contents of file")
@click.option("-t", "--use-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("-i", "--ignore-list", type=str, help="variable ignore list file")
@click.option("-s", "--test-size", type=float, default=0.40, help="training test size", show_default=True)
@click.option("-d", "--use-dilep", is_flag=True, help="train with dilepton samples")
@click.option("--overwrite", is_flag=True, help="overwrite existing workspace")
@click.option("--and-submit", is_flag=True, help="submit the condor jobs")
def scan(
    datadir,
    region,
    workspace,
    pre_exec,
    nlo_method,
    early_stop,
    override_selection,
    use_tptrw,
    ignore_list,
    test_size,
    use_dilep,
    overwrite,
    and_submit,
):
    """Perform a parameter scan via condor jobs.

    DATADIR points to the intput ROOT files, training is performed on
    the REGION and all output is saved to WORKSPACE.

    $ tdub train-scan /data/path 2j2b scan_2j2b

    """

    if pre_exec is not None:
        exec(PosixPath(pre_exec).read_text())

    # from tdub.batch import (
    #     create_condor_workspace,
    #     condor_preamble,
    #     add_condor_arguments,
    #     condor_submit,
    # )

    from tdub.batch import create_condor_workspace
    import tdub.config
    import itertools

    ws = create_condor_workspace(workspace, overwrite=overwrite)
    (ws / "res").mkdir()

    log.info(f"Preparing a scan; results saved to {workspace}")
    log.info(f"  - region: {region}")
    log.info(f"  - NLO method: {nlo_method}")
    log.info("  - Using {} samples".format("dilepton" if use_dilep else "inclusive"))
    log.info("  - Apply top pt reweight: {}".format("yes" if use_tptrw else "no"))

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
    if pre_exec is None:
        pre_exec = "_NONE"
    else:
        pre_exec = str(PosixPath(pre_exec).resolve())

    pd = tdub.config.DEFAULT_SCAN_PARAMETERS
    itr = itertools.product(
        pd.get("max_depth"),
        pd.get("num_leaves"),
        pd.get("learning_rate"),
        pd.get("min_child_samples"),
        pd.get("reg_lambda"),
    )

    log.info("Scan grid:")
    log.info(" - max_depth: {}".format(pd.get("max_depth")))
    log.info(" - num_leaves: {}".format(pd.get("num_leaves")))
    log.info(" - learning_rate: {}".format(pd.get("learning_rate")))
    log.info(" - min_child_samples: {}".format(pd.get("min_child_samples")))
    log.info(" - reg_lambda: {}".format(pd.get("reg_lambda")))

    for (max_depth, num_leaves, learning_rate, min_child_samples, reg_lambda) in itr:
        suffix = "{}-{}-{}-{}-{}".format(
            max_depth, num_leaves, learning_rate, min_child_samples, reg_lambda,
        )
        outdir = ws / "res" / f"{i:04d}_{suffix}"
        arglist = (
            "{} {} {} -s {} -n {} -x {} -i {} -p {} "
            "--learning-rate {} "
            "--num-leaves {} "
            "--min-child-samples {} "
            "--max-depth {} "
            "--reg-lambda {} "
            "--early-stop {} "
            "{}"
            "{}"
        ).format(
            datadir,
            region,
            outdir,
            test_size,
            nlo_method,
            override_sel,
            ignore_list,
            pre_exec,
            learning_rate,
            num_leaves,
            min_child_samples,
            max_depth,
            reg_lambda,
            early_stop,
            "-t " if use_tptrw else "",
            "-d " if use_dilep else "",
        )
        arglist = arglist.replace("-x _NONE ", "")
        arglist = arglist.replace("-i _NONE ", "")
        arglist = arglist.replace("-p _NONE ", "")
        runs.append(arglist)
        i += 1

    with (ws / "run.sh").open("w") as outscript:
        print("#!/bin/bash\n\n", file=outscript)
        for run in runs:
            print(f"tdub train-single {run}\n", file=outscript)
    os.chmod(ws / "run.sh", 0o755)

    import pycondor

    condor_dag = pycondor.Dagman(name="dag_train_scan", submit=str(ws / "sub"))
    condor_job_scan = pycondor.Job(
        name="job_train_scan",
        universe="vanilla",
        getenv=True,
        notification="Error",
        extra_lines=["notify_user = ddavis@phy.duke.edu"],
        executable=shutil.which("tdub"),
        submit=str(ws / "sub"),
        error=str(ws / "err"),
        output=str(ws / "out"),
        log=str(ws / "log"),
        dag=condor_dag,
    )
    for run in runs:
        condor_job_scan.add_arg(f"train-single {run}")
    condor_job_check = pycondor.Job(
        name="job_train_check",
        universe="vanilla",
        getenv=True,
        notification="Error",
        extra_lines=["notify_user = ddavis@phy.duke.edu"],
        executable=shutil.which("tdub"),
        submit=str(ws / "sub"),
        error=str(ws / "err"),
        output=str(ws / "out"),
        log=str(ws / "log"),
        dag=condor_dag,
    )
    condor_job_check.add_arg(f"train-check {ws}")
    condor_job_check.add_parent(condor_job_scan)

    if and_submit:
        condor_dag.build_submit()
    else:
        condor_dag.build()

    # log.info(f"prepared {len(runs)} jobs for submission")
    # with (ws / "condor.sub").open("w") as f:
    #     condor_preamble(ws, shutil.which("tdub"), memory="2GB", GetEnv=True, to_file=f)
    #     for run in runs:
    #         add_condor_arguments(f"train-single {run}", f)
    # if and_submit:
    #     condor_submit(workspace)

    return 0


@cli.command("train-check", context_settings=dict(max_content_width=92))
@click.argument("workspace", type=click.Path(exists=True))
@click.option("-p", "--print-top", is_flag=True, help="Print the top results")
@click.option("-n", "--n-res", type=int, default=10, help="Number of top results to print", show_default=True)
def check(workspace, print_top, n_res):
    """Check the results of a parameter scan WORKSPACE."""
    from tdub.ml_train import SingleTrainingSummary
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
            res = SingleTrainingSummary(**summary)
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
@click.option("-r", "--random-seed", type=int, default=414, help="random seed for folding", show_default=True)
@click.option("-n", "--n-splits", type=int, default=3, help="number of splits for folding", show_default=True)
def fold(scandir, datadir, use_tptrw, random_seed, n_splits):
    """Perform a folded training based on a hyperparameter scan result."""
    from tdub.ml_train import folded_training, prepare_from_root
    from tdub.data import quick_files

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
        kfold_kw={"n_splits": n_splits, "shuffle": True, "random_state": random_seed},
    )
    return 0


@cli.command("apply-single", context_settings=dict(max_content_width=92))
@click.argument("infile", type=click.Path(exists=True))
@click.argument("arrname", type=str)
@click.argument("outdir", type=click.Path())
@click.option("-f", "--fold-results", type=click.Path(exists=True), multiple=True, help="fold output directories")
@click.option("-s", "--single-results", type=click.Path(exists=True), multiple=True, help="single result dirs")
def apply_single(infile, arrname, outdir, fold_results=None, single_results=None):
    """Generate BDT response array for INFILE and save to .npy file.

    We generate the .npy files using either single training results
    (-s flag) or folded training results (-f flag).

    """
    if len(single_results) > 0 and len(fold_results) > 0:
        raise ValueError("Cannot use -f and -s together with apply-single")

    from tdub.ml_apply import build_array, FoldedTrainSummary, SingleTrainSummary
    from tdub.data import SampleInfo
    from tdub.data import selection_branches
    from tdub.frames import raw_dataframe
    import numpy as np

    outdir = PosixPath(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    trs = None
    if len(fold_results) > 0:
        trs = [FoldedTrainSummary(p) for p in fold_results]
    elif len(single_results) > 0:
        trs = [SingleTrainSummary(p) for p in single_results]
    else:
        raise ValueError("-f or -s required")

    necessary_branches = ["OS", "elmu", "reg2j1b", "reg2j2b", "reg1j1b"]
    for res in trs:
        necessary_branches += res.features
        necessary_branches += selection_branches(res.selection_used)
    necessary_branches = sorted(set(necessary_branches), key=str.lower)

    log.info("Loading necessary branches:")
    for nb in necessary_branches:
        log.info(f" - {nb}")

    stem = PosixPath(infile).stem
    sampinfo = SampleInfo(stem)
    tree = f"WtLoop_{sampinfo.tree}"
    log.info(f"Using tree {tree}")
    df = raw_dataframe(infile, tree=tree, branches=necessary_branches)
    npyfilename = outdir / f"{stem}.{arrname}.npy"
    result_arr = build_array(trs, df)
    np.save(npyfilename, result_arr)


@cli.command("apply-all", context_settings=dict(max_content_width=92))
@click.argument("datadir", type=click.Path(exists=True))
@click.argument("arrname", type=str)
@click.argument("outdir", type=click.Path(resolve_path=True))
@click.argument("workspace", type=click.Path())
@click.option("-f", "--fold-results", type=click.Path(exists=True, resolve_path=True), multiple=True, help="fold output directories")
@click.option("-s", "--single-results", type=click.Path(exists=True, resolve_path=True), multiple=True, help="single result dirs")
@click.option("--and-submit", is_flag=True, help="submit the condor jobs")
def apply_all(
    datadir,
    arrname,
    outdir,
    workspace,
    fold_results=None,
    single_results=None,
    and_submit=False,
):
    """Generate BDT response arrays for all ROOT files in DATAIR"""
    import glob
    import shutil
    import pycondor

    if len(single_results) > 0 and len(fold_results) > 0:
        raise ValueError("Cannot use -f and -s together with apply-single")
    results_flags = None
    if len(fold_results) > 0:
        results_flags = "-f {}".format(" -f ".join(fold_results))
    elif len(single_results) > 0:
        results_flags = "-s {}".format(" -s ".join(single_results))
    else:
        raise ValueError("-f or -s required")

    ws = PosixPath(workspace).resolve()

    outpath = PosixPath(outdir).resolve()
    outpath.mkdir(exist_ok=True)

    datapath = PosixPath(datadir).resolve(strict=True)
    all_files = glob.glob(f"{datapath}/*.root")
    arglist = [f"{f} {arrname} {outpath} {results_flags}" for f in all_files]

    condor_dag = pycondor.Dagman(name="dag_train_scan", submit=str(ws / "sub"))
    condor_job_scan = pycondor.Job(
        name="job_apply_all",
        universe="vanilla",
        getenv=True,
        notification="Error",
        extra_lines=["notify_user = ddavis@phy.duke.edu"],
        executable=shutil.which("tdub"),
        submit=str(ws / "sub"),
        error=str(ws / "err"),
        output=str(ws / "out"),
        log=str(ws / "log"),
        dag=condor_dag,
    )
    for run in arglist:
        condor_job_scan.add_arg(f"apply-single {run}")

    if and_submit:
        condor_dag.build_submit()
    else:
        condor_dag.build()


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
    from tdub.data import quick_files
    from tdub.data import selection_branches

    with open(selections, "r") as f:
        selections = json.load(f)

    necessary_branches = set()
    for selection, query in selections.items():
        necessary_branches |= selection_branches(query)
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


@cli.command("rex-plot", context_settings=dict(max_content_width=92))
@click.argument("workspace", type=click.Path(exists=True))
@click.option("-o", "outdir", type=str, default="auto")
@click.option("--chisq/--no-chisq", default=True, help="Do or don't print chi-square information.")
@click.option("--impact-only", is_flag=True, help="Only produce the impact plot.")
def rex_plot(workspace, outdir, chisq, impact_only):
    """Generate plots from TRExFitter WORKSPACE using METADATA."""
    import tdub.rex
    import tdub.config

    if outdir == "auto":
        outdir = PosixPath(workspace) / "matplotlib"
    else:
        outdir = PosixPath(outdir)
    outdir.mkdir(exist_ok=True)
    if impact_only:
        tdub.rex.nuispar_impact_plot_top15(workspace)
        return 0
    tdub.config.init_meta_table()
    tdub.config.init_meta_logy()
    tdub.rex.plot_all_regions(workspace, outdir, stage="pre", show_chisq=chisq)
    tdub.rex.plot_all_regions(workspace, outdir, stage="post", show_chisq=chisq)


@cli.command("imp-tables", context_settings=dict(max_content_width=92))
@click.argument("summary-file", type=click.Path(exists=True))
def imp_tables(summary_file):
    """Generate importance tables."""
    import tdub.config
    import json
    from textwrap import dedent
    tdub.config.init_meta_table()
    summary = json.loads(PosixPath(summary_file).read_text())
    imp_gain = summary["importances_gain"]
    imp_split = summary["importances_split"]
    names = imp_gain.keys()
    imp_gain = [round(x, 4) for x in imp_gain.values()]
    imp_split = [round(x, 4) for x in imp_split.values()]
    imp_gain, names, imp_split = (list(reversed(t)) for t in zip(*sorted(zip(imp_gain, names, imp_split))))
    print(dedent("""\
    \\begin{table}[htbp]
      \\begin{center}
        \\caption{XXX}
        \\label{XXX}
        \\begin{tabular}{lcc}
          \\toprule
          Variable & Importance (gain) & Importance (split) \\\\
          \\midrule"""))
    for n, g, s in zip(names, imp_gain, imp_split):
        print("      {} & {} & {} \\\\".format(tdub.config.PLOTTING_META_TABLE["titles"][n]["mpl"], g, s))
    print(dedent("""\
          \\bottomrule
        \\end{tabular}
      \\end{center}
    \\end{table}
    """))


def run_cli():
    import tdub.config
    tdub.config.AVOID_IN_CLF_1j1b = []
    tdub.config.AVOID_IN_CLF_2j1b = []
    tdub.config.AVOID_IN_CLF_2j2b = []
    cli()


if __name__ == "__main__":
    run_cli()

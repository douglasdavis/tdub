"""
tdub command line interface
"""

# stdlib
from pathlib import PosixPath
import logging
import json

# third party
import click

# tdub
from tdub import setup_logging
from tdub.utils import quick_files

setup_logging()
log = logging.getLogger("tdub-cli")

DESCRIPTION = ""
EXECUTABLE = str(PosixPath(__file__).resolve())
BNL_CONDOR_HEADER = """
Universe        = vanilla
notification    = Error
notify_user     = ddavis@phy.duke.edu
GetEnv          = True
Executable      = {exe}
Output          = {out_dir}/logs/job.out.hopt.$(cluster).$(process)
Error           = {out_dir}/logs/job.err.hopt.$(cluster).$(process)
Log             = {out_dir}/logs/job.log.$(cluster).$(process)
request_memory  = 2.0G
"""


@click.group(context_settings=dict(max_content_width=92))
def cli():
    pass


@cli.command("train-single", context_settings=dict(max_content_width=92))
@click.option("-d", "--data-dir", type=click.Path(exists=True), help="directory containing data files", required=True)
@click.option("-r", "--region", type=str, required=True, help="the region to train on")
@click.option("-o", "--out-dir", type=str, required=True, help="output directory name")
@click.option("-n", "--nlo-method", type=str, default="DR", help="tW simluation NLO method", show_default=True)
@click.option("-x", "--override-selection", type=str, help="override selection with contents of file")
@click.option("-t", "--use-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("-i", "--ignore-list", type=str, help="variable ignore list file")
@click.option("-e", "--early-stop", type=int, help="number of early stopping rounds")
@click.option("--learning-rate", type=float, required=True, help="learning_rate model parameter")
@click.option("--num-leaves", type=int, required=True, help="num_leaves model parameter")
@click.option("--min-child-samples", type=int, required=True, help="min_child_samples model parameter")
@click.option("--max-depth", type=int, required=True, help="max_depth model parameter")
@click.option("--n-estimators", type=int, required=True, help="n_estimators model parameter")
def single(data_dir, region, out_dir, nlo_method, override_selection, use_tptrw, ignore_list,
           early_stopping_rounds, learning_rate, num_leaves, min_child_samples, max_depth, n_estimators):
    """Execute a single training round."""
    from tdub.train import single_training, prepare_from_root
    from tdub.utils import get_avoids
    from tdub.frames import drop_cols
    qf = quick_files(data_dir)
    override_sel = override_selection
    if override_sel:
        override_sel = PosixPath(override_sel).read_text().strip()
    df, y, w = prepare_from_root(
        qf[f"tW_{nlo_method}"],
        qf["ttbar"],
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
        out_dir,
        early_stopping_rounds=early_stopping_rounds,
        extra_summary_entries=extra_sum,
    )
    return 0

@cli.command("train-scan", context_settings=dict(max_content_width=140))
@click.argument("config", type=click.Path(exists=True))
@click.argument("data-dir", type=click.Path(exists=True))
@click.option("-r", "--region", type=str, required=True, help="the region to train on")
@click.option("-o", "--out-dir", type=str, required=True, help="output directory name")
@click.option("-n", "--nlo-method", type=str, default="DR", help="tW simluation NLO method", show_default=True)
@click.option("-s", "--script-name", type=str, default="hopt.scan.REGION.sub", help="output script name", show_default=True)
@click.option("-x", "--override-selection", type=str, help="override selection with contents of file")
@click.option("-t", "--use-tptrw", is_flag=True, help="apply top pt reweighting")
@click.option("-i", "--ignore-list", type=str, help="variable ignore list file")
@click.option("-e", "--early-stop", type=int, help="number of early stopping rounds")
def scan(config, data_dir, region, out_dir, nlo_method, script_name, override_selection,
         use_tptrw, ignore_list, early_stopping_rounds):
    """Given CONFIG and DATA_DIR, perform a parameter scan via condor jobs.

    The scan parameters are defined in the CONFIG file, and the data
    to use is in the DATA_DIR. Example:

    $ hopt.py scan conf.json /data/path -r 2j1b -e 10 -o scan_2j1b

    """
    with open(config, "r") as f:
        pd = json.load(f)
    p = PosixPath(out_dir).resolve()
    p.mkdir(exist_ok=False)
    (p / "logs").mkdir(exist_ok=False)
    pname = str(p)
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
                            "-d {} "
                            "-o {}/res{:04d}_{} "
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
                            data_dir,
                            pname,
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
                            early_stopping_rounds
                        )
                        arglist = arglist.replace("-x _NONE ", "")
                        arglist = arglist.replace("-i _NONE ", "")
                        runs.append(arglist)
                        i += 1
    log.info(f"prepared {len(runs)} jobs for submission")
    output_script_name = script_name.replace("REGION", region)
    output_script = PosixPath(output_script_name)
    with output_script.open("w") as f:
        print(BNL_CONDOR_HEADER.format(exe=EXECUTABLE, out_dir=pname), file=f)
        for run in runs:
            print(f"Arguments = train-single {run}\nQueue\n\n", file=f)

    return 0


@cli.command("train-check", context_settings=dict(max_content_width=92))
@click.argument("directory", type=click.Path(exists=True))
@click.option("-p", "--print-top", is_flag=True, help="Print the top results")
@click.option("-n", "--n-res", type=int, default=10, help="Number of top results to print", show_default=True)
def check(directory, print_top, n_res):
    """Check the results of a parameter scan DIRECTORY."""
    from tdub.train import SingleTrainingResult
    results = []
    top_dir = PosixPath(directory)
    for resdir in top_dir.iterdir():
        if resdir.name == "logs" or not resdir.is_dir():
            continue
        summary_file = resdir / "summary.json"
        if not summary_file.exists():
            continue
        with summary_file.open("r") as f:
            summary = json.load(f)
            if summary["bad_ks"]:
                continue
            res = SingleTrainingResult(**summary)
            res.directory = resdir.name
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

    best_res = potentials[0]
    print(best_res)
    print(best_res.summary)
    print(best_res.directory)

    for result in potentials:
        print(result)

    with open(top_dir / "summary.json", "w") as f:
        json.dump(potentials[0].summary, f, indent=4)

    return 0


@cli.command("train-fold", context_settings=dict(max_content_width=92))
@click.option("-s", "--scan-dir", type=click.Path(exists=True), help="scan step's output directory")
@click.option("-d", "--data-dir", type=click.Path(exists=True), help="directory containing data files", required=True)
@click.option("-o", "--out-dir", type=str, help="directory to save output", required=True)
@click.option("-t", "--use-tptrw", is_flag=True, help="use top pt reweighting")
@click.option("-r", "--random-seed", type=int, default=414, help="random seed for folding", show_default=True)
@click.option("-n", "--n-splits", type=int, default=3, help="number of splits for folding", show_default=True)
def fold(scan_dir, data_dir, out_dir, use_tptrw, random_seed, n_splits):
    """Perform a folded training based on a hyperparameter scan result."""
    from tdub.train import folded_training, prepare_from_root
    with open(f"{scan_dir}/summary.json", "r") as f:
        summary = json.load(f)
    nlo_method = summary["nlo_method"]
    best_iteration = summary["best_iteration"]
    if best_iteration > 0:
        summary["all_params"]["n_estimators"] = best_iteration
    region = summary["region"]
    branches = summary["features"]
    selection = summary["selection_used"]
    qf = quick_files(data_dir)
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
        out_dir,
        summary["region"],
        kfold_kw={
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": random_seed,
        },
    )
    return 0


@cli.command("apply-gen-npy", context_settings=dict(max_content_width=92))
@click.option("--bnl", type=str, help="all files in a BNL data directory")
@click.option("--single", type=str, help="a single ROOT file")
@click.option("-f", "--folds", type=str, multiple=True, help="fold output directories")
@click.option("-n", "--arr-name", type=str, help="array name")
@click.option("-o", "--out-dir", type=str, help="save output to directory", required=True)
@click.option("--bnl-script-name", type=str, help="BNL condor submit script name")
def apply_gen_npy(bnl, single, folds, arr_name, out_dir, bnl_script_name):
    """Generate BDT response array(s) and save to .npy file."""
    if single is not None and bnl is not None:
        raise ValueError("can only choose --bnl or --single, not both")

    from tdub.batch import gen_apply_npy_script
    from tdub.apply import generate_npy, FoldedResult
    from tdub.utils import SampleInfo, minimal_branches
    from tdub.frames import raw_dataframe

    out_dir = PosixPath(out_dir)

    if bnl is not None:
        gen_apply_npy_script(EXECUTABLE, bnl, folds, out_dir, arr_name, bnl_script_name)
        return 0

    frs = [FoldedResult(p) for p in folds]
    necessary_branches = ["OS", "elmu", "reg2j1b", "reg2j2b", "reg1j1b"]
    for fold in frs:
        necessary_branches += fold.features
        necessary_branches += minimal_branches(fold.selection_used)
    necessary_branches = sorted(set(necessary_branches), key=str.lower)

    log.info("Loading necessary branches:")
    for nb in necessary_branches:
        log.info(f" - {nb}")

    def process_sample(sample_name):
        stem = PosixPath(sample_name).stem
        sampinfo = SampleInfo(stem)
        tree = f"WtLoop_{sampinfo.tree}"
        df = raw_dataframe(sample_name, tree=tree, branches=necessary_branches)
        npyfilename = out_dir / f"{stem}.{arr_name}.npy"
        generate_npy(frs, df, npyfilename)

    if single is not None:
        process_sample(single)


@cli.command("soverb", context_settings=dict(max_content_width=92))
@click.argument("data-dir", type=click.Path(exists=True))
@click.argument("selections", type=click.Path(exists=True))
def soverb(data_dir, selections):
    """Get signal over background using data in DATA_DIR and a SELECTIONS file.

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

    qf = quick_files(data_dir)
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

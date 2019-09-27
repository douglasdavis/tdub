"""
Module for applying trained models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PosixPath
import json
import logging

import uproot
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold

from tdub.regions import Region
from tdub.frames import specific_dataframe

log = logging.getLogger(__name__)


class FoldedResult:
    """A class to hold the output from a folded training result

    Parameters
    ----------
    datadir : str
       the directory with the training output
    region : Region or str
       the region where the training was performed

    Attributes
    ----------
    model0 : lightgbm.LGBMClassifier
       the model for the 0th fold from training
    model1 : lightgbm.LGBMClassifier
       the model for the 1st fold from training
    model2 : lightgbm.LGBMClassifier
       the model for the 2nd fold from training
    region : Region
       the region for this training
    features : list(str)
       the list of kinematic features used by the model
    folder : sklearn.model_selection.KFold
       the folding object that the training session used

    """

    def __init__(self, datadir: str, region: Union[Region, str]) -> FoldedResult:
        data_path = PosixPath(datadir)
        if not data_path.exists():
            raise ValueError(f"{datadir} does not exit")
        data_path = data_path.resolve()
        self._model0 = joblib.load(data_path / "model_fold0.joblib")
        self._model1 = joblib.load(data_path / "model_fold1.joblib")
        self._model2 = joblib.load(data_path / "model_fold2.joblib")

        if isinstance(region, str):
            if not region.startswith("r"):
                self._region = Region[f"r{region}"]
            else:
                self._region = Region[region]
        else:
            self._region = region

        feature_file = data_path / "features.txt"
        self._features = feature_file.read_text().split("\n")[:-1]

        fold_file = data_path / "kfold.json"
        self._folder = KFold(**(json.loads(fold_file.read_text())))

    @property
    def model0(self) -> lightgbm.LGBMClassifier:
        return self._model0

    @property
    def model1(self) -> lightgbm.LGBMClassifier:
        return self._model1

    @property
    def model2(self) -> lightgbm.LGBMClassifier:
        return self._model2

    @property
    def features(self) -> List[str]:
        return self._features

    @property
    def region(self) -> Region:
        return self._region

    @property
    def folder(self) -> KFold:
        return self._folder

    def apply_to_minor(
        self,
        infile: Union[str, os.PathLike],
        tree: str = "WtLoop_nominal",
        weight_name: str = "weight_nominal",
        outfile: Optional[str] = None,
        use_root_pandas = False,
    ) -> None:
        """apply the folded result to a single minor background file

        Parameters
        ----------
        infile : str or os.PathLike
           the input file to open and apply
        tree : str
           the name of the tree to extract data from
        weight_name : str
           name of the weight branch
        outfile : str (optional)
           output file name
        use_root_pandas : bool
           use root_pandas instead of uproot for writing the file (uproot's
           TTree writing is pretty new and has some kinks to iron out)

        """
        dfim = specific_dataframe(
            infile, self.region, tree=tree, weight_name=weight_name, to_ram=True
        )
        dfim._df = dfim.df[self.features]

        X = dfim.df.to_numpy()
        y0 = self.model0.predict_proba(X)[:, 1]
        y1 = self.model1.predict_proba(X)[:, 1]
        y2 = self.model2.predict_proba(X)[:, 1]
        y = np.mean([y0, y1, y2], axis=0)

        if self.region == Region.r1j1b:
            reg = "reg1j1b"
        elif self.region == Region.r2j1b:
            reg = "reg2j1b"
        elif self.region == Region.r2j2b:
            reg = "reg2j2b"

        reg_branch = np.array([1 for _ in range(dfim.df.shape[0])], dtype=np.int32)
        OS_branch = reg_branch
        weight_branch = dfim.weights.to_numpy()

        if outfile is None:
            outfile = "{}_applied.root".format(str(infile))

        if use_root_pandas:
            import root_pandas
            dfim._df[weight_name] = weight_branch
            dfim._df[reg] = reg_branch
            dfim._df["OS"] = OS_branch
            dfim._df[f"bdt_reponse_{reg}"] = y
            root_pandas.to_root(dfim._df, outfile, key=tree)

        else:
            out_dict = {}
            out_dict[reg] = reg_branch
            out_dict["OS"] = OS_branch
            out_dict[weight_name] = weight_branch
            out_dict[f"bdt_response_{reg}"] = y
            for c in list(dfim.df.columns):
                out_dict[c] = dfim.df[c].to_numpy()

            out_types = {}
            for c, a in out_dict.items():
                if a.dtype == np.uint32:
                    out_dict[c] = out_dict[c].astype(np.int32)
                out_types[c] = uproot.newbranch(out_dict[c].dtype)
                log.info(f"Saving branch {c} with dtype {out_types[c]}")

            with uproot.recreate(outfile) as f:
                f[tree] = uproot.newtree(out_types, flushsize="10 MB")
                f[tree].extend(out_dict, flush=False)

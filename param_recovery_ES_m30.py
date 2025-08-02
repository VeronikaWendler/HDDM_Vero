# parameter recovery - similar to the tutorial by Pan et al. (2025) in their dockerhddm paper

# libraries

# import libraries  
import pandas as pd
import numpy as np
import hddm
import os, sys, pickle, time
import datetime
import math
import scipy as sp
import matplotlib
matplotlib.use("Agg")                   # for backend (does not require GUI)
import os, pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import itertools
#import pp
import joblib
from IPython import embed as shell
import hddm
import kabuki
import statsmodels.formula.api as sm
from patsy import dmatrix
from joblib import Parallel, delayed
import time
import arviz as az
# patch: make a dummy _gdbm module so “import _gdbm” never fails
import types, sys
sys.modules.setdefault('winreg', types.ModuleType('winreg'))
sys.modules.setdefault('_gdbm', types.ModuleType('_gdbm'))
import dill as pickle
from copy import deepcopy   # for modfiying z to be 0.55 (like in Sebastian's Matlab)
import argparse
# warning settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Plotting
# Stats 
from statsmodels.distributions.empirical_distribution import ECDF
# HDDM
from hddm.simulators.hddm_dataset_generators import simulator_h_c
from pathlib import Path
# scitnific computing and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import statsmodels.api as sm

# HDDM related packages
import pymc as pm
import hddm
import kabuki
import arviz as az

#------------------------------------------------------------------------------------------------------------------------------------------
PROJECT_DIR   = pathlib.Path("/workspace").resolve()
BASE_MODEL_DIR = PROJECT_DIR / "models_dir_combined"

# --------------------------------------------------------------------------
# helpers
def load_chains(model_dir: pathlib.Path, stem: str, n: int = 3) -> az.InferenceData:
    chains = [az.from_netcdf(model_dir / f"{stem}_{i}.nc") for i in range(n)]
    return az.concat(chains, dim="chain")


def az_summary(inf: az.InferenceData) -> pd.DataFrame:
    """One row per subject, columns = mean of each parameter."""
    df  = (az.summary(inf, kind="stats")
             .reset_index(names="param_name"))
    pat = r"(?P<param>.*)_subj\.(?P<subj>\d+)"
    df  = (df
           .assign(**df["param_name"].str.extract(pat))
           .dropna(subset=["subj"])
           .astype({"subj": int}))
    wide = df.pivot(index="subj", columns="param", values="mean")
    wide.index.name = "subj_idx"
    return wide.reset_index()


def _get(series: pd.Series, pattern: str, cat: str):
    """Robust lookup for a(high) vs a_high etc."""
    cand1 = pattern.format(cat)                       # a(high)
    cand2 = (cand1.replace("(", "_").replace(")", "")
                   .replace(":", "_").replace("[", "_").replace("]", ""))
    for c in (cand1, cand2):
        if c in series:
            return series[c]
    raise KeyError(f"{cand1} not found.")


def simulate_from_subject_params(raw: pd.DataFrame,
                                 pars: pd.DataFrame) -> pd.DataFrame:
    """Simulate one synthetic trial for every real trial."""
    pars = pars.set_index("subj_idx")
    rows = []

    for (subj, ov), trials in raw.groupby(["subj_idx", "OVcate"]):
        p      = pars.loc[subj]
        v_int  = p["v_Intercept"]
        v_c    = p["v_z_IAW_chart"]
        v_i    = p["v_z_IAW_image"]
        v_att  = _get(p, "v_z_AttentionW:C(OVcate)[{}]", ov)
        a_val  = _get(p, "a({})", ov)
        t_val  = p["t"]

        for _, tr in trials.iterrows():
            v_trial = (v_int +
                       v_att * tr.get("z_AttentionW", 0.) +
                       v_c   * tr.get("z_IAW_chart", 0.) +
                       v_i   * tr.get("z_IAW_image", 0.))

            sim_tr, _ = hddm.generate.gen_rand_data(
                {"v": v_trial, "a": a_val, "t": t_val},
                size=1, subjs=1)

            sim_tr["subj_idx"]     = subj
            sim_tr["OVcate"]       = ov
            rows.append(sim_tr)

    sim = pd.concat(rows, ignore_index=True)

    # drop subjects with *all* NaN RT
    bad = sim.groupby("subj_idx")["rt"].apply(lambda s: s.isna().all())
    sim = sim[~sim["subj_idx"].isin(bad[bad].index)]

    return sim


def fit_recovery_model(data: pd.DataFrame, out_dir: pathlib.Path):
    v_reg = {"model": "v ~ 1 + z_AttentionW:C(OVcate) + z_IAW_chart + z_IAW_image",
             "link_func": lambda x: x}

    mdl = hddm.HDDMRegressor(data,
                             [v_reg],
                             depends_on={"a": "OVcate"},
                             include=["a", "t", "v"],
                             p_outlier=0.05,
                             keep_regressor_trace=True)
    mdl.find_starting_values()
    mdl, inf = mdl.sample(1000, burn=100,
                          dbname=str(out_dir / "mES_30_recovery"),
                          db="pickle", return_infdata=True,
                          ppc=True, loglike=True)
    az.to_netcdf(inf, out_dir / "mES_30_recovery.nc")
    return mdl


# --------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    inf  = load_chains(BASE_MODEL_DIR, "combined_replication_ES_30")
    pars = az_summary(inf)                 # ⬅ contains a(high), a(low), …

    raw  = (inf.observed_data
                 .to_dataframe()
                 .reset_index(drop=True)
                 .assign(subj_idx=lambda d: d["subj_idx"].astype(int)))

    sim  = simulate_from_subject_params(raw, pars)
    print(f"Simulated trials : {len(sim):,}")
    print(f"Subjects kept    : {sim['subj_idx'].nunique()}")

    fit_recovery_model(sim, BASE_MODEL_DIR)


if __name__ == "__main__":
    main()






#------------------------------------------
# subject_summary = az_summary(es27_infdata)['mean'].reset_index(names=['subj_idx'])
# df_ind_summary = (
#     data_ES_27.groupby(['subj_idx', 'OVcate'])['rt']
#     .describe()
#     .reset_index()
# )
# df_ind_summary = (
#     df_ind_summary.set_index('subj_idx')
#     .join(subject_summary.set_index('subj_idx'))
#     .reset_index()
# )

# # get group/subject parameters for the weighted terms (if needed)
# # e.g., you might want to pull them per trial from df_ind_summary
# sim_data = []

# for (subj, ov), trial_group in data_ES_27.groupby(['subj_idx', 'OVcate']):
#     j = df_ind_summary[(df_ind_summary['subj_idx'] == subj) & (df_ind_summary['OVcate'] == ov)].iloc[0]
#     v_int = j["v_Intercept"]
#     v_vald = j["v_val_diff"]
#     v_DwellPA = j["v_DwellPropAdvantage"]
#     v_gquad = j["v_gaze_quad"]
#     a_int = j["a_Intercept"]
#     a_abs_DwellPAov = j[f"a_abs_DwellPropAdv:C(OVcate)[{ov}]"]
#     t_val = j["t"]

#     for _, trial in trial_group.iterrows():
#         val_diff_trial = trial.get("val_diff", 0)
#         DwellPA_trial = trial.get("DwellPropAdvantage", 0)
#         gaze_quad_trial = trial.get("gaze_quad", 0)
#         abs_DwellPAov_trial = trial.get("abs_DwellPropAdv", 0)

#         # weighted drift and boundary
#         v_trial = v_int + v_vald * val_diff_trial + v_DwellPA * DwellPA_trial + v_gquad * gaze_quad_trial
#         a_trial = a_int + a_abs_DwellPAov * abs_DwellPAov_trial

#         # simulate (you can do multiple repeats if desired)
#         sim_trial, _ = hddm.generate.gen_rand_data(
#             {"v": v_trial, "a": a_trial, "t": t_val},
#             size=1, subjs=1
#         )
#         sim_trial["subj_idx"] = subj
#         sim_trial["OVcate"] = ov
#         sim_trial["val_diff"] = val_diff_trial
#         sim_trial["DwellPropAdvantage"] = DwellPA_trial
#         sim_trial["gaze_quad"] = gaze_quad_trial
#         sim_trial["abs_DwellPropAdv"] = abs_DwellPAov_trial

#         sim_data.append(sim_trial)

# sim_data = pd.concat(sim_data, ignore_index=True)

# # Final guard: enforce subj_idx <= 20 in sim_data
# sim_data['subj_idx'] = pd.to_numeric(sim_data['subj_idx'], errors='coerce').astype(int)
# sim_data = sim_data[sim_data['subj_idx'] <= 20]

# if "condition" in sim_data.columns:
#     sim_data.drop("condition", axis=1, inplace=True)

# # Diagnostics
# print(sim_data.head(10))
# print("\nUnique subjects in simulation:", sorted(sim_data['subj_idx'].unique()))
# print("OVcate counts in simulation:\n", sim_data['OVcate'].value_counts())


# print(sim_data.to_string())


# # ---------- subject-level summary (no OVcate) ----------
# subject_summary = az_summary(es27_infdata)["mean"].reset_index(names="subj_idx")

# # rt summary per subject
# df_ind_summary = (
#     data_ES_27.groupby("subj_idx")["rt"]
#     .describe()
#     .reset_index()
# )

# # Merge subject-level posterior means with the RT summary
# df_ind_summary = df_ind_summary.merge(subject_summary, on="subj_idx", how="inner")

# # ---------- simulate trial-by-trial using those subject-level estimates ----------
# sim_data_list = []
# for subj, trial_group in data_ES_27.groupby("subj_idx"):
#     j = df_ind_summary[df_ind_summary["subj_idx"] == subj].iloc[0]

#     v_int = j["v_Intercept"]
#     v_vald = j["v_val_diff"]
#     v_valbal = j["v_val_bal_int"]          # from your orthogonalised term
#     t_val = j["t"]
#     a_val = j.get("a_Intercept", j.get("a", None))  # adjust depending on what your summary names it

#     for _, trial in trial_group.iterrows():
#         val_diff_trial = trial.get("val_diff", 0)
#         val_bal_trial = trial.get("val_bal_int", 0)

#         # drift with interaction-style structure
#         v_trial = v_int + v_vald * val_diff_trial + v_valbal * val_bal_trial

#         sim_trial, _ = hddm.generate.gen_rand_data(
#             {"v": v_trial, "a": a_val, "t": t_val},
#             size=1, subjs=1
#         )
#         sim_trial["subj_idx"] = subj
#         sim_trial["val_diff"] = val_diff_trial
#         sim_trial["val_bal_int"] = val_bal_trial

#         sim_data_list.append(sim_trial)

# sim_data = pd.concat(sim_data_list, ignore_index=True)

# # final filtering etc.
# sim_data["subj_idx"] = pd.to_numeric(sim_data["subj_idx"], errors="coerce").astype(int)
# sim_data = sim_data[sim_data["subj_idx"] <= 20]
# if "condition" in sim_data.columns:
#     sim_data.drop("condition", axis=1, inplace=True)

# # diagnostics
# print(sim_data.head(10))
# print("\nUnique subjects in simulation:", sorted(sim_data["subj_idx"].unique()))
# print(sim_data.to_string())




#-----------------------------------------------------------------------------------------------------------------------------------
#recovered_nc = os.path.join(BASE_MODEL_DIR, "mES_27_recovery.nc")
#m_recovery_infdata = az.from_netcdf(recovered_nc)

#------------------------------------------------------------------------------------------------------------------------------------



# #-------------------------------------------------------------------------------------------------------------------
# print(sim_data[['subj_idx', 'OVcate', 'rt', 'response']].head())
# print("\nUnique subjects:", sim_data['subj_idx'].nunique())
# print("OVcate counts:\n", sim_data['OVcate'].value_counts())
# #-----------------------------------------------------------------------------------------------------------------------
# #Re-Fitting the model (like in hcp tutorial)

# # helper function to wrap the sampling procedure
# def run_sampling(model, model_db_name, progress_bar=True):
#     model.find_starting_values()
#     result = model.sample(
#         1000,                # nr of samples (
#         burn=100,            # Burn-in samples
#         dbname=model_db_name,# path for saving the chain
#         db='pickle',         # Save chain using pickle
#         return_infdata=True, # Return an InferenceData object for diagnostics/plots
#         loglike=True,        # allow for loglikelihood computation
#         ppc=True             # to get the ppc
#     )
#     if isinstance(result, tuple):
#         model_out = result[0]
#         infdata = result[1]
#         return model_out, infdata
#     else:
#         return model, result

# # model specification (best fitting OV-modulated Inattention and t model)
# v_reg = {'model': 'v ~ 1 + val_diff + DwellPropAdvantage + gaze_quad', 'link_func': lambda x: x}
# a_reg = {'model': 'a ~ 1 + abs_DwellPropAdv:C(OVcate)', 'link_func': lambda x: x }
# reg_descr = [v_reg, a_reg]

# # HDDMRegressor using simulated data
# m_recovery = hddm.HDDMRegressor(
#     sim_data,              
#     reg_descr,             
#     include=['a', 't', 'v'], 
#     p_outlier=0.05,
#     group_only_regressors=False,
#     keep_regressor_trace=True
# )

# # full path for saving
# model_db_name = os.path.join(BASE_MODEL_DIR, "mES_27_recovery")
# m_recovery, m_recovery_infdata = run_sampling(m_recovery, model_db_name, progress_bar=False)
# az.to_netcdf(m_recovery_infdata, os.path.join(BASE_MODEL_DIR, "mES_27_recovery.nc"))

# #-----------------------------------------------------------------------------------------------------------------------------------
# #recovered_nc = os.path.join(BASE_MODEL_DIR, "mES_27_recovery.nc")
# #m_recovery_infdata = az.from_netcdf(recovered_nc)

# #------------------------------------------------------------------------------------------------------------------------------------


# #-------------------------------------------------------------------------------------------------------------------
# print(sim_data[['subj_idx', 'rt', 'response']].head())
# print("\nUnique subjects:", sim_data['subj_idx'].nunique())
# #-----------------------------------------------------------------------------------------------------------------------
# #Re-Fitting the model (like in hcp tutorial)

# # helper function to wrap the sampling procedure
# def run_sampling(model, model_db_name, progress_bar=True):
#     model.find_starting_values()
#     result = model.sample(
#         1000,                # nr of samples (
#         burn=100,            # Burn-in samples
#         dbname=model_db_name,# path for saving the chain
#         db='pickle',         # Save chain using pickle
#         return_infdata=True, # Return an InferenceData object for diagnostics/plots
#         loglike=True,        # allow for loglikelihood computation
#         ppc=True             # to get the ppc
#     )
#     if isinstance(result, tuple):
#         model_out = result[0]
#         infdata = result[1]
#         return model_out, infdata
#     else:
#         return model, result

# # model specification (best fitting OV-modulated Inattention and t model)
# v_reg = {'model': 'v ~ 1 + val_diff + val_bal_int', 'link_func': lambda x: x}
# reg_descr = [v_reg]

# # HDDMRegressor using simulated data
# m_recovery = hddm.HDDMRegressor(
#     sim_data,              
#     reg_descr,             
#     include=['a', 't', 'v'], 
#     p_outlier=0.05,
#     group_only_regressors=False,
#     keep_regressor_trace=True
# )

# # full path for saving
# model_db_name = os.path.join(BASE_MODEL_DIR, "mES_29_recovery")
# m_recovery, m_recovery_infdata = run_sampling(m_recovery, model_db_name, progress_bar=False)
# az.to_netcdf(m_recovery_infdata, os.path.join(BASE_MODEL_DIR, "mES_29_recovery.nc"))

#-----------------------------------------------------------------------------------------------------------------------------------
#recovered_nc = os.path.join(BASE_MODEL_DIR, "mES_27_recovery.nc")
#m_recovery_infdata = az.from_netcdf(recovered_nc)

#------------------------------------------------------------------------------------------------------------------------------------


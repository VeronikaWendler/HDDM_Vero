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
# dir
PROJECT_DIR   = pathlib.Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# BASE_MODEL_DIR = PROJECT_DIR / "models_dir_garcia"
# FIG_DIR_ROOT   = PROJECT_DIR / "figures_dir_garcia/garcia_replication_ES_27/diagnostics"
BASE_MODEL_DIR = PROJECT_DIR / "models_dir_combined"
FIG_DIR_ROOT   = PROJECT_DIR / "figures_dir_combined/combined_replication_ES_30/diagnostics"


chain0 = az.from_netcdf(BASE_MODEL_DIR / "combined_replication_ES_30_0.nc")
chain1 = az.from_netcdf(BASE_MODEL_DIR / "combined_replication_ES_30_1.nc")
chain2 = az.from_netcdf(BASE_MODEL_DIR / "combined_replication_ES_30_2.nc")


es27_infdata = az.concat([ chain0, chain1, chain2], dim="chain")


#--------------------------------------------------------------------------------------------------------------------------------------------
#REG PLOT FUNCTION
# AZ SUMMARY
# some functions for some plots
def regplot_with_corr(
    data=None,
    x="x",
    y="y",
    cor_anonot=True,
    reg_anonot=True,
    annot_kws={
        "fontsize": 8,
        "xy": (0.95, 0.05),
        "ha": 'right',
        "va": 'bottom'
    },
    scatter_kws={
        's': 40,
        "alpha": 0.4
    },
    ax=None,
    **kwargs
):
    """

    Example:
    --------
    >>> Example usage
    >>> import pandas as pd
    >>> data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 3, 5, 7, 11]})
    >>> regplot_with_corr(data)
    >>> plt.show()
    """
    if ax is None:
        ax = plt.gca()
    if data is not None:
        data_x = data[x]
        data_y = data[y]
    else: 
        data_x = x
        data_y = y

    # Plot regression line and scatter plot
    sns.regplot(
        x=data_x,
        y=data_y,
        ci=None if len(np.unique(data_y)) == 1 else 95,
        scatter_kws=scatter_kws,
        ax=ax
    )

    annot_text = ""
    if cor_anonot:
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(data_x, data_y)
        # if np.isnan(correlation):
        #     correlation = 0
        # if np.isnan(p_value):
        #     p_value = 1
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        annot_text += f"$r={correlation:.2f}$\n${p_str}$"

    if reg_anonot:
        # Calculate regression coefficients
        X = sm.add_constant(data_x)  # Adds a constant term to the predictor
        model = sm.OLS(data_y, X).fit()
        intercept, slope = model.params
        annot_text += f"\n$\\beta_0={intercept:.2f}$\n$\\beta_1={slope:.2f}$"

    # Annotate the plot with correlation, p-value, intercept, and slope
    if annot_text != "":
        ax.annotate(
            annot_text,
            **annot_kws,
            xycoords='axes fraction',
            bbox=dict(
                boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'
            )
        )
    
    return ax


def az_summary(infdata=None, half_a=False, param_names_order=None, **kwargs):

    param_df = az.summary(infdata, kind="stats",
                            **kwargs).reset_index(names="param_name")
    # col_values = ['mean', 'sd', "hdi_3%", "hdi_97%"]
    col_values = list(param_df.columns[1:5])

    pattern = r'(.*)_subj\.(\d+)'

    param_df[['param',
                'subj_idx']] = param_df['param_name'].str.extract(pattern)
    # param_df['param'] = param_df['param'].apply(lambda x: f'${x}$')
    param_df = param_df.dropna(subset=['subj_idx'])
    param_df['subj_idx'] = param_df['subj_idx'].astype(int)

    if half_a:
        param_df.loc[param_df['param'] == 'a',
                        col_values] = param_df.loc[param_df['param'] == 'a',
                                                col_values] / 2

    param_df = param_df.pivot(
        index='subj_idx', columns='param', values=col_values
    )

    if param_names_order is not None:
        new_index = pd.MultiIndex.from_tuples(
            [
                (level_0, param) for level_0 in col_values
                for param in param_names_order
            ],
            names=[None, 'param']
        )
        param_df = param_df.reindex(columns=new_index)

    param_df.reset_index(inplace=True)
    param_df.columns.names = [None, None]

    return param_df

#  
#-----------------------------------------------------------------------------------------------------------------------------------------
summary_df = az_summary(es27_infdata)['mean']          
# … but push the index out into a real column
summary_df = summary_df.reset_index()                  
print(summary_df.columns.tolist())

# read in model
data_ES_27 = es27_infdata.observed_data.to_dataframe().reset_index(drop=True)

# read in model and coerce subj_idx
data_ES_27 = es27_infdata.observed_data.to_dataframe().reset_index(drop=True)
data_ES_27 = data_ES_27.copy()
data_ES_27['subj_idx'] = pd.to_numeric(data_ES_27['subj_idx'], errors='coerce')
data_ES_27 = data_ES_27.dropna(subset=['subj_idx'])
data_ES_27['subj_idx'] = data_ES_27['subj_idx'].astype(int)


def az_summary_group(infdata, **kwargs):
    # full summary as a DataFrame
    summary_df = az.summary(infdata, kind="stats", **kwargs).reset_index()
    # Check the actual column name for parameters (might be 'index' instead of 'param_name')
    param_col = 'index' if 'index' in summary_df.columns else 'param_name'
    # Set the index to the parameter names and select the mean estimates
    return summary_df.set_index(param_col)["mean"]

# check what's in the infdata
group_params = az_summary_group(es27_infdata)
print(group_params.index.tolist())


group_params = az.summary(es27_infdata, var_names=['~subj', '~std'], filter_vars='regex')
subject_params = az_summary(es27_infdata)['mean']


            
subject_summary = az_summary(es27_infdata)['mean'].reset_index(names=['subj_idx'])
df_ind_summary = (
    data_ES_27.groupby(['subj_idx', 'OVcate'])['rt']
    .describe()
    .reset_index()
)
df_ind_summary = (
    df_ind_summary.set_index('subj_idx')
    .join(subject_summary.set_index('subj_idx'))
    .reset_index()
)

# get group/subject parameters for the weighted terms (if needed)
# e.g., you might want to pull them per trial from df_ind_summary
# ------------------------------------------------------------------
# ----  Build a single table of subject-level parameters  ----------
# (summary_df already has one row per subject, columns are parameters)
subj_par  = summary_df.set_index('subj_idx')    #  rows = subj_idx

# quick sanity check
needed_cols = ['v_Intercept', 'v_z_IAW_chart', 'v_z_IAW_image',
               't', 'a(high)', 'a(medium)', 'a(low)',
               'v_z_AttentionW:C(OVcate)[high]',
               'v_z_AttentionW:C(OVcate)[medium]',
               'v_z_AttentionW:C(OVcate)[low]']
missing = [c for c in needed_cols if c not in subj_par.columns]
assert not missing, f"parameter(s) missing from summary_df: {missing}"

# ------------------------------------------------------------------
# ----  simulate ----------------------------------------------------
sim_rows = []   # list of data frames (one per simulated trial)

for (subj, ov), trial_grp in data_ES_27.groupby(['subj_idx', 'OVcate']):
    
    # -------- fixed parameters for THIS subject & OV ---------------
    p        =  subj_par.loc[subj]                     # row of parameters
    v_int    =  p['v_Intercept']
    v_zIAWc  =  p['v_z_IAW_chart']
    v_zIAWi  =  p['v_z_IAW_image']
    v_zAttW  =  p[f'v_z_AttentionW:C(OVcate)[{ov}]']   # e.g. “high”
    t_val    =  p['t']
    a_val    =  p[f'a({ov})']                          # “a(high)” etc.
    
    # -------- iterate over that subject’s trials -------------------
    for _, tr in trial_grp.iterrows():
        # trial-wise predictors
        zIAWc_trial  = tr.get('z_IAW_chart',  0)
        zIAWi_trial  = tr.get('z_IAW_image',  0)
        zAttW_trial  = tr.get('z_AttentionW', 0)

        # drift & simulate one response
        v_trial = (v_int +
                   v_zAttW * zAttW_trial +
                   v_zIAWc * zIAWc_trial +
                   v_zIAWi * zIAWi_trial)

        sim_tr, _ = hddm.generate.gen_rand_data(
            dict(v=v_trial, a=a_val, t=t_val),
            size=1, subjs=1
        )

        # annotate
        sim_tr['subj_idx']      = subj
        sim_tr['OVcate']        = ov
        sim_tr['z_IAW_chart']   = zIAWc_trial
        sim_tr['z_IAW_image']   = zIAWi_trial
        sim_tr['z_AttentionW']  = zAttW_trial
        sim_rows.append(sim_tr)

# ------------------------------------------------------------------
# ----  concatenate & clean  ---------------------------------------
sim_data = pd.concat(sim_rows, ignore_index=True)

dead_subs = (sim_data.groupby('subj_idx')['rt']
                     .apply(lambda s: s.isna().all()))   # True = all NaN
if dead_subs.any():
    sim_data = sim_data[~sim_data['subj_idx'].isin(dead_subs[dead_subs].index)]

# (optional) if you instead want to drop subjects with *any* NaN RT:
# bad_subs = sim_data.loc[sim_data['rt'].isna(),'subj_idx'].unique()
# sim_data = sim_data[~sim_data['subj_idx'].isin(bad_subs)]

# ----------  tidy-up & quick diagnostics ----------------------------
sim_data = sim_data.loc[:, ~sim_data.columns.isin(['condition'])]

print(sim_data.head())
print("subjects kept:", sim_data['subj_idx'].nunique(),
      "| dropped:", dead_subs.sum())

print(sim_data.to_string())


print(sim_data[['subj_idx', 'OVcate', 'rt', 'response']].head())
print("\nUnique subjects:", sim_data['subj_idx'].nunique())
print("OVcate counts:\n", sim_data['OVcate'].value_counts())
#-----------------------------------------------------------------------------------------------------------------------
#Re-Fitting the model (like in hcp tutorial)

# helper function to wrap the sampling procedure
def run_sampling(model, model_db_name, progress_bar=True):
    model.find_starting_values()
    result = model.sample(
        1000,                # nr of samples (
        burn=100,            # Burn-in samples
        dbname=model_db_name,# path for saving the chain
        db='pickle',         # Save chain using pickle
        return_infdata=True, # Return an InferenceData object for diagnostics/plots
        loglike=True,        # allow for loglikelihood computation
        ppc=True             # to get the ppc
    )
    if isinstance(result, tuple):
        model_out = result[0]
        infdata = result[1]
        return model_out, infdata
    else:
        return model, result

# model specification 
v_reg = {'model': 'v ~ 1 + z_AttentionW:C(OVcate) + z_IAW_chart + z_IAW_image', 'link_func': lambda x: x}
reg_descr = [v_reg]
depends_on={'a': 'OVcate'} 
            

# HDDMRegressor using simulated data
m_recovery = hddm.HDDMRegressor(
    sim_data,              
    reg_descr,
    depends_on=depends_on,         
    include=['a', 't', 'v'], 
    p_outlier=0.05,
    group_only_regressors=False,
    keep_regressor_trace=True
)

# full path for saving
model_db_name = os.path.join(BASE_MODEL_DIR, "mES_30_recovery")
m_recovery, m_recovery_infdata = run_sampling(m_recovery, model_db_name, progress_bar=False)
az.to_netcdf(m_recovery_infdata, os.path.join(BASE_MODEL_DIR, "mES_30_recovery.nc"))







































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


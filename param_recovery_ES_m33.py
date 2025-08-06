# import libraries  
import pandas as pd
import numpy as np
import hddm
import os, sys, pickle, time
import datetime
import math
import re
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

BASE_MODEL_DIR = PROJECT_DIR / "models_dir_garcia"
FIG_DIR_ROOT   = PROJECT_DIR / "figures_dir_garcia/garcia_replication_ES_34/diagnostics"

chain0 = az.from_netcdf(BASE_MODEL_DIR / "garcia_replication_ES_34_0.nc")
chain1 = az.from_netcdf(BASE_MODEL_DIR / "garcia_replication_ES_34_1.nc")
chain2 = az.from_netcdf(BASE_MODEL_DIR / "garcia_replication_ES_34_2.nc")

es27_infdata = az.concat([ chain0, chain1, chain2], dim="chain")

#--------------------------------------------------------------------------------------------------------------------------------------------


def az_summary(infdata=None, half_a=False, param_names_order=None, **kwargs):

    param_df = az.summary(infdata, kind="stats",
                            **kwargs).reset_index(names="param_name")
    # col_values = ['mean', 'sd', "hdi_3%", "hdi_97%"]
    col_values = list(param_df.columns[1:5])

    pattern = r'(.*)_subj\.(\d+)'
    param_df[['param', 'subj_idx']] = param_df['param_name'].str.extract(pattern)

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
print(summary_df.columns.tolist())

# read in model
data_ES_27 = es27_infdata.observed_data.to_dataframe().reset_index(drop=True)

# read in model and coerce subj_idx
data_ES_27 = es27_infdata.observed_data.to_dataframe().reset_index(drop=True)
data_ES_27 = data_ES_27.copy()
data_ES_27['subj_idx'] = pd.to_numeric(data_ES_27['subj_idx'], errors='coerce')
data_ES_27 = data_ES_27.dropna(subset=['subj_idx'])
data_ES_27['subj_idx'] = data_ES_27['subj_idx'].astype(int)

bad_raw = data_ES_27.groupby('subj_idx')['rt'].apply(lambda s: s.isna().all())
if bad_raw.any():
     data_ES_27 = data_ES_27[~data_ES_27['subj_idx'].isin(bad_raw[bad_raw].index)]
# ---------------------------------------------------------------------

orig_subjects = sorted(data_ES_27['subj_idx'].unique())

# Subject-level summary restricted to kept subjects
subject_summary = az_summary(es27_infdata)['mean'].reset_index(names=['subj_idx'])
subject_summary = subject_summary[subject_summary['subj_idx'].isin(orig_subjects)]


def az_summary_group(infdata, **kwargs):
    # full summary as a DataFrame
    summary_df = az.summary(infdata, kind="stats", **kwargs).reset_index()
    param_col = 'index' if 'index' in summary_df.columns else 'param_name'
    return summary_df.set_index(param_col)["mean"]  # we should do something like select the most likely parameter values (for all params instead of mean as Amir said - seems to be a good approach )

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

# group  mapping for a / or t 
# a_params = {}
# for param in group_params.index:
#     if param.startswith('a_subj'):
#         parts = param.split(')')
#         ov_level = parts[0].split('(')[1]
#         subj_id = int(parts[1].split('.')[1])
#         if subj_id not in a_params:
#             a_params[subj_id] = {}
#         a_params[subj_id][ov_level] = group_params.loc[param, 'mean']


sim_data = []

for (subj, ov), trial_group in data_ES_27.groupby(['subj_idx', 'OVcate']):
    j = df_ind_summary[(df_ind_summary['subj_idx'] == subj) & (df_ind_summary['OVcate'] == ov)].iloc[0]
    v_int = j["v_Intercept"]
    v_chartInatt = j[f"v_z_IAW_chart:C(OVcate)[{ov}]"]
    v_imageInatt = j[f"v_z_IAW_image:C(OVcate)[{ov}]"]
    v_att = j["v_z_AttentionW"]
    a_val = j[f"a_OVcate[T.{ov}]"]
    t_val = j["t"]
    
    # try:
    #     a_val = a_params[int(subj)][ov.lower()]
    # except KeyError:
    #     a_val = group_params.loc[f'a({ov.lower()})', 'mean']
        
    for _, trial in trial_group.iterrows():
        v_chartInatt_trial = trial.get("z_IAW_chart", 0)
        v_imageInatt_trial = trial.get("z_IAW_image", 0)
        v_att_trial = trial.get("z_AttentionW", 0)

        # weighted drift and boundary
        v_trial = v_int + v_att * v_att_trial + v_chartInatt * v_chartInatt_trial + v_imageInatt * v_imageInatt_trial

        sim_trial, _ = hddm.generate.gen_rand_data(
            {"v": v_trial, "t": t_val, "a": a_val},
            size=1, subjs=1
        )
        sim_trial["subj_idx"] = subj
        sim_trial["OVcate"] = ov
        sim_trial["z_IAW_chart"] = v_chartInatt_trial
        sim_trial["z_IAW_image"] = v_imageInatt_trial
        sim_trial["z_AttentionW"] = v_att_trial

        sim_data.append(sim_trial)

sim_data = pd.concat(sim_data, ignore_index=True)

# # dropping subs with nan simulations
bad_sim = sim_data.groupby('subj_idx')['rt'].apply(lambda s: s.isna().all())
if bad_sim.any():
    sim_data = sim_data[~sim_data['subj_idx'].isin(bad_sim[bad_sim].index)]

sim_data['subj_idx'] = pd.to_numeric(sim_data['subj_idx'], errors='coerce').astype(int)

if "condition" in sim_data.columns:
    sim_data.drop("condition", axis=1, inplace=True)

# Diagnostics
print(sim_data.head(10))
print("\nUnique subjects in simulation:", sorted(sim_data['subj_idx'].unique()))
print("OVcate counts in simulation:\n", sim_data['OVcate'].value_counts())

print(sim_data.to_string())


#-------------------------------------------------------------------------------------------------------------------
print(sim_data[['subj_idx', 'OVcate', 'rt', 'response']].head())
print("\nUnique subjects:", sim_data['subj_idx'].nunique())
print("OVcate counts:\n", sim_data['OVcate'].value_counts())
#-----------------------------------------------------------------------------------------------------------------------
#Re-Fitting the model (like in hcp tutorial)

# helper function to wrap the sampling procedure
def run_sampling(model, model_db_name, progress_bar=True):
    model.find_starting_values()
    result = model.sample(
        2000,                # nr of samples 
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


v_reg = {'model': 'v ~ 1 + z_AttentionW + z_IAW_chart:C(OVcate) + z_IAW_image:C(OVcate)', 'link_func': lambda x: x}
a_reg = {'model': 'a ~ 1 + OVcate', 'link_func': lambda x: x}
reg_descr = [v_reg, a_reg]
#depends_on={'a': 'OVcate'} 

# HDDMRegressor using simulated data
m_recovery = hddm.HDDMRegressor(
    sim_data,              
    reg_descr,   
    include=['t', 'a', 'v'], 
    p_outlier=0.05,
    group_only_regressors=False,
    keep_regressor_trace=True
)

model_db_name = os.path.join(BASE_MODEL_DIR, "mES_combined_34_recovery")
m_recovery, m_recovery_infdata = run_sampling(m_recovery, model_db_name, progress_bar=False)
az.to_netcdf(m_recovery_infdata, os.path.join(BASE_MODEL_DIR, "mES_combined_34_recovery.nc"))




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

BASE_MODEL_DIR = PROJECT_DIR / "models_dir_combined"
FIG_DIR_ROOT   = PROJECT_DIR / "figures_dir_combined/combined_replication_ES_1/diagnostics"

chain0 = az.from_netcdf(BASE_MODEL_DIR / "combined_replication_ES_1_0.nc")
chain1 = az.from_netcdf(BASE_MODEL_DIR / "combined_replication_ES_1_1.nc")
chain2 = az.from_netcdf(BASE_MODEL_DIR / "combined_replication_ES_1_2.nc")

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
print(summary_df.columns.tolist())

# read in model
data_ES_27 = es27_infdata.observed_data.to_dataframe().reset_index(drop=True)

# Normalize OVcate in the observed data early to avoid whitespace/casing mismatches
data_ES_27['OVcate'] = data_ES_27['OVcate'].astype(str).str.strip()

# Subject-level summary (means) once
subject_summary = az_summary(es27_infdata)['mean'].reset_index(names=['subj_idx'])

# Per-(subj, OVcate) RT summary
df_ind_summary = (
    data_ES_27.groupby(['subj_idx', 'OVcate'])['rt']
    .describe()
    .reset_index()
)

# Merge in subject-level parameter estimates
df_ind_summary = df_ind_summary.merge(subject_summary, on='subj_idx', how='left')

# Diagnostics: which subjects are in data but missing from df_ind_summary
data_subjects = set(data_ES_27['subj_idx'].unique())
df_subjects = set(df_ind_summary['subj_idx'].unique())
missing = sorted(data_subjects - df_subjects)
if missing:
    print("WARNING: these subjects are in data_ES_27 but missing from df_ind_summary:", missing)
print("Unique subjects in data_ES_27:", sorted(data_subjects))
print("Unique subjects in df_ind_summary:", sorted(df_subjects))

sim_data = []

for (subj, ov), trial_group in data_ES_27.groupby(['subj_idx', 'OVcate']):
    ov = str(ov).strip()

    matching = df_ind_summary[
        (df_ind_summary['subj_idx'] == subj) & (df_ind_summary['OVcate'].astype(str).str.strip() == ov)
    ]
    if matching.empty:
        print(f"Missing summary for subj {subj}, OVcate '{ov}' — skipping.")
        continue
    if len(matching) > 1:
        print(f"Warning: multiple summary rows for subj {subj}, OVcate '{ov}'; using first.")

    j = matching.iloc[0]
    
    # safe extraction helper
    def safe_get(j, key, default=0.0):
        if key in j.index:
            return float(j[key])
        else:
            print(f"Warning: {key} missing for subj={subj}, OVcate={ov}. Using {default}")
            return float(default)

    v_int = safe_get(j, "v_Intercept")
    v_vald = safe_get(j, "v_val_diff")
    v_DwellPA = safe_get(j, "v_DwellPropAdvantage")
    v_gquad = safe_get(j, "v_gaze_quad")
    a_int = safe_get(j, "a_Intercept")
    if ov == "high":
        a_effect = 0.0
    else:
        a_effect = safe_get(j, f"a_OVcate[T.{ov}]")

    t_val = safe_get(j, "t")

    for _, trial in trial_group.iterrows():
        val_diff_trial = trial.get("val_diff", 0)
        DwellPA_trial = trial.get("DwellPropAdvantage", 0)
        gaze_quad_trial = trial.get("gaze_quad", 0)

        v_trial = v_int + v_vald * val_diff_trial + v_DwellPA * DwellPA_trial + v_gquad * gaze_quad_trial
        a_trial = a_int + a_effect


        sim_trial, _ = hddm.generate.gen_rand_data(
            {"v": float(v_trial), "a": float(a_trial), "t": float(t_val)},
            size=1, subjs=1
        )
        sim_trial["subj_idx"] = subj
        sim_trial["OVcate"] = ov
        sim_trial["val_diff"] = val_diff_trial
        sim_trial["DwellPropAdvantage"] = DwellPA_trial
        sim_trial["gaze_quad"] = gaze_quad_trial
        sim_trial["a_used"] = a_trial  # record what boundary was used

        sim_data.append(sim_trial)


sim_data = pd.concat(sim_data, ignore_index=True)

sim_data['subj_idx'] = pd.to_numeric(sim_data['subj_idx'], errors='coerce').astype(int)

sim_data = sim_data.groupby('subj_idx').filter(lambda g: not g['rt'].isna().all())

# ---------------------------------------------------------------------
if "condition" in sim_data.columns:
    sim_data.drop("condition", axis=1, inplace=True)
    
before = sim_data['subj_idx'].nunique()
#sim_data = sim_data.groupby('subj_idx').filter(lambda g: not g['rt'].isna().all())
after = sim_data['subj_idx'].nunique()
print(f"Removed {before - after} subjects with all-NaN RTs (kept {after}).")


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

# model specification (best fitting OV-modulated Inattention and t model)
v_reg = {'model': 'v ~ 1 + val_diff + DwellPropAdvantage + gaze_quad', 'link_func': lambda x: x}
a_reg = {'model': 'a ~ 1 + OVcate', 'link_func': lambda x: x }
reg_descr = [v_reg, a_reg]

# HDDMRegressor using simulated data
m_recovery = hddm.HDDMRegressor(
    sim_data,              
    reg_descr,             
    include=['a', 't', 'v'], 
    p_outlier=0.05,
    group_only_regressors=False,
    keep_regressor_trace=True
)

# full path for saving
model_db_name = os.path.join(BASE_MODEL_DIR, "mES_1_combined_recovery_nr2")
m_recovery, m_recovery_infdata = run_sampling(m_recovery, model_db_name, progress_bar=False)
az.to_netcdf(m_recovery_infdata, os.path.join(BASE_MODEL_DIR, "mES_1_combined_recovery_nr2.nc"))


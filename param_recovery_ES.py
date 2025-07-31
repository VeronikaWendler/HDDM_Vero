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

BASE_MODEL_DIR = PROJECT_DIR / "models_dir_garcia"
FIG_DIR_ROOT   = PROJECT_DIR / "figures_dir_garcia/garcia_replication_ES_27/diagnostics"


#-------------------------------------------------------------------------------------------------------------------------------------------   
# here also improatant, set which model to lead (concatenate the chains you ran)
chain0 = az.from_netcdf(BASE_MODEL_DIR / "garcia_replication_ES_27_0.nc")
chain1 = az.from_netcdf(BASE_MODEL_DIR / "garcia_replication_ES_27_1.nc")
chain2 = az.from_netcdf(BASE_MODEL_DIR / "garcia_replication_ES_27_2.nc")

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
        param_df.loc[param_df['param'] == 'a_Intercept',
                        col_values] = param_df.loc[param_df['param'] == 'a_Intercept',
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


#-----------------------------------------------------------------------------------------------------------------------------------------
summary_df = az_summary(es27_infdata)['mean']
print(summary_df.columns.tolist())

# read in model
data_ES_27 = es27_infdata.observed_data.to_dataframe().reset_index(drop=True)
df_ind_summary = data_ES_27.groupby(['subj_idx','OVcate'])['rt'].describe().reset_index()
df_ind_summary = df_ind_summary.set_index('subj_idx').join(
    az_summary(es27_infdata)['mean'].reset_index(names=['subj_idx']).set_index('subj_idx')
    ).reset_index()
df_ind_summary.head()

print(es27_infdata.groups())
#------------------------------------------------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------------------------------------------------
# SIMULATION 
#---------------------------------------------------------------------------------------------------------------------------------------------


group_params = az.summary(es27_infdata, var_names=['~subj', '~std'], filter_vars='regex')
subject_params = az_summary(es27_infdata)['mean']

sim_data = pd.DataFrame()

# loop over subject × OVcate groups
for i, j in df_ind_summary.groupby(['subj_idx', 'OVcate']):
    subj, ov = i  # unpack for readability

    t_val     = j["t"].iloc[0]
    v_int     = j["v_Intercept"].iloc[0]
    v_vald    = j["v_val_diff"].iloc[0]
    v_DwellPA = j["v_DwellPropAdvantage"].iloc[0]
    v_gquad   = j["v_gaze_quad"].iloc[0]
    a_val     = j["a_Intercept"].iloc[0]
    a_DwellPAov   = j[f"a_abs_DwellPropAdv:C(OVcate)[{ov}]"].iloc[0]

    v = v_int + v_vald + v_DwellPA + v_gquad                                                  # NO trial-level multipliers
    bound = a_val + a_DwellPAov
    n_trials = int(j["count"].iloc[0])
    

    data, params = hddm.generate.gen_rand_data(
        {
            "v": v,
            "a": bound,
            "t": t_val
        },
        size=n_trials,  # same number of trials as in real data
        subjs=1,
    )

    # Add subject/condition labels
    data[["subj_idx", "OVcate"]] = i  
    data["val_diff"] = v_vald
    data["DwellPropAdvantage"] = v_DwellPA
    data["abs_DwellPropAdv"] = a_DwellPAov
    data["gaze_quad"] = v_gquad

    sim_data = pd.concat([sim_data, data], ignore_index=True)

# Drop condition column if present
if "condition" in sim_data.columns:
    sim_data.drop("condition", axis=1, inplace=True)

# check simulated data
print(sim_data.head(10))
print("\nUnique subjects in simulation:", sim_data['subj_idx'].nunique())
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
a_reg = {'model': 'a ~ 1 + abs_DwellPropAdv:C(OVcate)', 'link_func': lambda x: x }
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
model_db_name = os.path.join(BASE_MODEL_DIR, "mES_27_recovery")
m_recovery, m_recovery_infdata = run_sampling(m_recovery, model_db_name, progress_bar=False)
az.to_netcdf(m_recovery_infdata, os.path.join(BASE_MODEL_DIR, "mES_27_recovery.nc"))

#-----------------------------------------------------------------------------------------------------------------------------------
#recovered_nc = os.path.join(BASE_MODEL_DIR, "mES_27_recovery.nc")
#m_recovery_infdata = az.from_netcdf(recovered_nc)

#------------------------------------------------------------------------------------------------------------------------------------


# Veronika Wendler
# 22.01.25
# code for the attentional drift diffusion model - originally, I used this in summer 2024 in Quebec and is a very broad modification of code from Dr Jan WIllem De Gee, but by now I'd call it my code
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
from IPython import embed as shells
import hddm
import kabuki
import statsmodels.formula.api as sm
from patsy import dmatrix
from joblib import Parallel, delayed
import time
import arviz as az
import dill as pickle
import re
# warning settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Plotting
# Stats 
from statsmodels.distributions.empirical_distribution import ECDF
# HDDM
from hddm.simulators.hddm_dataset_generators import simulator_h_c

from pathlib import Path

PROJECT_DIR = pathlib.Path(os.getenv("PROJECT_DIR", "/workspace"))

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# sanitizing the saving function:
import re
from pathlib import Path

import arviz as az

import os
# disable _all_ Numba JIT caching & compilation
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numba
numba.config.CACHE_ENABLE = False

#------------------------------------------------------------------------------------------------------------------

# addm regression formula: this is how v is influenced by the value, behavioural data and gaze
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2,low ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ϵ
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2,low ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ β3 x (gazeopt -gazesub)
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2,low ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ β3 x (gazeS -gazeE)

# params:
version = 8     # set which version you want to run
run = False       # if True, the the models run, if False the models load

phase = ['For_paper']  #['ES', 'EE']  #

# determine whether to use a single phase or the combined ESEE model or LEESEE
if set(phase) == {'ES', 'EE'}:
    phase_key = 'ESEE'  
elif set(phase) == {'LE', 'ES', 'EE'}:
    phase_key = 'LEESEE'
elif len(phase) == 1:
    phase_key = phase[0]  
else:
    raise ValueError(f"Invalid phase selection: {phase}")

# update the phase variable
phase = phase_key

nr_models = 3 
nr_samples = 1200
parallel = True

# dir
PROJECT_DIR   = pathlib.Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

BASE_MODEL_DIR = PROJECT_DIR / "models_dir_OV"
FIG_DIR_ROOT   = PROJECT_DIR / "figures_dir_OV"


model_base_name = 'OV_replication_'

model_versions = {
    'LE': ['LE_1', 'LE_2', 'LE_3', 'LE_4', 'LE_5', 'LE_6', 'LE_7'],
    'ES': ['ES_1', 'ES_2', 'ES_3', 'ES_4', 'ES_5', 'ES_6', 'ES_7'],
    'EE': ['EE_1', 'EE_2', 'EE_3', 'EE_4', 'EE_5'],
    'ESEE': ['ESEE_1', 'ESEE_2', 'ESEE_3', 'ESEE_4', 'ESEE_5'],
    'LEESEE': ['LEESEE_1', 'LEESEE_2', 'LEESEE_3', 'LEESEE_4', 'LEESEE_5'],
    'ES_VAL': ["ES_VAL_1", "ES_VAL_2", "ES_VAL_3","ES_VAL_4", "ES_VAL_5", "ES_VAL_6", "ES_VAL_7", "ES_VAL_8", "ES_VAL_8", "ES_VAL_9", "ES_VAL_10", "ES_VAL_11"],
    "For_paper": ["For_paper_0", "For_paper_1","For_paper_2","For_paper_3","For_paper_4","For_paper_5","For_paper_6","For_paper_7","For_paper_8","For_paper_9","For_paper_10","For_paper_11", "For_paper_12", "For_paper_13" ],
    "LE_RL": ["LE_RL_1", "LE_RL_2"],
}

if phase not in model_versions:
    raise ValueError(f"Invalid phase selection '{phase}'. Choose from: {list(model_versions.keys())}")

model_name = model_versions[phase][version]

data = pd.read_csv((PROJECT_DIR / "data_sets" / "data_sets_OV" / "OVParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv").as_posix(), sep=",")


PHASE_TO_SOURCE = {
    "ES_ZBIAS": "ES", 
    "ES_quad": "ES",    
    "LE_RL": "LE",
    "ES_VAL": "ES",
    "For_paper": "ES",
}


#data filtering
if phase == 'ESEE':
    data = data[data['phase'].isin(['ES', 'EE'])]  # include both ES and EE trials
elif phase == 'LEESEE':
    data = data[data['phase'].isin(['LE', 'ES', 'EE'])] # include LE ES and EE phases - might not be super realistic to run a ddm for a RL task but I'll just see what happens
else:
    data = data[data['phase'] == phase]  # no change


data["phase"] = data["phase"].astype("category")

# preparing the data to fit the modelling framework
data["rt"] = pd.to_numeric(data['rtime'], errors='coerce')  

# exclude RTs below 0.250 immediately
data = data[data["rt"] > 0.250]

#Check min and max RT values after filtering
print("Min RT after filtering:", data['rt'].min())
print("Max RT after filtering:", data['rt'].max())

data['response'] = pd.to_numeric(data['corr'], errors = 'coerce')
data["OVcate"] = data['OVcate_2'].astype("category")
data["Abscate"] = data['Abscate_2'].astype("category")
data["cond"] = data["cond"].fillna(-1)
data["cond"] = data["cond"].astype("int")

data["AttentionW"] = pd.to_numeric(data["AttentionW"], errors = 'coerce')
data["InattentionW"] = pd.to_numeric(data["InattentionW"], errors = 'coerce')
data["subj_idx"] = data['sub_id']

# data["feedback"] = pd.to_numeric(data["feedback"], errors = 'coerce')
# data["feedback"] = data["feedback"].astype(float)
### LE phase specific (this is if you are interested in RL models)
# Process 'split_by' only for phase == 'LE'
# data.loc[data["phase"] == "LE", "split_by"] = pd.to_numeric(data.loc[data["phase"] == "LE", "split_by"], errors='coerce').astype("Int64")
# data.loc[data["phase"] == "LE", "trial"] = pd.to_numeric(data.loc[data["phase"] == "LE", "trial"], errors='coerce').astype("Int64")
# data.loc[data["phase"] == "LE", "q_init"] = pd.to_numeric(data.loc[data["phase"] == "LE", "q_init"], errors='coerce').astype(float)

#data = data.dropna(subset= ['feedback','split_by', 'trial', 'q_init'])
# participant exclusion set
exclude_part = {6, 14, 20, 26, 2, 9, 18}

#data = data[data['phase'] == phase]

data = data[~data['subj_idx'].isin(exclude_part)]    
data = data.dropna(subset=['rt', 'response', 'OVcate', 'Abscate', 'subj_idx', 'AttentionW', 'InattentionW', 'cond'])


# debugging information
print(f"\nFiltering data for phase: {phase}")
print("Unique phases in filtered data:", data['phase'].unique())
print(f"Data shape after filtering: {data.shape}")
print(f"Unique participants in filtered data: {data['subj_idx'].unique()}")
# to get the number of trials per category of OV:
category_counts = data['OVcate'].value_counts()
print("\nOVcate Category Counts:\n", category_counts)
print(f"Selected phase_key: {phase_key}")
print(f"Model to run: {model_base_name + model_name}")
print(f"Filtered Data Unique Phases: {data['phase'].unique()}")
print(f"Data Shape After Filtering: {data.shape}")


# # this is the response histogram for correct and incorrect responses - don't use it flips errors like in the tutorials (CCT didn't use it)
# data.loc[data['response'] == 0, 'rt'] = -data.loc[data['response'] == 0, 'rt']
# fig = plt.figure()
# ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
# ax.set_xlim(-10, 10)
# for i, subj_data in data.groupby('subj_idx'):
#     subj_data['rt'].hist(bins=20, histtype='step', ax=ax)
# plt.show()
# #data
#Flipping Errors only for EE and ES phases the RL model does not work on this
# data = hddm.utils.flip_errors(data)
    
# Plotting RT distributions
fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
for _, subj_data in data.groupby('subj_idx'):
    subj_data.rt.hist(bins=20, histtype='step', ax=ax)
# instead of plt.show():
fig.savefig((FIG_DIR_ROOT / f"{model_base_name}{model_name}" / "diagnostics" / "rt_distributions.pdf").as_posix(),
            bbox_inches="tight")
plt.close(fig)

# ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# model dir:
model_dir = BASE_MODEL_DIR
ensure_dir(model_dir)

def sanitize_infdata(infdata):
    """Convert pd.NA values to np.nan in all groups of the InferenceData object (important for if you have columns which you don't use, for example)."""
    for group in infdata._groups_all:
        if hasattr(infdata, group):
            dataset = getattr(infdata, group)
            for var in dataset.data_vars:
                values = dataset[var].values
                if isinstance(values, np.ndarray) and values.dtype == "object":
                    mask = pd.isna(values)
                    if mask.any():
                        print(f"Sanitizing variable '{var}' in group '{group}' (contains pd.NA)")
                        values[mask] = np.nan
                        dataset[var].values = values
    return infdata


def _sanitize_filename(fname):
    # replace any of : ( ) [ ] , with underscore
    safe = re.sub(r'[:\(\)\[\],]', '_', fname)
    # collapse runs of underscores to a single underscore
    safe = re.sub(r'_+', '_', safe)
    return safe

def _inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))

def _summ_from_samples(arr_1d):
    arr = np.asarray(arr_1d).ravel()
    qs = np.percentile(arr, [2.5, 25, 50, 75, 97.5])
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr, ddof=1)),
        "2.5q": qs[0],
        "25q":  qs[1],
        "50q":  qs[2],
        "75q":  qs[3],
        "97.5q":qs[4],
    }


# 1000 draws from the posterior for PPC instead of mean, SD (for For_model 7) 

def export_posterior_draws(model_name, model_dir, n_jobs=3, S=1000):
    # load and combine chains
    idatas = [az.from_netcdf(Path(model_dir) / f"{model_name}_{i}.nc") for i in range(n_jobs)]
    idata  = az.concat(idatas, dim="chain")
    post = idata.posterior.stack(sample=("chain","draw"))

    n_samps = post.sizes["sample"]
    idx = np.random.choice(n_samps, size=min(S, n_samps), replace=False)
    post_s = post.isel(sample=idx)
    
    # data frame
    all_params = list(post_s.data_vars)
    df_all = post_s[all_params].to_dataframe().reset_index(drop=True)
    out_csv = Path(model_dir) / f"{model_name}_posterior_draws.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"Saved {df_all.shape[0]} draws × {df_all.shape[1]} columns to {out_csv}")

    return out_csv

fig_dir = FIG_DIR_ROOT / f"{model_base_name}{model_name}"
ensure_dir(fig_dir / "diagnostics")

# try:
#     os.system('mkdir {}'.format(fig_dir))
#     os.system('mkdir {}'.format(os.path.join(fig_dir, 'diagnostics')))
# except:
#     pass

# fig_dir = FIG_DIR_ROOT / full_model_name
# ensure_dir(fig_dir / "diagnostics")

## subjects
subjects = np.unique(data.subj_idx)
nr_subjects = subjects.shape[0]
print(nr_subjects)
print(subjects)


#########################################################################################################
# drift diffusion models
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# function that runs the different versions of DDM regressions

def run_model(trace_id, data, model_dir, model_name, version, samples=11000, accuracy_coding=True):  #shoukd be 5000 samples but can be changed depending on computing capacities
    import os
    import numpy as np
    import hddm
    from patsy import dmatrix  

    ensure_dir(model_dir)   
    
    depends_on = {}
    
    if phase == 'LE':
        if version == 0:     # r1 # this is the 0 model with fully fixed parameters across OV levels
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # r2 fixated option weights varies by OV level 
            v_reg = {'model': 'v ~ 1 + AttentionW:C(cond) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  #r3  non-fixated option weights varies by OV level
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(cond)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: # r4 non-fixated options weights varies by OV level and boundary separation
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(cond)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'cond'}      
        elif version == 4: # r5 r4 non-fixated options weights varies by OV level and ndt
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(cond)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'cond'}
        elif version == 5: # r5 r4 non-fixated options weights varies by OV level and ndt
            v_reg = {'model': 'v ~ 1 + AttentionW:C(cond) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'cond'}
        elif version == 6: # r5 r4 non-fixated options weights varies by OV level and ndt
            v_reg = {'model': 'v ~ 1 + AttentionW:C(cond) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'cond'}         
        elif version == 7:
            v_reg = {'model': 'v ~ 1 + AttentionW:C + InattentionW:C(cond) + gazeCI', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 8:
            v_reg = {'model': 'v ~ 1 + AttentionW:C + InattentionW:C + gazeCI:C(cond)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'cond'}
        elif version == 9:
            v_reg = {'model': 'v ~ 1 + AttentionW:C + InattentionW:C + gazeSE:C(cond)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'cond'}
        else:
            raise ValueError(f"check version {version} ??")

        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True)
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,      #is variable
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    
    elif phase == 'ES':
        accuracy_coding = True
        if version == 0:     # r1 # this is the 0 model with fully fixed parameters across OV levels
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # r2 fixated option weights varies by OV level 
            v_reg = {'model': 'v ~ 1 + AttentionW:C(OVcate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  #r3  non-fixated option weights varies by OV level
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: # r4 non-fixated options weights varies by OV level and boundary separation
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}      
        elif version == 4:
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate'} 
        elif version == 5:
            v_reg = {'model': 'v ~ 1 + AttentionW:C(OVcate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}      
        elif version == 6:
            v_reg = {'model': 'v ~ 1 + AttentionW:C(OVcate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'OVcate'}   
        else: # r6
            v_reg = {'model': 'v ~ 1 + AttentionW:C(Abscate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True
                                    )
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,                 #500
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    elif phase == 'EE':
        accuracy_coding = True
        if version == 0:     # r1 # this is the 0 model with fully fixed parameters across OV levels
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # r2 fixated option weights varies by OV level 
            v_reg = {'model': 'v ~ 1 + AttentionW:C(OVcate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  #r3  non-fixated option weights varies by OV level
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: # r4 non-fixated options weights varies by OV level and boundary separation
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}      
        elif version == 4: # r5
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate'}      
        
        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True
                                    )
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata

    elif phase == 'ESEE':  # Combined model for ES + EE
        accuracy_coding = True
        if version == 0:  # Baseline model with fixed parameters across phases
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # Drift rate varies by phase (ES vs. EE)
            v_reg = {'model': 'v ~ 1 + AttentionW:C(phase) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  # Non-fixated option weights vary by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 3:  # Boundary separation varies by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'phase'}
        elif version == 4:  # Non-decision time varies by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'phase'}
        else:
            raise ValueError(f"Invalid version {version}")
     
        
        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True
                                    )
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    elif phase == 'LEESEE':  # Combined model for LE + ES + EE
        accuracy_coding = True
        if version == 0:  # Baseline model with fixed parameters across phases
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # Drift rate varies by phase (LE vs ES vs. EE)
            v_reg = {'model': 'v ~ 1 + AttentionW:C(phase) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  # Non-fixated option weights vary by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 3:  # Boundary separation varies by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'phase'}
        elif version == 4:  # Non-decision time varies by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'phase'}
        else:
            raise ValueError(f"Invalid version {version}")
     
        
        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    depends_on=depends_on, 
                                    p_outlier=.05, 
                                    include=['a', 't', 'v'],
                                    group_only_regressors=False,
                                    keep_regressor_trace=True
                                    )
        m.find_starting_values()
        infdata = m.sample(samples,
                   burn=1000,
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    
    
    elif phase == "ES_VAL":
        depends_on = {}
        if version == 0:   
            v_reg = {'model': 'v ~ Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:
            v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 3:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 4:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}    
        elif version == 5:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        # S is upper boundary
        elif version == 6:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 7:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 8:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 9:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E:C(OVcate) + ES_InattentionW_S:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main function for running/loading models

###############################################################################################################


import dill as pickle  # to create the pkl object

def drift_diffusion_hddm(data, 
                         samples=600,
                         n_jobs=3,
                         run=True,
                         parallel=True,
                         model_name='model',
                         model_dir='.', 
                         version=version,
                         phase=phase,
                         accuracy_coding=True):

    if run:
        if parallel:
            start_time = time.time()
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_model)(trace_id,
                                   data,
                                   model_dir,
                                   model_name,
                                   version, 
                                   samples,
                                   accuracy_coding
                                   ) 
                for trace_id in range(n_jobs)
            )
            print("Time elapsed:", time.time() - start_time, "s")
            
            # for i in range(n_jobs):
            #     model = results[i]
                
            #     #HDDM format
            #     model.save(os.path.join(model_dir, f"{model_name}_{i}.hddm"))

            #     with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
            #         pickle.dump(model, f)  
                    
            for i in range(n_jobs):
                model, infdata = results[i]
                model.save(os.path.join(model_dir, f"{model_name}_{i}.hddm"))
                with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
                    pickle.dump(model, f)
                infdata = sanitize_infdata(infdata)  #clean
                az.to_netcdf(infdata, os.path.join(model_dir, f"{model_name}_{i}.nc"))


        else:
            # model = run_model(1,
            #                   data,
            #                   model_dir,
            #                   model_name,
            #                   version, 
            #                   samples,
            #                   accuracy_coding 
            #                   )
            
            # model.save(os.path.join(model_dir, model_name + ".hddm"))

            # with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
            #     pickle.dump(model, f)  
            
            model, infdata = run_model(1,
                                       data,
                                       model_dir,
                                       model_name,
                                       version, 
                                       samples,
                                       accuracy_coding 
                                       )
            model.save(os.path.join(model_dir, model_name + ".hddm"))
            with open(os.path.join(model_dir, f"{model_name}.pkl"), "wb") as f:
                pickle.dump(model, f)
            infdata = sanitize_infdata(infdata)
            az.to_netcdf(infdata, os.path.join(model_dir, f"{model_name}.nc"))

    else:
        print('Loading existing models')
        models = [hddm.load(os.path.join(model_dir, f"{model_name}_{i}.hddm")) for i in range(n_jobs)]
        return models
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
# for the RL models (if used)
import dill as pickle

def drift_diffusion_hddmRL(data, 
                         samples=11000, #6000
                         n_jobs=5,
                         run=True,
                         parallel=True,
                         model_name='model',
                         model_dir='.', 
                         version=version,
                         phase=phase,
                         accuracy_coding=True):

    if run:
        if parallel:
            start_time = time.time()
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_model)(trace_id,
                                   data,
                                   model_dir,
                                   model_name,
                                   version, 
                                   samples,
                                   accuracy_coding
                                   ) 
                for trace_id in range(n_jobs)
            )
            print("Time elapsed:", time.time() - start_time, "s")
            
            for i in range(n_jobs):
                model = results[i]
                
                # Save in HDDM format
                model.save(os.path.join(model_dir, f"{model_name}_{i}.hddm"))

                with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
                    model = pickle.load(f)

        else:
            model = run_model(1,
                              data,
                              model_dir,
                              model_name,
                              version, 
                              samples,
                              accuracy_coding 
                              )
            
            model.save(os.path.join(model_dir, model_name + ".hddm"))

            with open(os.path.join(model_dir, model_name + ".pkl"), 'wb') as f:
                pickle.dump(model, f)

    else:
        print('Loading existing models')
        # models = [hddm.load(os.path.join(model_dir, f"{model_name}_{i}.hddm")) for i in range(n_jobs)]
        # return models
      
        infdatas = []
        for i in range(n_jobs):
            nc_path = os.path.join(model_dir, f"{model_name}_{i}.nc")
            infdatas.append(az.from_netcdf(nc_path))
        return infdatas

    

######################################################################################################################################
# analyzing the models   


full_model_name = model_base_name + model_name
fig_dir        = FIG_DIR_ROOT / full_model_name
ensure_dir(fig_dir)
ensure_dir(fig_dir/"diagnostics")


def analyze_model(models, fig_dir, nr_models, version, phase):
    # 'sns.set_theme(style='darkgrid', font='sans-serif', font_scale=0.5)
    # # combine the 3 modles with kabuki utils
    # combined_model = kabuki.utils.concat_models(models)'
    
    print(f"Analyzing {len(models)} models for {phase}, version {version}")
    print(f"Saving figures to: {fig_dir}")
    sns.set_theme(style='darkgrid', font='sans-serif', font_scale=0.5)

    # Check if models are valid
    if not models or models[0] is None:
        print("ERROR: Models are empty or invalid.")
        return

    # combinine models
    try:
        combined_model = kabuki.utils.concat_models(models)
        print("Models combined successfully.")
    except Exception as e:
        print(f"Error combining models: {e}")
        return
    # plot and name parameters of interest (I'm sure this can be done more intelligently but for now it seems to work)

    if phase == 'LE':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
                'Boundary sep.',
                'Non-dec. time',
                'Intercept drift rate',
                'Drift AttentionW',
                'Drift InattentionW',
                'starting point'
            ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(cond)[0]',
            'v_AttentionW:C(cond)[1]',
            'v_AttentionW:C(cond)[2]',
            'v_AttentionW:C(cond)[3]',
            'v_InattentionW'
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(cond)[0]_subj', 
            'v_AttentionW:C(cond)[1]_subj',
            'v_AttentionW:C(cond)[2]_subj',
            'v_AttentionW:C(cond)[3]_subj',
            'v_InattentionW_subj'
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(cond)[90/70]',
            'Drift AttentionW:C(cond)[80/40]', 
            'Drift AttentionW:C(cond)[60/20]',
            'Drift AttentionW:C(cond)[30/10]',
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(cond)[0]',
            'v_InattentionW:C(cond)[1]', 
            'v_InattentionW:C(cond)[2]',
            'v_InattentionW:C(cond)[3]',
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(cond)[0]_subj',
            'v_InattentionW:C(cond)[1]_subj', 
            'v_InattentionW:C(cond)[2]_subj',
            'v_InattentionW:C(cond)[3]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(cond)[90/70]',
            'Drift InattentionW:C(cond)[80/40]',
            'Drift InattentionW:C(cond)[60/20]',
            'Drift InattentionW:C(cond)[30/10]'
            ]
        elif version == 3:
            params_of_interest = [
            'a(0)',
            'a(1)',
            'a(2)',
            'a(3)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(cond)[0]',
            'v_InattentionW:C(cond)[1]',
            'v_InattentionW:C(cond)[2]',
            'v_InattentionW:C(cond)[3]',
            ]
            params_of_interest_s = [
            'a(0)_subj',
            'a(1)_subj',
            'a(2)_subj',
            'a(3)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[0]_subj',
            'v_InattentionW:C(OVcate)[1]_subj',
            'v_InattentionW:C(OVcate)[2]_subj',
            'v_InattentionW:C(OVcate)[3]_subj',
            ]
            titles = [
            'Boundary sep. (0)',
            'Boundary sep. (1)',
            'Boundary sep. (2)',
            'Boundary sep. (3)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(cond)[90/70]',
            'Drift InattentionW:C(cond)[80/40]',
            'Drift InattentionW:C(cond)[60/20]',
            'Drift InattentionW:C(cond)[30/10]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(0)',
            't(1)',
            't(2)',
            't(3)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(cond)[0]',
            'v_InattentionW:C(cond)[1]',
            'v_InattentionW:C(cond)[2]',
            'v_InattentionW:C(cond)[3]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(0)_subj',
            't(1)_subj',
            't(2)_subj',
            't(3)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[0]_subj',
            'v_InattentionW:C(OVcate)[1]_subj',
            'v_InattentionW:C(OVcate)[2]_subj',
            'v_InattentionW:C(OVcate)[3]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (0',
            'Non-dec. time (1)',
            'Non-dec. time (2)',
            'Non-dec. time (3)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[90/70]',
            'Drift InattentionW:C(OVcate)[80/40]',
            'Drift InattentionW:C(OVcate)[60/20]',
            'Drift InattentionW:C(OVcate)[30/10]',
            ]
            
        elif version == 5:
            params_of_interest = [
            't',
            'a(0)',
            'a(1)',
            'a(2)',
            'a(3)',
            'v_Intercept',
            'v_InattentionW',
            'v_AttentionW:C(cond)[0]',
            'v_AttentionW:C(cond)[1]',
            'v_AttentionW:C(cond)[2]',
            'v_AttentionW:C(cond)[3]',
            ]
            params_of_interest_s = [
            't_subj',
            'a(0)_subj',
            'a(1)_subj',
            'a(2)_subj',
            'a(3)_subj',
            'v_Intercept_subj',
            'v_InattentionW_subj',
            'v_AttentionW:C(cond)[0]_subj',
            'v_AttentionW:C(cond)[1]_subj',
            'v_AttentionW:C(cond)[2]_subj',
            'v_AttentionW:C(cond)[3]_subj',
            ]
            titles = [
            'Non-dec. time',
            'Boundary sep. (0)',
            'Boundary sep. (1)',
            'Boundary sep. (2)',
            'Boundary sep. (3)',
            'Intercept drift rate',
            'Drift InattentionW',
            'Drift AttentionW:C(cond)[90/70]',
            'Drift AttentionW:C(cond)[80/40]',
            'Drift AttentionW:C(cond)[60/20]',
            'Drift AttentionW:C(cond)[30/10]',
            ]
        elif version == 6:
            params_of_interest = [
            'a',
            't(0)',
            't(1)',
            't(2)',
            't(3)',
            'v_Intercept',
            'v_InattentionW',
            'v_AttentionW:C(cond)[0]',
            'v_AttentionW:C(cond)[1]',
            'v_AttentionW:C(cond)[2]',
            'v_AttentionW:C(cond)[3]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(0)_subj',
            't(1)_subj',
            't(2)_subj',
            't(3)_subj',
            'v_Intercept_subj',
            'v_InattentionW_subj',
            'v_AttentionW:C(cond)[0]_subj',
            'v_AttentionW:C(cond)[1]_subj',
            'v_AttentionW:C(cond)[2]_subj',
            'v_AttentionW:C(cond)[3]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (0)',
            'Non-dec. time (1)',
            'Non-dec. time (2)',
            'Non-dec. time (3)',
            'Intercept drift rate',
            'Drift InattentionW',
            'Drift AttentionW:C(cond)[90/70]',
            'Drift AttentionW:C(cond)[80/40]',
            'Drift AttentionW:C(cond)[60/20]',
            'Drift AttentionW:C(cond)[30/10]',
            ]
    if phase == 'ES':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW',
            ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(OVcate)[low]',
            'v_AttentionW:C(OVcate)[medium]',
            'v_AttentionW:C(OVcate)[high]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(OVcate)[low]_subj', 
            'v_AttentionW:C(OVcate)[medium]_subj',
            'v_AttentionW:C(OVcate)[high]_subj',
            'v_InattentionW_subj', 
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(OVcate)[low]',
            'Drift AttentionW:C(OVcate)[medium]', 
            'Drift AttentionW:C(OVcate)[high]',
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]', 
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj', 
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 3:
            params_of_interest = [
            'a(low)',
            'a(medium)',
            'a(high)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a(low)_subj',
            'a(medium)_subj',
            'a(high)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep. (low OVcate)',
            'Boundary sep. (medium OVcate)',
            'Boundary sep. (high OVcate)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(low)',
            't(medium)',
            't(high)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(low)_subj',
            't(medium)_subj',
            't(high)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (low OVcate)',
            'Non-dec. time (medium OVcate)',
            'Non-dec. time (high OVcate)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 5:
            params_of_interest = [
            't',
            'a(low)',
            'a(medium)',
            'a(high)',
            'v_Intercept',
            'v_AttentionW:C(OVcate)[low]',
            'v_AttentionW:C(OVcate)[medium]',
            'v_AttentionW:C(OVcate)[high]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            't_subj',
            'a(low)_subj',
            'a(medium)_subj',
            'a(high)_subj',
            'v_Intercept_subj',
            'v_AttentionW:C(OVcate)[low]_subj', 
            'v_AttentionW:C(OVcate)[medium]_subj',
            'v_AttentionW:C(OVcate)[high]_subj',
            'v_InattentionW_subj', 
            ]
            titles = [
            'Non-dec. time',
            'Boundary sep. (low OVcate)',
            'Boundary sep. (medium OVcate)',
            'Boundary sep. (high OVcate)',
            'Intercept drift rate',
            'Drift AttentionW:C(OVcate)[low]',
            'Drift AttentionW:C(OVcate)[medium]', 
            'Drift AttentionW:C(OVcate)[high]',
            'Drift InattentionW', 
            ]
        elif version == 6:
            params_of_interest = [
            'a',
            't(low)',
            't(medium)',
            't(high)',
            'v_Intercept',
            'v_AttentionW:C(OVcate)[low]',
            'v_AttentionW:C(OVcate)[medium]',
            'v_AttentionW:C(OVcate)[high]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            'a_subj',
            't(low)_subj',
            't(medium)_subj',
            't(high)_subj',
            'v_Intercept_subj',
            'v_AttentionW:C(OVcate)[low]_subj', 
            'v_AttentionW:C(OVcate)[medium]_subj',
            'v_AttentionW:C(OVcate)[high]_subj',
            'v_InattentionW_subj', 
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (low OVcate)',
            'Non-dec. time (medium OVcate)',
            'Non-dec. time (high OVcate)',
            'Intercept drift rate',
            'Drift AttentionW:C(OVcate)[low]',
            'Drift AttentionW:C(OVcate)[medium]', 
            'Drift AttentionW:C(OVcate)[high]',
            'Drift InattentionW', 
            ]
    if phase == 'EE':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW',
            ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(OVcate)[low]',
            'v_AttentionW:C(OVcate)[medium]',
            'v_AttentionW:C(OVcate)[high]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(OVcate)[low]_subj', 
            'v_AttentionW:C(OVcate)[medium]_subj',
            'v_AttentionW:C(OVcate)[high]_subj',
            'v_InattentionW_subj'
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(OVcate)[low]',
            'Drift AttentionW:C(OVcate)[medium]', 
            'Drift AttentionW:C(OVcate)[high]',
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]', 
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj', 
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 3:
            params_of_interest = [
            'a(low)',
            'a(medium)',
            'a(high)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a(low)_subj',
            'a(medium)_subj',
            'a(high)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj'
            
            ]
            titles = [
            'Boundary sep. (low OVcate)',
            'Boundary sep. (medium OVcate)',
            'Boundary sep. (high OVcate)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(low)',
            't(medium)',
            't(high)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(OVcate)[low]',
            'v_InattentionW:C(OVcate)[medium]',
            'v_InattentionW:C(OVcate)[high]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(low)_subj',
            't(medium)_subj',
            't(high)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(OVcate)[low]_subj',
            'v_InattentionW:C(OVcate)[medium]_subj',
            'v_InattentionW:C(OVcate)[high]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (low OVcate)',
            'Non-dec. time (medium OVcate)',
            'Non-dec. time (high OVcate)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(OVcate)[low]',
            'Drift InattentionW:C(OVcate)[medium]',
            'Drift InattentionW:C(OVcate)[high]',
            ]
            
    if phase == 'ESEE':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
                'Boundary sep.',
                'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW',
            ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(phase)[ES]',
            'v_AttentionW:C(phase)[EE]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(phase)[ES]_subj', 
            'v_AttentionW:C(phase)[EE]_subj',
            'v_InattentionW_subj'
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(phase)[ES]',
            'Drift AttentionW:C(phase)[EE]', 
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]', 
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj', 
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
        elif version == 3:
            params_of_interest = [
            'a(ES)',
            'a(EE)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]'
            ]
            params_of_interest_s = [
            'a(ES)_subj',
            'a(EE)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj'            
            ]
            titles = [
            'Boundary sep. (ES)',
            'Boundary sep. (EE)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(ES)',
            't(EE)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(ES)_subj',
            't(EE)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (ES)',
            'Non-dec. time (EE)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
            
            
    if phase == 'LEESEE':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
                'Boundary sep.',
                'Non-dec. time',
                'Intercept drift rate',
                'Drift AttentionW',
                'Drift InattentionW',
                ]
        elif version == 1:
            params_of_interest = [
            'a',
            't', 
            'v_Intercept',
            'v_AttentionW:C(phase)[LE]',
            'v_AttentionW:C(phase)[ES]',
            'v_AttentionW:C(phase)[EE]',
            'v_InattentionW',
            ]
            params_of_interest_s = [
            'a_subj',
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW:C(phase)[LE]_subj', 
            'v_AttentionW:C(phase)[ES]_subj', 
            'v_AttentionW:C(phase)[EE]_subj',
            'v_InattentionW_subj'
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW:C(phase)[LE]',
            'Drift AttentionW:C(phase)[ES]',
            'Drift AttentionW:C(phase)[EE]', 
            'Drift InattentionW', 
            ]
        elif version == 2:
            params_of_interest = [
            'a',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[LE]',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]', 
            ]
            params_of_interest_s = [
            'a_subj', 
            't_subj', 
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[LE]_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj', 
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[LE]',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
        elif version == 3:
            params_of_interest = [
            'a(ES)',
            'a(ES)',
            'a(EE)',
            't',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[LE]',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]'
            ]
            params_of_interest_s = [
            'a(LE)_subj',
            'a(ES)_subj',
            'a(EE)_subj',
            't_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[LE]_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj'            
            ]
            titles = [
            'Boundary sep. (LE)',
            'Boundary sep. (ES)',
            'Boundary sep. (EE)',
            'Non-dec. time',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[LE]',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
        elif version == 4:
            params_of_interest = [
            'a',
            't(LE)',
            't(ES)',
            't(EE)',
            'v_Intercept',
            'v_AttentionW',
            'v_InattentionW:C(phase)[LE]',
            'v_InattentionW:C(phase)[ES]',
            'v_InattentionW:C(phase)[EE]',
            ]
            params_of_interest_s = [
            'a_subj',
            't(LE)_subj',
            't(ES)_subj',
            't(EE)_subj',
            'v_Intercept_subj',
            'v_AttentionW_subj',
            'v_InattentionW:C(phase)[LE]_subj',
            'v_InattentionW:C(phase)[ES]_subj',
            'v_InattentionW:C(phase)[EE]_subj',
            ]
            titles = [
            'Boundary sep.',
            'Non-dec. time (LE)',
            'Non-dec. time (ES)',
            'Non-dec. time (EE)',
            'Intercept drift rate',
            'Drift AttentionW',
            'Drift InattentionW:C(phase)[LE]',
            'Drift InattentionW:C(phase)[ES]',
            'Drift InattentionW:C(phase)[EE]',
            ]
            
    elif phase == 'ES_VAL':
        if version == 0:  
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_Intercept',
                'v_Value_diff']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_Intercept',
                'v_Value_diff']
        elif phase == 1:
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_Value_diff']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_Value_diff']
        elif version == 2:
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW']
        elif version == 3:
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_Intercept',
                'v_ES_AttentionW',
                'v_ES_InattentionW']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_Intercept',
                'v_ES_AttentionW',
                'v_ES_InattentionW']    
        elif version == 4:
            params_of_interest = [    
                'a(high)',
                'a(low)',
                'a(medium)',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW:C(OVcate)[high]',
                'v_ES_InattentionW:C(OVcate)[low]',
                'v_ES_InattentionW:C(OVcate)[medium]',
                ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a(high)',
                'a(low)',
                'a(medium)',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW:C(OVcate)[high]',
                'v_ES_InattentionW:C(OVcate)[low]',
                'v_ES_InattentionW:C(OVcate)[medium]']  
        elif version == 5:
            params_of_interest = [    
                'a(high)',
                'a(low)',
                'a(medium)',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a(high)',
                'a(low)',
                'a(medium)',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW']    
        elif version == 6:
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_Intercept',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_Intercept',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S']
            
        elif version == 7:
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S']
        elif version == 8:
            params_of_interest = [    
                'a(high)',
                'a(low)',
                'a(medium)',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a(high)',
                'a(low)',
                'a(medium)',                
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S'] 

        elif version == 9:
            params_of_interest = [    
                'a(high)',
                'a(low)',
                'a(medium)',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E:C(OVcate)[high]',
                'v_ES_InattentionW_E:C(OVcate)[low]',
                'v_ES_InattentionW_E:C(OVcate)[medium]',
                'v_ES_InattentionW_S:C(OVcate)[high]',
                'v_ES_InattentionW_S:C(OVcate)[low]',
                'v_ES_InattentionW_S:C(OVcate)[medium]']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a(high)',
                'a(low)',
                'a(medium)',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E:C(OVcate)[high]',
                'v_ES_InattentionW_E:C(OVcate)[low]',
                'v_ES_InattentionW_E:C(OVcate)[medium]',
                'v_ES_InattentionW_S:C(OVcate)[high]',
                'v_ES_InattentionW_S:C(OVcate)[low]',
                'v_ES_InattentionW_S:C(OVcate)[medium]']
    
    elif phase == "For_paper":
        if version == 0:  
            params_of_interest = [    
                'a',
                't',
                'z',
                'v'
                ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v']
        elif version == 1:
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_Value_diff'
                ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_Value_diff']
            
        elif version == 2:
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW'
                ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW']
        elif version == 3:
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_Value_diff',
                'v_DTA'
                ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_Value_diff',
                'v_DTA']
        elif version == 4:
            params_of_interest = [    
                'a',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S'
                ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S'
                ]
            
            export_posterior_draws(
                 model_name="OV_replication_For_paper_5",
                 model_dir=BASE_MODEL_DIR,
                 n_jobs=nr_models,
                 S=1000
            )

        elif version == 5:
            params_of_interest = [    
                'a(high)',
                'a(low)',
                'a(medium)',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S']
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a(high)',
                'a(low)',
                'a(medium)',
                't',
                'z',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E',
                'v_ES_InattentionW_S']
            
            # export_posterior_draws(
            #     model_name="OV_replication_For_paper_6",
            #     model_dir=BASE_MODEL_DIR,
            #     n_jobs=nr_models,
            #     S=1000
            # )
        elif version == 6:
            params_of_interest = [    
                'a',
                't',
                'v_Value_diff'
                ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'v_Value_diff' 
                ]
        # aDDM without z
        elif version == 7:
            params_of_interest = [    
                'a',
                't',
                'v_ES_AttentionW',
                'v_ES_InattentionW'
                ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',
                't',
                'v_ES_AttentionW',
                'v_ES_InattentionW'
                ]   
            export_posterior_draws(
                model_name="OV_replication_For_paper_8",
                model_dir=BASE_MODEL_DIR,
                n_jobs=nr_models,
                S=1000
            )
        # elif version == 8:
        #     params_of_interest = [    
        #         'a(high)',
        #         'a(low)',
        #         'a(medium)',
        #         't',
        #         'v_ES_AttentionW',
        #         'v_ES_InattentionW'
        #         ]
        #     params_of_interest_s = [p + "_subj" for p in params_of_interest]
        #     titles = [
        #         'a(high)',
        #         'a(low)',
        #         'a(medium)',                
        #         't',
        #         'v_ES_AttentionW',
        #         'v_ES_InattentionW'
        #         ]   
        elif version == 8:
            params_of_interest = [    
                'a',
                't',
                'v_ES_AttentionW',
                'v_ES_InattentionW_E'
                'v_ES_InattentionW_S'
                ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                'a',                
                't',
                'v_ES_AttentionW',
                'v_ES_InattentionW'
                ]   
            export_posterior_draws(
                model_name="OV_replication_For_paper_9",
                model_dir=BASE_MODEL_DIR,
                n_jobs=nr_models,
                S=1000
            )
    elif phase == "LE_RL":
        if version == 0:
            params_of_interest = [
                "a",
                "t",
                "v",
                "alpha"
            ]
            params_of_interest_s = [p + "_subj" for p in params_of_interest]
            titles = [
                "a",
                "t",
                "v",
                "alpha",]
    
    # diagnistics
    diag_dir = Path(fig_dir) / "diagnostics"
    ensure_dir(diag_dir)
    
    # Gelman-Rubin
    gr = hddm.analyze.gelman_rubin(models)
    with open(diag_dir / "gelman_rubin.txt", "w") as f:
        for param, val in gr.items():
            f.write(f"{param}: {val}\n")

    # DIC
    dic = combined_model.dic
    (diag_dir / "DIC.txt").write_text(f"DIC: {dic}\n")

    size_plot = len(combined_model.data.subj_idx.unique()) / 3.0 * 1.5
    combined_model.plot_posterior_predictive(samples=10, bins=100, figsize=(6, size_plot), save=True, path=str(diag_dir), format="pdf")
    
    matplotlib.rcParams.update({"font.size": 6})
    combined_model.plot_posteriors(save=True,
                                   path=str(diag_dir),
                                   format="pdf")
    matplotlib.rcParams.update({"font.size": 12})

    results = combined_model.gen_stats()
    results.to_csv(diag_dir / "results.csv")
    
    # get the posterior traces (helper)
    def _get_trace(model, name):
        try:
            return model.nodes_db.loc[name, "node"].trace()
        except Exception:
            return None
    
    panel_params = [
        ("v_ES_AttentionW",   "Attention weight (β_att)"),
        ("v_ES_InattentionW_E", "Inattention to E (β_IAW-E)"),
        ("v_ES_InattentionW_S", "Inattention to S (β_IAW-S)"),
    ]
    panel_traces = []
    panel_labels = []
    for p, label in panel_params:
        tr = _get_trace(combined_model, p)
        if tr is not None:
            panel_traces.append(np.asarray(tr))
            panel_labels.append(label)
    
    if panel_traces:
        # fonts
        title_size = 27
        label_size = 26
        tick_size  = 24
    
        n = len(panel_traces)
        fig, axes = plt.subplots(
            1, n, figsize=(6.0 * n, 5), constrained_layout=True
        )
        if n == 1:
            axes = [axes]
    
        for ax, tr, label in zip(axes, panel_traces, panel_labels):
            # horizontal KDE
            sns.kdeplot(x=tr, fill=True, ax=ax)
            ax.axvline(0.0, ls="--", lw=1, color="grey")
            ax.set_title(label, fontsize=title_size, pad=12)
            ax.set_xlabel("Parameter value", fontsize=label_size, labelpad=6)
            ax.set_ylabel("Density", fontsize=label_size)
            ax.tick_params(axis="both", labelsize=tick_size, width=1.2)
            for side in ["top","right"]:
                ax.spines[side].set_visible(False)
            for side in ["left","bottom"]:
                ax.spines[side].set_linewidth(1.2)
    
        fig.suptitle("Posterior densities (horizontal)", fontsize=title_size+2)
        fig.savefig(diag_dir / "kde_attention_inattention_horizontal.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        print("No traces found for attention/inattention weights; skipping panel.")
    
    group_params_to_plot = [
        "z",
        "v_ES_AttentionW",
        "v_ES_InattentionW_E",
        "v_ES_InattentionW_S",
    ]
    group_vplot_dir = diag_dir / "group_param_vertical_kdes"
    group_vplot_dir.mkdir(parents=True, exist_ok=True)
    
    vz_title = 27
    vz_label = 26
    vz_tick  = 24
    
    for param in group_params_to_plot:
        tr = _get_trace(combined_model, param)
        if tr is None:
            print(f"[Group-KDE] Skipping missing parameter: {param}")
            continue
    
        fig, ax = plt.subplots(figsize=(5, 8))
        sns.kdeplot(y=tr, fill=True, ax=ax)
        ax.set_facecolor("white")
    
        if param == "z":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=5)
    
            # Two-sided posterior probability that z != 0.5
            tr_arr = np.asarray(tr)
            p_gt = np.mean(tr_arr > 0.5)
            p_lt = np.mean(tr_arr < 0.5)
            p_two_sided = 2 * min(p_gt, p_lt)
    
            # HDI for delta = z - 0.5
            delta = tr_arr - 0.5
            hdi_lo, hdi_hi = az.hdi(delta, hdi_prob=0.95).ravel()
            hdi_text = f"95% HDI(z-0.5)=[{hdi_lo:.3f}, {hdi_hi:.3f}]"
    
            # ROPE around 0.5 
            rope = 0.02
            p_in_rope = np.mean((np.abs(delta) <= rope))
    
            ax.set_title(
                f"{param}  |  P(z!=0.5)={1-p_two_sided:.3f}\n{hdi_text} | P(|z-0.5|<={rope:.2f})={p_in_rope:.3f}",
                fontsize=vz_title, pad=12
            )
        else:
            ax.set_title(param, fontsize=vz_title, pad=12)
    
        ax.set_xlabel("Density", fontsize=vz_label, labelpad=10)
        ax.set_ylabel("Value", fontsize=vz_label)
        ax.tick_params(axis="both", labelsize=vz_tick, width=1.2)
        for side in ["top","right"]:
            ax.spines[side].set_visible(False)
        for side in ["left","bottom"]:
            ax.spines[side].set_linewidth(1.2)
    
        plt.tight_layout()
        fig.savefig(group_vplot_dir / f"{param}_vertical_kde_big.pdf", bbox_inches="tight")
        plt.close(fig)
    
    # z-diagnostics text file
    z_trace = _get_trace(combined_model, "z")
    if z_trace is not None:
        z_arr = np.asarray(z_trace)
        delta = z_arr - 0.5
        p_gt = np.mean(z_arr > 0.5)
        p_lt = np.mean(z_arr < 0.5)
        p_two_sided = 2 * min(p_gt, p_lt)
        hdi_lo, hdi_hi = az.hdi(delta, hdi_prob=0.95).ravel()
        rope = 0.02
        p_in_rope = np.mean((np.abs(delta) <= rope))
    
        with open(diag_dir / "z_diagnostics.txt", "w") as f:
            f.write("z diagnostics (group-level)\n")
            f.write("---------------------------\n")
            f.write(f"mean(z)        = {z_arr.mean():.4f}\n")
            f.write(f"sd(z)          = {z_arr.std(ddof=1):.4f}\n")
            f.write(f"P(z > 0.5)     = {p_gt:.4f}\n")
            f.write(f"P(z < 0.5)     = {p_lt:.4f}\n")
            f.write(f"Two-sided P(z != 0.5) = {1 - p_two_sided:.4f}\n")
            f.write(f"95% HDI(z-0.5) = [{hdi_lo:.4f}, {hdi_hi:.4f}]  (excludes 0? {'YES' if (hdi_lo>0 or hdi_hi<0) else 'NO'})\n")
            f.write(f"ROPE +- {rope:.2f}: P(|z-0.5| <= ROPE) = {p_in_rope:.4f}\n")
    else:
        print("No group-level z trace found; skipping z_diagnostics.")
    
    
    for f in os.listdir(diag_dir):
        if not f.endswith('.pdf') and not f.endswith('.csv'):
            continue
        safe = _sanitize_filename(f)
        if safe != f:
            os.rename(diag_dir / f, diag_dir / safe)

def plot_inatt_forest(
    fig_dir,
    model_dir,
    model_base,
    hdi_prob=0.95,
    param_E="v_ES_InattentionW_E_subj",
    param_S="v_ES_InattentionW_S_subj",
    n_chains=3
    ):
    from scipy.stats import gaussian_kde, norm
    out_dir = Path(fig_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_files = []
    for c in range(n_chains):
        candidate = Path(model_dir) / f"{model_base}_{c}.nc"
        if candidate.exists():
            nc_files.append(candidate)
    if not nc_files:
        print(f"[HDI] No .nc files found under {model_dir} for base '{model_base}_<chain>.nc'")
        return

    idatas = [az.from_netcdf(str(f)) for f in nc_files]
    idata  = az.concat(idatas, dim="chain")
    post   = idata.posterior.stack(sample=("chain", "draw"))

    # find all subject-level vars
    subj_E = [v for v in post.data_vars if v.startswith(param_E)]
    subj_S = [v for v in post.data_vars if v.startswith(param_S)]

    ids_E = {int(v.split(".")[-1]) for v in subj_E}
    ids_S = {int(v.split(".")[-1]) for v in subj_S}
    subj_ids = sorted(ids_E & ids_S)

    if not subj_ids:
        print("No overlapping subjects in nc posterior")
        return

    rows = []
    all_deltas = []
    for subj in subj_ids:
        keyE = f"{param_E}.{subj}"
        keyS = f"{param_S}.{subj}"
        E = np.abs(np.asarray(post[keyE]))
        S = np.abs(np.asarray(post[keyS]))
        delta = S - E
        all_deltas.append(delta)

        hdi_bounds = np.asarray(az.hdi(delta, hdi_prob=hdi_prob)).ravel()
        hdi_low, hdi_high = float(hdi_bounds[0]), float(hdi_bounds[-1])

        rows.append({
            "subj": subj,
            "delta_mean": float(delta.mean()),
            "hdi_low": hdi_low,
            "hdi_high": hdi_high,
            "credible": int((hdi_low > 0) or (hdi_high < 0))
        })

    hdi_df = pd.DataFrame(rows).sort_values("subj")
    hdi_csv = out_dir / "inatt_asymmetry_HDI.csv"
    hdi_df.to_csv(hdi_csv, index=False)
    print(f"[HDI] Saved: {hdi_csv}")

    # group-level Bayes factor
    group_delta = np.concatenate(all_deltas)
    kde = gaussian_kde(group_delta)
    post_at_0 = kde.evaluate([0])[0]
    prior_at_0 = norm.pdf(0, loc=0, scale=1)
    BF_01 = post_at_0 / prior_at_0
    BF_10 = 1 / BF_01
    bf_file = out_dir / "inatt_asymmetry_BayesFactor.txt"
    with open(bf_file, "w") as f:
        f.write(f"BF_01 (H0/H1): {BF_01:.3f}\n")
        f.write(f"BF_10 (H1/H0): {BF_10:.3f}\n")

    print(f"[BF] Saved Bayes factor results to {bf_file}")
    print(f"  BF_01 = {BF_01:.3f}, BF_10 = {BF_10:.3f}")

    # forest plot
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(hdi_df)))
    ax.set_facecolor("white")
    ax.grid(False)

    ypos = np.arange(len(hdi_df))
    for i, row in enumerate(hdi_df.itertuples(index=False)):
        ax.plot([row.hdi_low, row.hdi_high], [ypos[i], ypos[i]], "k-", lw=1)
        ax.plot(row.delta_mean, ypos[i], "o", color="purple")

    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(hdi_df["subj"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ inattentional weight (|S| − |E|), {int(hdi_prob*100)}% HDI")
    ax.set_title(f"Subject-level inattentional asymmetry (HDI)\nGroup BF_10={BF_10:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "forest_inatt_asymmetry_HDI.pdf", bbox_inches="tight")
    plt.close(fig)

# this is being called when the RL model is analyzed 
def analyze_rl(infdatas, fig_dir, version):
    fig_dir = Path(fig_dir)
    diag_dir = fig_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # concatenate chains
    idata = az.concat(infdatas, dim="chain")
    print(idata)
    print(idata.posterior)
    print("Data variables in posterior:", list(idata.posterior.data_vars))
    rhat = az.rhat(idata)
    with open(diag_dir / "gelman_rubin.txt", "w") as f:
        for var in rhat.data_vars:
            val = float(rhat[var].values)  # extract scalar
            f.write(f"{var}: {val:.3f}\n")

    # optional other scores
    # waic_res = az.waic(idata)
    # dic_df = pd.DataFrame({
    #     "metric": ["DIC", "DIC_se"],
    #     "value":  [waic_res.waic, waic_res.waic_se]
    # })
    # dic_df.to_csv(diag_dir/"dic.csv", index=False)
    # az.plot_ppc(idata)  
    # plt.savefig(diag_dir / "posterior_predictive.pdf")
    # plt.close()

    # stats table
    summary = az.summary(idata)
    summary.to_csv(diag_dir / "results.csv")

    if "alpha" not in idata.posterior.data_vars:
        print("[alpha-transform] No 'alpha' in posterior; skipping transformed CSV.")
        return

    # Transform group-level alpha
    alpha_draws = idata.posterior["alpha"].values.reshape(-1)  # (chains*draws,)
    alpha_prob  = _inv_logit(alpha_draws)
    alpha_summ  = _summ_from_samples(alpha_prob)

    # Transform subject-level alphas (if present)
    subj_vars = [v for v in idata.posterior.data_vars if v.startswith("alpha_subj.")]
    subj_summ_rows = {}
    subj_prob_matrix = []  # will become shape (n_draws, n_subj) for SD on prob-scale

    if subj_vars:
        # build matrix: columns = subjects, rows = draws (all chains collapsed)
        for v in sorted(subj_vars, key=lambda x: int(x.split("alpha_subj.")[-1])):
            arr = idata.posterior[v].values.reshape(-1)
            arr_prob = _inv_logit(arr)
            subj_prob_matrix.append(arr_prob)
            subj_summ_rows[v] = _summ_from_samples(arr_prob)

        subj_prob_matrix = np.vstack(subj_prob_matrix).T  # (draws, subj)
        sd_draws = np.std(subj_prob_matrix, axis=1, ddof=1)
        alpha_std_summ = _summ_from_samples(sd_draws)
    else:
        alpha_std_summ = None

    # transformed copy of the ArviZ summary and replace alpha rows 
    summary_t = summary.copy()

    # Replace group alpha row (if present)
    if "alpha" in summary_t.index:
        for k, v in alpha_summ.items():
            summary_t.loc["alpha", k] = v

    # Replace alpha_std row (if present & we could compute it) #
    if ("alpha_std" in summary_t.index) and (alpha_std_summ is not None):
        for k, v in alpha_std_summ.items():
            summary_t.loc["alpha_std", k] = v

    # Replace subject rows
    for v, stats in subj_summ_rows.items():
        if v in summary_t.index:
            for k, val in stats.items():
                summary_t.loc[v, k] = val

    out_csv = diag_dir / "results_alpha_transformed.csv"
    summary_t.to_csv(out_csv)

    if subj_vars:
        means = []
        for v in sorted(subj_vars, key=lambda x: int(x.split("alpha_subj.")[-1])):
            arr = idata.posterior[v].values.reshape(-1)
            arr_prob = _inv_logit(arr)
            means.append({"param": v, "mean_prob": float(np.mean(arr_prob))})
        pd.DataFrame(means).to_csv(diag_dir / "params_of_interest_s_alpha_transformed.csv", index=False)

    print(f"[alpha-transform] Wrote:\n  - {out_csv}")
    if subj_vars:
        print(f"  - {diag_dir / 'params_of_interest_s_alpha_transformed.csv'}")



    #Posterior‐trace + KDE plot
    var_names = ["alpha"]
    titles = ["alpha"]
    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()
    plt.savefig(diag_dir / "trace_plots.pdf")
    plt.close()
    # Posterior KDEs
    matplotlib.rcParams.update({"font.size": 6})
    
    # figure with one column per variable
    fig, axes = plt.subplots(1, len(var_names), figsize=(len(var_names) * 2, 4))
    axes_flat = np.atleast_1d(axes).flatten()
    for i, p in enumerate(var_names):
        ax = axes_flat[i]
        arr = idata.posterior[p].values.reshape(-1)
        if p == "alpha":
            arr = np.exp(arr) / (1 + np.exp(arr))
        sns.kdeplot(y=arr, fill=True, ax=ax)
        ax.set_title(p)
        ax.set_xlim(0, 15)      
        ax.set_ylim(0, 0.5)    
        ax.set_ylabel("Density")
        ax.set_xlabel("Value")

    plt.tight_layout()
    plt.savefig(diag_dir / "posteriors.pdf", bbox_inches="tight")
    plt.close(fig)
        # Per-subject parameter CSV
        # Per-subject parameter CSV
    subj_vars = [v for v in idata.posterior.data_vars if v.startswith("alpha_subj.")]
    if subj_vars:
        subj_means = {}
        for var in subj_vars:
            subj = int(var.split("alpha_subj.")[-1])
            arr  = idata.posterior[var].values  # (chain, draw)
            subj_means[subj] = arr.reshape(-1).mean()
        df = pd.DataFrame.from_dict(
            subj_means, orient="index", columns=["alpha_subj"]
        )
        df.index.name = "subj_idx"
        df.reset_index(inplace=True)
        df.to_csv(diag_dir/"params_of_interest_s.csv", index=False)
    else:
        print("ERROR")
     
model_dir = BASE_MODEL_DIR
ensure_dir(model_dir)



# # this calls ddm functions depending on whether we run or load models
# for version_idx, model_name in enumerate(model_versions[phase], start=0):
#     full_model_name = model_base_name + model_name
#     fig_dir        = FIG_DIR_ROOT / full_model_name
#     ensure_dir(fig_dir)

#     print(f"\n=== Phase={phase}, Version={version_idx} ({model_name}) ===")
#     if run:
#         # for ES/EE you call drift_diffusion_hddm
#         if phase in ('ES', 'EE'):
#             print(f'Running DDM {full_model_name}')
#             models = drift_diffusion_hddm(
#                 data=data,
#                 samples=nr_samples,
#                 n_jobs=nr_models,
#                 run=run,
#                 parallel=parallel,
#                 model_name=full_model_name,
#                 model_dir=model_dir,
#                 version=version_idx,
#                 phase=phase,
#                 accuracy_coding=True
#             )
#         # …and similarly for ESEE/LEESEE or RL…
#         elif phase in ('ESEE','LEESEE'):
#             print(f'Running combined {full_model_name}')
#             models = drift_diffusion_hddm(
#                 data=data,
#                 samples=nr_samples,
#                 n_jobs=nr_models,
#                 run=run,
#                 parallel=parallel,
#                 model_name=full_model_name,
#                 model_dir=model_dir,
#                 version=version_idx,
#                 phase=phase,
#                 accuracy_coding=True
#             )
#         else:
#             print(f'Running HDDMRL {full_model_name}')
#             models = drift_diffusion_hddmRL(
#                 data=data,
#                 samples=nr_samples,
#                 n_jobs=nr_models,
#                 run=run,
#                 parallel=parallel,
#                 model_name=full_model_name,
#                 model_dir=model_dir,
#                 version=version_idx,
#                 phase=phase,
#             )

#     else:
#         # loading + analysis
#         if phase in ('ES', 'EE'):
#             print(f'Loading DDM {full_model_name}')
#             models = drift_diffusion_hddm(
#                 data=data,
#                 samples=nr_samples,
#                 n_jobs=nr_models,
#                 run=run,
#                 parallel=parallel,
#                 model_name=full_model_name,
#                 model_dir=model_dir,
#                 version=version_idx,
#                 phase=phase,
#                 accuracy_coding=True
#             )
#             analyze_model(models, fig_dir, nr_models, version_idx, phase)

#         elif phase in ('ESEE','LEESEE'):
#             print(f'Loading combined {full_model_name}')
#             models = drift_diffusion_hddm(
#                 data=data,
#                 samples=nr_samples,
#                 n_jobs=nr_models,
#                 run=run,
#                 parallel=parallel,
#                 model_name=full_model_name,
#                 model_dir=model_dir,
#                 version=version_idx,
#                 phase=phase,
#                 accuracy_coding=True
#             )
#             analyze_model(models, fig_dir, nr_models, version_idx, phase)

#         else:
#             print(f'Loading HDDMRL {full_model_name}')
#             models = drift_diffusion_hddmRL(
#                 data=data,
#                 samples=nr_samples,
#                 n_jobs=nr_models,
#                 run=run,
#                 parallel=parallel,
#                 model_name=full_model_name,
#                 model_dir=model_dir,
#                 version=version_idx,
#                 phase=phase,
#             )
#             analyze_model(models, fig_dir, nr_models, version_idx, phase)





#ingle model running version - use for manual
# this calls our ddm functions depending on whether we run or load models
if run:
    if phase == 'EE' or phase == 'ES' or phase == 'ES_VAL' or phase == 'For_paper':
        print(f'Running DDM... {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase, 
            accuracy_coding=True
        )
    
    elif phase == 'ESEE': 
        print(f'Running Combined Model (ES+EE)... {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,  
            accuracy_coding=True
        )
    elif phase == 'LEESEE': 
        print(f'Running Combined Model (LE+ES+EE)... {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,  
            accuracy_coding=True
        )
    elif phase == 'ES_ZBIAS':
        print(f'Running ES_ZBIAS DDM… {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,
            accuracy_coding=True
        )
    else:
        print(f'Running HDDMRL... {model_base_name + model_name}')
        models = drift_diffusion_hddmRL(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,  
        )
else:
    if phase == 'EE' or phase == 'ES' or phase == 'ES_VAL' or phase == 'For_paper':
        print(f'loading DDM... {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,  
            accuracy_coding=True
        )
        analyze_model(models, fig_dir, nr_models, version, phase)
        diag_dir = Path(fig_dir) / "diagnostics"
        plot_inatt_forest(
            fig_dir=fig_dir,
            model_dir=model_dir,
            model_base=model_base_name + model_name,
            param_E="v_ES_InattentionW_E_subj",
            param_S="v_ES_InattentionW_S_subj"
        )
        


    elif phase == 'ESEE':  
        print(f'loading Combined DDM Model (ES+EE)... {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,  
            accuracy_coding=True
        )
        analyze_model(models, fig_dir, nr_models, version, phase)
        
    elif phase == 'LEESEE':  
        print(f'loading Combined DDM Model (LE+ES+EE)... {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,  
            accuracy_coding=True
        )
    elif phase == 'ES_ZBIAS':  
        print(f'loading DDM Model (ES_ZBIAS)... {model_base_name + model_name}')
        models = drift_diffusion_hddm(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,  
            accuracy_coding=True
        )
        analyze_model(models, fig_dir, nr_models, version, phase)
    
    elif phase == 'LE_RL':
        print(f'Loading RL chains for {full_model_name}…')
        infdatas = drift_diffusion_hddmRL(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=full_model_name,
            model_dir=model_dir,
            version=version,
            phase=phase,
            )
        analyze_rl(infdatas, fig_dir, version)
    else:
        print(f'Running HDDMRL... {model_base_name + model_name}')
        models = drift_diffusion_hddmRL(
            data=data,
            samples=nr_samples,
            n_jobs=nr_models,
            run=run,
            parallel=parallel,
            model_name=model_base_name + model_name,
            model_dir=model_dir,
            version=version,
            phase=phase, 
        )
        analyze_model(models, fig_dir, nr_models, version, phase)
    




# Very old - don't use 

# if run:
#     if phase == 'EE' or phase == 'ES':
#         print('Running DDM...{}'.format(model_base_name + model_name))
#         models = drift_diffusion_hddm(
#             data=data,
#             samples=nr_samples,
#             n_jobs=nr_models,
#             run=run,
#             parallel=parallel,
#             model_name=model_base_name + model_name,
#             model_dir=model_dir,
#             version=version,
#             phase = phase,
#             accuracy_coding=True
#             )
#     else:
#         print('Running HDDMRL...{}'.format(model_base_name + model_name))
#         models = drift_diffusion_hddmRL(
#             data=data,
#             samples=nr_samples,
#             n_jobs=nr_models,
#             run=run,
#             parallel=parallel,
#             model_name=model_base_name + model_name,
#             model_dir=model_dir,
#             version=version,
#             phase = phase,
#             )
# else:
#     if phase == 'EE' or phase == 'ES':
#         print('loading DDM... {}'.format(model_base_name + model_name))
#         models = drift_diffusion_hddm(
#             data=data,
#             samples=nr_samples,
#             n_jobs=nr_models,
#             run=run,
#             parallel=parallel,
#             model_name=model_base_name + model_name,
#             model_dir=model_dir,
#             version=version,
#             phase = phase,
#             accuracy_coding=True
#             )
#         analyze_model(models, fig_dir, nr_models, version, phase)

#     else:
#         print('loading HDDMRL ... {}'.format(model_base_name + model_name))
#         models = drift_diffusion_hddmRL(
#             data=data,
#             samples=nr_samples,
#             n_jobs=nr_models,
#             run=run,
#             parallel=parallel,
#             model_name=model_base_name + model_name,
#             model_dir=model_dir,
#             version=version,
#             phase = phase,
#             )
        
#         analyze_model(models, fig_dir, nr_models, version, phase)
#       



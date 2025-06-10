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
from IPython import embed as shell
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
#------------------------------------------------------------------------------------------------------------------

# addm regression formula: this is how v is influenced by the value, behavioural data and gaze
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2,low ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ϵ


# params:
version = 0     # set which version you want to run
run = False       # if True, the the models run, if False the models load

phase = ['LE']  #['ES', 'EE']  

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

nr_models = 5 
nr_samples = 6000
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
    'LEESEE': ['LEESEE_1', 'LEESEE_2', 'LEESEE_3', 'LEESEE_4', 'LEESEE_5']
}

if phase not in model_versions:
    raise ValueError(f"Invalid phase selection '{phase}'. Choose from: {list(model_versions.keys())}")

model_name = model_versions[phase][version]

data = pd.read_csv((PROJECT_DIR / "data_sets" / "data_sets_OV" / "OVParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv").as_posix(), sep=",")

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
    
# RT distributions plot
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

# make data genuinely numeric (so that nans are also detected)
def sanitize_infdata(infdata):
    """Convert pd.NA values to np.nan in all groups of the InferenceData object."""
    for group in infdata._groups_all:
        if hasattr(infdata, group):
            dataset = getattr(infdata, group)
            for var in dataset.data_vars:
                values = dataset[var].values
                if isinstance(values, np.ndarray) and values.dtype == "object":
                    # pd.NA --> np.nan
                    mask = pd.isna(values)
                    if mask.any():
                        print(f"{var} in group '{group}' (contains pd.NA)")
                        values[mask] = np.nan
                        dataset[var].values = values
    return infdata


def _sanitize_filename(fname):
    # replace any of : ( ) [ ] , with underscore
    safe = re.sub(r'[:\(\)\[\],]', '_', fname)
    # collapse runs of underscores to a single underscore
    safe = re.sub(r'_+', '_', safe)
    return safe


fig_dir = FIG_DIR_ROOT / f"{model_base_name}{model_name}"
ensure_dir(fig_dir / "diagnostics")


# subjects:
subjects = np.unique(data.subj_idx)
nr_subjects = subjects.shape[0]
print(nr_subjects)
print(subjects)


#########################################################################################################
# drift diffusion models
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# function that runs the different versions of DDM regressions

def run_model(trace_id, data, model_dir, model_name, version, samples=2100, accuracy_coding=True):  #shoukd be 5000 samples but can be changed depending on computing capacities
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
                   burn=100,
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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main function for running/loading models

###############################################################################################################

import dill as pickle  # to create the pkl object

def drift_diffusion_hddm(data, 
                         samples=6000,
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
                infdata = sanitize_infdata(infdata) 
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
import dill as pickle

def drift_diffusion_hddmRL(data, 
                         samples=6000, #5000
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
                # save in HDDM format
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
        models = [hddm.load(os.path.join(model_dir, f"{model_name}_{i}.hddm")) for i in range(n_jobs)]
        return models

######################################################################################################################################
# analyzing the models   
        
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
    
    # shrink font for the next set of plots
    matplotlib.rcParams.update({"font.size": 6})
    combined_model.plot_posteriors(save=True,
                                   path=str(diag_dir),
                                   format="pdf")
    matplotlib.rcParams.update({"font.size": 12})

    # stats table
    results = combined_model.gen_stats()
    results.to_csv(diag_dir / "results.csv")
    
    # Posterior‐trace KDEs
    traces = [combined_model.nodes_db.node[p].trace() for p in params_of_interest]
    # optional alpha‐transform if RL is used for instance
    if "alpha" in params_of_interest:
        idx = params_of_interest.index("alpha")
        traces[idx] = np.exp(traces[idx]) / (1 + np.exp(traces[idx]))
    
    stats = [min(np.mean(t>0), np.mean(t<0)) for t in traces]
    n_cols = 5
    n_rows = int(np.ceil(len(traces) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*4))
    axes = axes.flatten()
    
    for i, (trace, title) in enumerate(zip(traces, titles)):
        sns.kdeplot(trace, vertical=True, shade=True, color='purple', ax=axes[i])
        axes[i].set_title(f"{title}\np={stats[i]:.3f}", fontsize=6)
        axes[i].set_xlim(left=0)
        if i % n_cols == 0:
            axes[i].set_ylabel("Parameter estimate (a.u.)")
        if i >= len(traces) - n_cols:
            axes[i].set_xlabel("Posterior probability")
        for side in ["top","bottom","left","right"]:
            axes[i].spines[side].set_linewidth(0.5)
            axes[i].tick_params(width=0.5, labelsize=6)   
            
    # drop extra axes
    for ax in axes[len(traces):]:
        fig.delaxes(ax)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    fig.savefig(diag_dir / "posteriors.pdf", bbox_inches="tight")
    plt.close(fig) 
    
    
    # save per‐subject parameters
    parameters = []
    for p in params_of_interest_s:
        param_values = []
        for s in np.unique(combined_model.data.subj_idx):
            param_name = f"{p}.{s}"
            try:
                val = results.loc[results.index == param_name, 'mean'].values
                if len(val):
                    v = val[0]
                    if 'alpha' in p:
                        # inverse‐logit transform for any alpha‐params
                        v = np.exp(v) / (1 + np.exp(v))
                    param_values.append(v)
            except KeyError:
                print(f"Param {param_name} missing. Skipping…")
        parameters.append(param_values)

    # turn into DataFrame, transpose so each subj is a row, then save
    param_df = pd.DataFrame(parameters).T
    param_df.columns = params_of_interest_s
    param_df.to_csv(diag_dir / "params_of_interest_s.csv", index=False)
    
    
    
    for f in os.listdir(diag_dir):
        if not f.endswith('.pdf') and not f.endswith('.csv'):
            continue
        safe = _sanitize_filename(f)
        if safe != f:
            os.rename(diag_dir / f, diag_dir / safe)

    
model_dir = BASE_MODEL_DIR
ensure_dir(model_dir)


# this calls our ddm functions depending on whether we run or load models
if run:
    if phase == 'EE' or phase == 'ES':
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
            phase=phase,  # Use updated phase key
            accuracy_coding=True
        )
    
    elif phase == 'ESEE':  # Ensure this condition runs only for the combined model
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
    if phase == 'EE' or phase == 'ES':
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
        analyze_model(models, fig_dir, nr_models, version, phase)
        
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
        



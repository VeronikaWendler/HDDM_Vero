# Veronika Wendler
# 22.01.25
# code for the attentional drift diffusion model - originally, I used this in summer 2024 in Quebec and was inspired by Jan WIllem De Gee's code somewhere on GitHub - but this version is pretty much my creation
 
# import libraries
import pandas as pd
import numpy as np
import hddm
import os, sys, pickle, time
import datetime
import math
import scipy as sp
import matplotlib
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
# warning settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# Stats 
from statsmodels.distributions.empirical_distribution import ECDF
# HDDM
from hddm.simulators.hddm_dataset_generators import simulator_h_c

# Import my own libraries - I don't really use it anymore 
current_directory = os.getcwd()
#from helper_functions_2 import prepare_data
#import compact_models

#------------------------------------------------------------------------------------------------------------------
# Structure of saving:

# /home/jovyan/OfficialTutorials/THESIS_HDDM
#     ├── data_sets_garcia/
#     │   └── GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv
#     ├── model_dir_garcia_CCT/
#     │   ├── EE_0 ...n
#     │   ├── ES_0 ..n
#     │   ├── ESEE_0 ...n
#     │   ├── LEESEE_0 ...n
#     ├── figures_garcia_CCT/
#     │   └── garcia_replication_EE_n/
#     │       ├── diagnostics/
#     │       │   ├── gelman_rubic.txt
#     │       │   ├── DIC.txt
#     │       │   ├── results.csv
#     │       │   └── posteriors.pdf
#     │   └── garcia_replication_ES_n/
#     │   └── garcia_replication_ESEE_n/
#     │   └── garcia_replication_LEESEE_n/
#     |   with_sub99_noRTcriteria - contains HDDM with no RT exclusion criteria + participant 99
#     ├── other_script.py
#------------------------------------------------------------------------------------------------------------------

# addm regression formula
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2 ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ϵ
# where ß0 = intercept,
# ß1 = AttentionW,
# ß2 = InattentionW,
# ϵ = noise
# PropDwell_opt = proportion of dwell time on the option with higher expected value
# PropDwell_sub = proportion of dwell time on the option with lower expected value
# V_opt​ = value if the better option
# V_sub = value of the worse option


# hard-coded 
nr_models       = 5         # number of MCMC chains
nr_samples      = 6000      # samples per chain
parallel        = True     # parallel
model_base_name = "garcia_replication_"
model_versions  = {
    "LE":     ["LE_1","LE_2","LE_3","LE_4","LE_5","LE_6","LE_7"],
    "ES":     ["ES_1","ES_2","ES_3","ES_4","ES_5"],
    "EE":     ["EE_1","EE_2","EE_3","EE_4","EE_5"],
    "ESEE":   ["ESEE_1","ESEE_2","ESEE_3","ESEE_4","ESEE_5"],
    "LEESEE": ["LEESEE_1","LEESEE_2","LEESEE_3","LEESEE_4","LEESEE_5"],
}

# ------------------------------------------------------------------
# BATCH-RUN CONTROL
PHASE_RUN_ORDER = ["LE"]               # order
SKIP_PHASES     = {"ES", "EE", "ESEE", "LEESEE"}                 # ignored this phase
RUN_ALL_MODELS  = True                    # False = just load existing fits

# selectivity
start_phase = "LE"
start_version = 3
started = False

# dir
BASE_MODEL_DIR = '/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia'

# ------------------------------------------------------------------



# reporting function
def quick_report(data, phase, version, model_name, phase_key):
    """Lightweight console & plotting diagnostics per (phase,version)."""
    print(f"\n Phase = {phase}   Version = {version}")
    print(f"Model name          : {model_name}")
    print(f"Selected phase_key  : {phase_key}")
    print(f"N trials            : {len(data):,}")
    print(f"Participants        : {sorted(data['subj_idx'].unique())}")
    print("OVcate counts:\n", data['OVcate'].value_counts(dropna=False))

    fig, ax = plt.subplots(figsize=(6,4))
    for _, d in data.groupby('subj_idx'):
        d['rt'].hist(bins=20, histtype='step', ax=ax, alpha=.4)
    ax.set(
        title=f"RT distribution – {phase} v{version}",
        xlabel="RT (s)",
        ylabel="count"
    )
    plt.show()


# ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def sanitize_infdata(infdata):
    """Convert pd.NA values to np.nan in all groups of the InferenceData object (important for if you have columns which we don't use, for example, particular RL cols)."""
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


###################################################################################################################
# drift diffusion models
#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
# function that runs/defines the different versions/models of DDM regressions for the selected phase or phases

def run_model(trace_id, data, model_dir, model_name, version, phase, samples=6000, accuracy_coding=True): 
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
        if version == 0:    # m1 # this is the 0 model with fully fixed parameters across OV levels
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # m2 fixated option weights varies by OV level 
            v_reg = {'model': 'v ~ 1 + AttentionW:C(OVcate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  #m3  non-fixated option weights varies by OV level
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: # m4 non-fixated options weights varies by OV level and boundary separation
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}      
        elif version == 4:  # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate'} 
        else:
            raise ValueError(f"check version {version} ??")
     
       
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
                   burn=1000,      #is variable
                   dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                   db='pickle',
                   return_infdata=True, loglike=True, ppc=True)

        return m, infdata
    
    elif phase == 'EE':
        accuracy_coding = True
        if version == 0:     # m1 # this is the 0 model with fully fixed parameters across OV levels
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # m2 attentional weight parameter (fixated) option weights varies by OV level 
            v_reg = {'model': 'v ~ 1 + AttentionW:C(OVcate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  #m3  non-fixated option weights varies by OV level
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: # m4 non-fixated options weights varies by OV level and boundary separation
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}      
        elif version == 4: # m5 non-fixated options weights varies by OV level and non-dec. time
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

    elif phase == 'ESEE':  # combined model for ES + EE (furhter confimation that theta varies by phase (not just OV))
        accuracy_coding = True
        if version == 0:  # baseline model with fixed parameters across phases (ES, EE)
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # drift rate varies by phase (ES vs. EE)
            v_reg = {'model': 'v ~ 1 + AttentionW:C(phase) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  # non-fix option weights vary by phase
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 3:  # decision threshold (a) varies by phase + InattentionW:C(phase)
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'phase'}
        elif version == 4:  # non-decision time varies by phase + InattentionW:C(phase)
            v_reg = {'model': 'v ~ 1 + AttentionW + InattentionW:C(phase)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'t': 'phase'}
        else:
            raise ValueError(f"check version {version} ??")
     
        
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


###############################################################################################################    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main function for running/loading models


import dill as pickle  # to create the pkl object

def drift_diffusion_hddm(data, 
                         samples=6000,
                         n_jobs=5,
                         run=True,
                         parallel=True,
                         model_name='model',
                         model_dir='.', 
                         accuracy_coding=True,
                         version=None,
                         phase=None):

    if run:
        if parallel:
            start_time = time.time()
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_model)(
                    trace_id=trace_id,
                    data=data,
                    model_dir=model_dir,
                    model_name=model_name,
                    version=version,
                    phase=phase,
                    samples=samples,
                    accuracy_coding=accuracy_coding
                    )
                for trace_id in range(n_jobs)
            )
            print("Time elapsed:", time.time() - start_time, "s")
            
           
            for i in range(n_jobs):
                model, infdata = results[i]
                model.save(os.path.join(model_dir, f"{model_name}_{i}.hddm"))
                with open(os.path.join(model_dir, f"{model_name}_{i}.pkl"), "wb") as f:
                    pickle.dump(model, f)
                infdata = sanitize_infdata(infdata)  # clean before saving
                az.to_netcdf(infdata, os.path.join(model_dir, f"{model_name}_{i}.nc"))


        else: 
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
                         samples=6000, 
                         n_jobs=5,
                         run=True,
                         parallel=True,
                         model_name='model',
                         model_dir='.', 
                         version=None,       
                         accuracy_coding=True):

    if run:
        if parallel:
            start_time = time.time()
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_model)(
                    trace_id=trace_id,
                    data=data,
                    model_dir=model_dir,
                    model_name=model_name,
                    version=version,
                    phase=phase,
                    samples=samples,
                    accuracy_coding=accuracy_coding
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
        models = [hddm.load(os.path.join(model_dir, f"{model_name}_{i}.hddm")) for i in range(n_jobs)]
        return models

#########################################################################################################################################################
#---------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------
# Analyzing the models

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

    # Try combining models
    try:
        combined_model = kabuki.utils.concat_models(models)
        print("Models combined successfully.")
    except Exception as e:
        print(f"Error combining models: {e}")
        return
    
    # names parameters 
    
    if phase == 'LE':
        if version == 0:
            params_of_interest = [
                'a',
                't',
                'alpha',
                'v_Intercept',
                'v_AttentionW',
                'v_InattentionW',
                ]
            params_of_interest_s = [
                'a_subj', 
                't_subj', 
                'alpha_subj',
                'v_Intercept_subj',
                'v_AttentionW_subj',
                'v_InattentionW_subj', 
                ]
            titles = [
                'Boundary sep.',
                'Non-dec. time',
                'Learning rate'
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
            'v_AttentionW:C(OVcate)[low]',
            'v_AttentionW:C(OVcate)[medium]',
            'v_AttentionW:C(OVcate)[high]',
            'v_InattentionW'
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
            'Drift InattentionW:C(OVcate)[high]'
            
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
            
    # Gelman-Rubin diagnostic
    gr = hddm.analyze.gelman_rubin(models)
    ensure_dir(os.path.join(fig_dir, 'diagnostics'))
    with open(os.path.join(fig_dir, 'diagnostics', 'gelman_rubin.txt'), 'w') as text_file:
        for p in gr.items():
            text_file.write(f"{p[0]}: {p[1]}\n")

    # DIC
    dic = combined_model.dic
    with open(os.path.join(fig_dir, 'diagnostics', 'DIC.txt'), 'w') as text_file:
        text_file.write(f"DIC: {dic}\n")
        
    # Plots
    size_plot = len(combined_model.data.subj_idx.unique()) / 3.0 * 1.5
    combined_model.plot_posterior_predictive(samples=10, bins=100, figsize=(6, size_plot), save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    matplotlib.rcParams.update({'font.size': 6}) 
    combined_model.plot_posteriors(save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    
    matplotlib.rcParams.update({'font.size': 12})

    # results
    results = combined_model.gen_stats()
    results.to_csv(os.path.join(fig_dir, 'diagnostics', 'results.csv'))
    
    # Posterior analysis and fixed starting point as in J.W. de Gee code
    traces = [combined_model.nodes_db.node[p].trace() for p in params_of_interest]
    #traces[0] = 1 / (1 + np.exp(-(traces[0])))
    
    # If 'alpha' is among the parameters, transform its trace
    if 'alpha' in params_of_interest:
        alpha_idx = params_of_interest.index('alpha')
        # Transform using the inverse logit: np.exp(alpha)/(1+np.exp(alpha))
        traces[alpha_idx] = np.exp(traces[alpha_idx]) / (1 + np.exp(traces[alpha_idx]))
    
    #Posterior Statistics for parameter traces, significance testing
    stats = []
    for trace in traces:
        stat = min(np.mean(trace > 0), np.mean(trace < 0))
        stats.append(min(stat, 1 - stat))
    stats = np.array(stats)
    
    n_cols = 5
    n_rows = int(np.ceil(len(params_of_interest) / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 3, n_rows * 4))
    axes = axes.flatten()

    for ax_nr, (trace, title) in enumerate(zip(traces, titles)):
        sns.kdeplot(trace, vertical=True, shade=True, color='purple', ax=axes[ax_nr])
        if ax_nr % n_cols == 0:
            axes[ax_nr].set_ylabel('Parameter estimate (a.u.)')
        if ax_nr >= len(params_of_interest) - n_cols:
            axes[ax_nr].set_xlabel('Posterior probability')
        axes[ax_nr].set_title(f'{title}\np={round(stats[ax_nr], 3)}', fontsize=6)  
        axes[ax_nr].set_xlim(xmin=0)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes[ax_nr].spines[axis].set_linewidth(0.5)
            axes[ax_nr].tick_params(width=0.5, labelsize=6)  

    for ax in axes[len(params_of_interest):]:
        fig.delaxes(ax)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'posteriors.pdf'), bbox_inches='tight') 

        
    # parameters = []
    # for p in params_of_interest_s:
    #     param_values = []
    #     for s in np.unique(combined_model.data.subj_idx):
    #         param_name = f"{p}.{s}"
    #         try:
    #             param_value = results.loc[results.index == param_name, 'mean'].values
    #             if len(param_value) > 0:
    #                 param_values.append(param_value[0])
    #         except KeyError:
    #             print(f"Param {param_name} missing. Skipping...")
    #             continue
    #     parameters.append(param_values)
    
    parameters = []
    for p in params_of_interest_s:
        param_values = []
        for s in np.unique(combined_model.data.subj_idx):
            param_name = f"{p}.{s}"
            try:
                param_value = results.loc[results.index == param_name, 'mean'].values
                if len(param_value) > 0:
                    value = param_value[0]
                    # If this parameter is alpha (or subject-level alpha), transform it
                    if 'alpha' in p:
                        value = np.exp(value) / (1 + np.exp(value))
                    param_values.append(value)
            except KeyError:
                print(f"Param {param_name} missing. Skipping...")
                continue
        parameters.append(param_values)

    parameters = pd.DataFrame(parameters).T
    parameters.columns = params_of_interest_s
    parameters.to_csv(os.path.join(fig_dir, 'diagnostics', 'params_of_interest_s.csv'))

# directories
model_dir = 'models_dir_garcia/'
ensure_dir(model_dir)



# ==================================================================
# BATCH DRIVER – runs every (phase, version) - pairing
# ==================================================================

if __name__ == "__main__":

    # csv
    data_full = pd.read_csv(
        os.path.join(current_directory,
                     "data_sets/data_sets_Garcia",
                     "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv"),
        sep=","
    )

    # loop over phases and versions
    for phase in PHASE_RUN_ORDER:
        if phase in SKIP_PHASES:
            continue                    
        
        phase_key = phase


        for version, model_name in enumerate(model_versions[phase]):
            
            # ------------- Start Control -----------------------------
            if not started:
                if phase == start_phase and version >= start_version:
                    started = True
                elif PHASE_RUN_ORDER.index(phase) > PHASE_RUN_ORDER.index(start_phase):
                    started = True
                else:
                    continue #skip

            # ----------------------------------------------------------
            
            full_model_name = model_base_name + model_name
            print(f"\n===  PHASE {phase} : {model_name}  ===")

            # --------------- filter data for this phase ---------------
            if phase == "ESEE":
                data = data_full[data_full["phase"].isin(["ES", "EE"])].copy()
            elif phase == "LEESEE":
                data = data_full[data_full["phase"].isin(["LE", "ES", "EE"])].copy()
            else:
                data = data_full[data_full["phase"] == phase].copy()

            # ---------------- preprocessing ---------------

            data["phase"]       = data["phase"].astype("category")
            data["rt"]          = pd.to_numeric(data["rtime"], errors="coerce")
            data                = data[data["rt"] > 0.250]
            data["response"]    = pd.to_numeric(data["corr"], errors="coerce")
            data["OVcate"]      = data["OVcate_2"].astype("category")
            data["Abscate"]     = data["Abscate_2"].astype("category")
            data["cond"]     = data["cond"].fillna(-1)
            data["cond"]     = data["cond"].astype("int")
            data["AttentionW"]  = pd.to_numeric(data["AttentionW"],  errors="coerce")
            data["InattentionW"]= pd.to_numeric(data["InattentionW"],errors="coerce")
            data["subj_idx"]    = data["sub_id"]
            data = data[~data["subj_idx"].isin({1,4,5,6,14,99})]
            data = data.dropna(subset=["rt","response","OVcate","Abscate",
                                       "subj_idx","AttentionW","InattentionW","cond"])
            
            # ------------------------------------------------------------
            # gives you a quick report at the start
            quick_report(data, phase, version, model_name, phase_key)

            fig_dir = os.path.join("figures_dir_garcia", full_model_name)
            ensure_dir(os.path.join(fig_dir, "diagnostics"))

            # run hddm function ------------------------------------------
            drift_diffusion_hddm(
                data=data,
                samples=nr_samples,
                n_jobs=nr_models,
                run=RUN_ALL_MODELS,
                parallel=parallel,
                model_name=full_model_name,
                model_dir=BASE_MODEL_DIR,        
                version=version,
                phase=phase,
                accuracy_coding=True
            )

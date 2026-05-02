# Veronika Wendler
# 22.01.25
# code for the attentional drift diffusion model
# originally, I used this in summer 2024 in Quebec and was inspired by Jan WIllem De Gee's framework somewhere on his GitHub; but this version is pretty much my creation

# import libraries  #
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
from joblib import Parallel, delayed
import cloudpickle, dill
import dill as pickle  # to create the pkl object
import pymc as pm
from kabuki.hierarchical import Knode

cloudpickle.dump = dill.dump

# for running on the cluster
#dummy _gdbm module so “import _gdbm” never fails
import types, sys
sys.modules.setdefault('winreg', types.ModuleType('winreg'))
sys.modules.setdefault('_gdbm', types.ModuleType('_gdbm'))
# -------------------------------------------------------------------------

import dill as pickle
from copy import deepcopy   # for modfiying z to be 0.55 (like in Sebastian's Matlab)
import argparse

# warning settings#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Plotting
# Stats 
from statsmodels.distributions.empirical_distribution import ECDF
# HDDM
from hddm.simulators.hddm_dataset_generators import simulator_h_c

from pathlib import Path

# Import my own libraries - I don't really use it anymore 
#current_directory = os.getcwd()    # we don't use this on the cluster

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

#from helper_functions_2 import prepare_data
#import compact_models

# for Z bias coding
from scipy.special import expit   # for inverse‑logit 


# This was an attempt to code a z-link function
def make_z_link(full_stimulus_vector):
    stim = np.asarray(full_stimulus_vector, dtype=int)

    def _link(x):
        if stim.size < len(x):
            reps = (len(x) // stim.size) + 1
            stim_aligned = np.tile(stim, reps)[:len(x)]
        else:
            stim_aligned = stim[:len(x)]

        z = np.where(stim_aligned == 0,
                     1.0 - expit(x),  
                     expit(x)) 

        if hasattr(x, "index"):            
            return pd.Series(z, index=x.index, name="z")
        return z                       

    return _link


# for modification of z 
class HDDMRegressorZAmplified(hddm.models.HDDMRegressor):
    def __init__(self, *args, z_gain=2.0, z_eps=1e-6, **kwargs):
        self.z_gain = float(z_gain)
        self.z_eps = float(z_eps)
        super(HDDMRegressorZAmplified, self).__init__(*args, **kwargs)

    def _amplify_z(self, z):
        z_eff = 0.5 + self.z_gain * (z - 0.5)
        return np.clip(z_eff, self.z_eps, 1.0 - self.z_eps)

    def _create_stochastic_knodes(self, include):
        knodes = super(HDDMRegressorZAmplified, self)._create_stochastic_knodes(include)

        if "z_bottom" in knodes:
            knodes["z_eff_bottom"] = Knode(
                pm.Deterministic,
                "z_eff",
                doc="Amplified starting-point bias used by wfpt",
                eval=lambda x: np.clip(
                    0.5 + self.z_gain * (x - 0.5),
                    self.z_eps,
                    1.0 - self.z_eps
                ),
                x=knodes["z_bottom"],
                plot=False,
                trace=False,   # set True temporarily if you want to debug it
                hidden=True,
            )

        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDMRegressorZAmplified, self)._create_wfpt_parents_dict(knodes)

        if "z_eff_bottom" in knodes:
            wfpt_parents["z"] = knodes["z_eff_bottom"]

        return wfpt_parents
    

def print_model_debug_header(phase, version, trace_id, model_name, depends_on, reg_descr, model_cls, extra_kwargs):
    print("\n" + "="*80)
    print(f"run_model()")
    print(f"phase      : {phase}")
    print(f"version    : {version}")
    print(f"trace_id   : {trace_id}")
    print(f"model_name : {model_name}")
    print(f"model_cls  : {model_cls.__name__}")
    print(f"depends_on : {depends_on}")
    print(f"reg_descr  : {reg_descr}")
    if extra_kwargs:
        print(f"extra kwargs: {extra_kwargs}")
    print("="*80)


def zscore_column(df, col):
    x = pd.to_numeric(df[col], errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=1)
    if pd.isna(sd) or np.isclose(sd, 0):
        raise ValueError(f"Cannot z-score column '{col}' because SD is 0 or NaN.")
    df[col + "_z"] = (x - mu) / sd
    print(f"[z-score] {col:20s} mean={mu:.6f}, sd={sd:.6f} -> created {col + '_z'}")
    return df

#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------

# addm regression formula
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2 ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)
# where ß0 = intercept,
# ß1 = AttentionW,
# ß2 = InattentionW,
# PropDwell_opt = proportion of dwell time on the option with higher expected value
# PropDwell_sub = proportion of dwell time on the option with lower expected value
# V_opt​ = value if the better option
# V_sub = value of the worse option

#data['ES_AttentionW'] = (data['PropDwell_Right'] * data['p2']) - (data['PropDwell_Left'] * data['p1'])
#data['ES_InattentionW'] = (data['PropDwell_Left'] * data['p2']) - (data['PropDwell_Right'] * data['p1'])


nr_models       = 3         # number of MCMC chains
nr_samples      = 6000      # samples per chain - do 6000 (+1000 for burn-in) but for now for a quick one we do 600
parallel        = True      # parallel
model_base_name = "garcia_replication_"
model_versions  = {
    "LE":      ["LE_1","LE_2","LE_3","LE_4"],     #"LE_5","LE_6","LE_7"
    "EE":      ["EE_0", "EE_1","EE_2","EE_3","EE_4","EE_5"],
    "LE_RL":   ["LE_RL_1","LE_RL_2"],
    "For_paper": ["For_paper_1","For_paper_2","For_paper_3","For_paper_4","For_paper_5","For_paper_6","For_paper_7",
                  "For_paper_8","For_paper_9","For_paper_10","For_paper_11", "For_paper_12", "For_paper_13", "For_paper_14", 
                  "For_paper_15", "For_paper_16", "For_paper_17", "For_paper_18", "For_paper_19", "For_paper_20", "For_paper_21", 
                  "For_paper_22", "For_paper_23", "For_paper_24", "For_paper_25", "For_paper_26", "For_paper_27", "For_paper_28", "For_paper_29"],
    "Final":    ["Final_0", "Final_1", "Final_2", "Final_3", "Final_4", "Final_5", "Final_6", "Final_7"]
}


PHASE_TO_SOURCE = {
    "ES_ZBIAS": "ES", 
    "ES_quad": "ES",    
    "LE_RL": "LE",
    "ES_VAL": "ES",
    "For_paper": "ES",
    "EE": "EE",
    "Final": "ES",
}

# BATCH-RUN CONTROL
PHASE_RUN_ORDER = ["Final"]                                         # order
SKIP_PHASES     = {"LE","ES_ZBIAS","ES","ES_VAL","EE","ES_quad", "ESEE", "LEESEE", "LE_RL", "For_paper"}  # ignored this phase
RUN_ALL_MODELS  = True                                           # False = just load existing fits (but loading is done in the aDDM_Garcia_LE_ES_EE.py file)

# selectivity
start_phase = "Final"
start_version = 4
started = False


DATA_FILE   = Path(os.getenv(
    "DATA_FILE",
    (PROJECT_DIR / "data_sets" / "data_sets_Garcia" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv").as_posix()
)).resolve()

BASE_MODEL_DIR = Path(os.getenv("MODEL_DIR", (PROJECT_DIR / "models_dir_garcia").as_posix())).resolve()
FIG_DIR_ROOT   = Path(os.getenv("FIG_DIR",   (PROJECT_DIR / "figures_dir_garcia").as_posix())).resolve()
LOG_DIR        = Path(os.getenv("LOG_DIR",   (PROJECT_DIR / "logs").as_posix())).resolve()

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

ensure_dir(BASE_MODEL_DIR)
ensure_dir(FIG_DIR_ROOT)
ensure_dir(LOG_DIR)




# reporting function
# can be seen in the cluster output
def quick_report(data, phase, version, model_name, phase_key):
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
#def ensure_dir(directory):
#    if not os.path.exists(directory):
#        os.makedirs(directory)


# function to clean bits of the data that have not been cleaned yet, for instance remaining NAN's and so on
def sanitize_infdata(infdata):
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


#####################################################################################################################################################################
# drift diffusion models
#------------------------------------------------------------------------------------------------------------------
# function that runs/defines the different versions/models of DDM regressions for the selected phase or phases

def run_model(trace_id, data, model_dir, model_name, version, phase, samples=6000, accuracy_coding=True): 
    import os
    import numpy as np
    import hddm
    from patsy import dmatrix  

    # ensure_dir(model_dir)   
    
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
    
    
        # #  Fix z at 0.55 #
        # cfg = deepcopy(hddm.model_config.model_config['ddm_hddm_base'])
        # idx_z = cfg['params'].index('z')        # position 2 in ['v','a','z','t'] according to hddm source code but not sure if this works
        # cfg['params_default'][idx_z] = 0.55     # slight bias towards E
        # assert cfg['params'][idx_z] == 'z'
        # assert cfg['params_default'][idx_z] == 0.55, \
        #     f"z default not 0.55 but {cfg['params_default'][idx_z]}"
        # m = hddm.models.HDDMRegressor(
        #     data,
        #     reg_descr,
        #     depends_on=depends_on,
        #     p_outlier=.05,
        #     include=['a', 't', 'v'],     #  z is not in include becuase not a free param
        #     group_only_regressors=False,
        #     keep_regressor_trace=True,
        #     model_config=cfg
        # )
        # zi = m.model_config['params'].index('z')
        # print(f"default for 'z' = {m.model_config['params_default'][zi]}\n")  
        # print("sampling nodes in m.nodes_db:\n",
        #       [n for n in m.nodes_db.index if n.split('_')[0] in ['a','t','v','z']])

        # m.find_starting_values()
        # infdata = m.sample(
        #     samples,
        #     burn=200,
        #     dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'),
        #     db='pickle',
        #     return_infdata=True,
        #     loglike=True,
        #     ppc=True
        # )

        # # final check that z never got sampled
        # assert "z" not in infdata.posterior.data_vars, \
        #     "ERROR: 'z' appeared in the posterior!"
        # print("z absent from posterior - confirmed fixed.")

        # return m, infdata
    
    
    elif phase == 'EE':
        accuracy_coding = True
        depends_on = {}
        if version == 0:     # m1 # this is the 0 model with fully fixed parameters across OV levels
            v_reg = {'model': 'v ~ 0 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:  # m2 attentional weight parameter (fixated) option weights varies by OV level 
            v_reg = {'model': 'v ~ 0 + AttentionW:C(OVcate) + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:  #m3  non-fixated option weights varies by OV level
            v_reg = {'model': 'v ~ 0 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]       
        elif version == 3: # m4 non-fixated options weights varies by OV level and boundary separation
            v_reg = {'model': 'v ~ 0 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'}      
        elif version == 4: # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 0 + AttentionW + InattentionW:C(OVcate)', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'t': 'OVcate'}  
        elif version == 5: # m5 non-fixated options weights varies by OV level and non-dec. time
            v_reg = {'model': 'v ~ 0 + AttentionW + InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a': 'OVcate'}      
        
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

        

    elif phase == "For_paper":
        depends_on = {}
        model_cls = hddm.models.HDDMRegressor
        extra_model_kwargs = {}

        # 0 model:
        if version == 0:
            # jsut start sampling

            m = hddm.models.HDDM(data, 
                                p_outlier=.05, 
                                include=['a', 't', 'v', 'z'],   #'z'
                                depends_on=depends_on,
                                    )
            m.find_starting_values()
            infdata = m.sample(samples,
                               burn=1000,
                               dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                               db='pickle',
                               return_infdata=True, loglike=True, ppc=True)
            return m, infdata
        # aDDM + SP
        elif version == 1:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 2:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 3:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 4:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        # ES - dual inattention models - do what Sebastian said regarding the recoding/cahnging the sgin of the ES_InattnetionW_S param

        elif version == 5:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 6:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 7:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 8:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
        elif version == 9:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW_E + ES_AttentionW_S + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        # again with more samples (10 thousand each)
        elif version == 10:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW_E + ES_AttentionW_S + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        
        
        elif version == 11:
            v_reg = {'model': 'v ~ 0 + V_E + V_S + PropDwell_Left + PropDwell_Right', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        elif version == 12:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW_dwell + ES_InattentionW_E_dwell + ES_InattentionW_S_dwell', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # z is inlcuded - DDM + SP    - 6000 samples
        elif version == 13:
            v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # z not included
        elif version == 14:
            v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # z not included
        elif version == 15:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            
        # z included
        elif version == 16:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_IAW_chart + ES_IAW_image', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 17:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_IAW_chart + ES_IAW_image', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on={'a':'OVcate'}
             
        #---------------------------------------------------------------------------------------------------------------------
        # Models with NON-Informative Priors (Why? Because we need more freedom around z and less hierarchical shrinkage)
        # the only change is informative=False in the model description - before it was True

        elif version == 18:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            depends_on = {'a': 'OVcate'} 
            
        # z not included
        elif version == 19:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW_E + ES_AttentionW_S + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 20:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 21:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            # has sz
        elif version == 22:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 23:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            sv_reg = {'model': 'sv ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg, sv_reg]
        elif version == 24:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            a_reg = {'model': 'a ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg, a_reg]
        elif version == 25:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_IAW_chart + ES_IAW_image', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        # try some costume prior for z (that is not too restricitive but closer to E, e.g. 0.4, 0.5 and add sz)    
        elif version == 26:
            def z_link(x, z0=0.4):
                # logit(z0)
                offset = np.log(z0 / (1.0 - z0))
                return 1.0 / (1.0 + np.exp(-(x + offset)))
            v_reg = {
                'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S',
                'link_func': lambda x: x
            }
            # intercept-only model for z
            z_reg = {
                'model': 'z ~ 1',
                'link_func': lambda x: z_link(x, z0=0.4)
            }
            reg_descr = [v_reg, z_reg]
        
        elif version == 27:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
            model_cls = HDDMRegressorZAmplified
            extra_model_kwargs = {'z_gain': 2.0}
        

        elif version == 28:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW_z + ES_InattentionW_E_z + ES_InattentionW_S_z + memory_precision_z + ES_AttentionW_z:memory_precision_z + ES_InattentionW_E_z:memory_precision_z + ES_InattentionW_S_z:memory_precision_z', 'link_func': lambda x: x}
            reg_descr = [v_reg]

        else:
            raise ValueError(f"Invalid version {version}")
        
        # ---------- shared model build / sample block ----------
        print_model_debug_header(
            phase=phase,
            version=version,
            trace_id=trace_id,
            model_name=model_name,
            depends_on=depends_on,
            reg_descr=reg_descr,
            model_cls=model_cls,
            extra_kwargs=extra_model_kwargs
        )

        m = model_cls(
            data,
            reg_descr,
            p_outlier=.05,
            include=['a', 't', 'v', 'z', 'sv', 'sz'],
            depends_on=depends_on,
            group_only_regressors=False,
            keep_regressor_trace=True,
            **extra_model_kwargs
        )

        print("Model created.")
        print("nodes containing 'z':", [n for n in m.nodes_db.index if 'z' in n])

        if version == 27:
            print(f"z amplification active: z_eff = 0.5 + {extra_model_kwargs['z_gain']} * (z - 0.5)")
            test_vals = np.array([0.40, 0.45, 0.50, 0.55, 0.60])
            z_eff_vals = 0.5 + extra_model_kwargs['z_gain'] * (test_vals - 0.5)
            print("example mapping:")
            for z0, z1 in zip(test_vals, z_eff_vals):
                print(f"  z={z0:.2f} -> z_eff={z1:.2f}")

        m.find_starting_values()
        print("Starting values found. Beginning sampling...")

        infdata = m.sample(
            samples,
            burn=1000,
            dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'),
            db='pickle',
            return_infdata=True,
            loglike=True,
            ppc=True
        )

        print("Sampling finished.")
        print("posterior vars with 'z':", [v for v in infdata.posterior.data_vars if 'z' in v])

        return m, infdata


        
    elif phase == "Final":
        
        depends_on = {}
        with_z = version in [0, 1, 4, 5]

        include_params = ['a', 't', 'v']
        if with_z:
            include_params.append('z')
        
        # with z 
        if version == 0:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 1:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]

        # without z
        elif version == 2:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 3:
            v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]

        # with z + intercept (to check ppc)
        elif version == 4:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 5:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]

        # without z + intercept (to check ppc)
        elif version == 6:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        elif version == 7:
            v_reg = {'model': 'v ~ 1 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
            reg_descr = [v_reg]
        
        else:
            raise ValueError(f"Is this version correct ? ")   
        

        m = hddm.models.HDDMRegressor(data, 
                                    reg_descr,
                                    p_outlier=.05, 
                                    include=include_params,   #'z'
                                    depends_on=depends_on,
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
        
    
    
    elif phase == 'LE_RL':
        if version == 0:
            m = hddm.models.HDDMrl(data, include=['a', 't', 'v', 'alpha'])
            m.find_starting_values()
            infdata = m.sample(samples,
                               burn=300,
                               dbname=os.path.join(model_dir, model_name + f'_db{trace_id}'), 
                               db='pickle',
                               return_infdata=True, loglike=True, ppc=False)

            return m, infdata
        else:
            raise ValueError(f"Invalid version {version}")
    
     
###############################################################################################################    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main function for running/loading models

#
### ONLY FOR RUNIING RL MODELS ##-------------------------------------------------------------------------------------------
# def run_and_save(trace_id, data, model_dir, model_name, version, phase, samples):
#     # instantiate + sample
#     model, infdata = run_model(
#         trace_id=trace_id,
#         data=data,
#         model_dir=model_dir,
#         model_name=model_name,
#         version=version,
#         phase=phase,
#         samples=samples,
#     )
#     fname = f"{model_name}_{trace_id}"
#     model.save(os.path.join(model_dir, fname + ".hddm"))
#     with open(os.path.join(model_dir, fname + ".pkl"), "wb") as f:
#         pickle.dump(model, f)
#     infdata = sanitize_infdata(infdata)
#     az.to_netcdf(infdata, os.path.join(model_dir, fname + ".nc"))
#     return fname
#----------------------------------------------------------------------------------------------------------------

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

from joblib import Parallel, delayed

def drift_diffusion_hddmRL(
    data,
    samples=2000,
    n_jobs=3,
    run=True,
    parallel=True,
    model_name='model',
    model_dir='.',
    version=None,
    phase=None,
):
    if not run:
        print('Loading existing RL models')
        return [
            hddm.load(os.path.join(model_dir, f"{model_name}_{i}.hddm"))
            for i in range(n_jobs)
        ]

    start_time = time.time()
    if parallel:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(run_model)(
                trace_id=i,
                data=data,
                model_dir=model_dir,
                model_name=model_name,
                version=version,
                phase=phase,
                samples=samples,
            )
            for i in range(n_jobs)
        )
        for i, (model, infdata) in enumerate(results):
            fname = f"{model_name}_{i}"
            # model.save(os.path.join(model_dir, fname + ".hddm"))
            # with open(os.path.join(model_dir, fname + ".pkl"), "wb") as f:
            #     pickle.dump(model, f)
            infdata = sanitize_infdata(infdata)
            az.to_netcdf(infdata, os.path.join(model_dir, fname + ".nc"))

        print(f"RL chains finished and saved: {[f'{model_name}_{i}' for i in range(n_jobs)]}")
    else:
        model, infdata = run_model(
            trace_id=0,
            data=data,
            model_dir=model_dir,
            model_name=model_name,
            version=version,
            phase=phase,
            samples=samples,
        )
        fname = f"{model_name}_0"
        model.save(os.path.join(model_dir, fname + ".hddm"))
        with open(os.path.join(model_dir, fname + ".pkl"), "wb") as f:
            pickle.dump(model, f)
        infdata = sanitize_infdata(infdata)
        az.to_netcdf(infdata, os.path.join(model_dir, fname + ".nc"))
        print(f"RL chain finished and saved: {fname}")

    print("Time elapsed:", time.time() - start_time, "s")
    
#########################################################################################################################################################

# directories
#model_dir = 'models_dir_garcia/'
#ensure_dir(model_dir)

model_dir = BASE_MODEL_DIR
#runs every (phase, version) pairing
if __name__ == "__main__":

    print(f"Reading data from: {DATA_FILE}")
    data_full = pd.read_csv(DATA_FILE.as_posix(), sep=",")
    # loop over phases and versions
    for phase in PHASE_RUN_ORDER:
        if phase in SKIP_PHASES:
            continue                    
        
        phase_key = phase

        for version, model_name in enumerate(model_versions[phase]):
            if not started:
                if phase == start_phase and version >= start_version:
                    started = True
                elif PHASE_RUN_ORDER.index(phase) > PHASE_RUN_ORDER.index(start_phase):
                    started = True
                else:
                    continue 
            
            full_model_name = model_base_name + model_name
            print(f"\n PHASE {phase} : {model_name}  ===")

            # filter data for this phase
            source_phase = PHASE_TO_SOURCE.get(phase, phase)   #assignes ES_ZBIAS

            if phase == "ESEE":
                data = data_full[data_full["phase"].isin(["ES", "EE"])].copy()
            elif phase == "LEESEE":
                data = data_full[data_full["phase"].isin(["LE", "ES", "EE"])].copy()
            else:
                data = data_full[data_full["phase"] == source_phase].copy() 
            
            if data.empty:
                raise ValueError(f"No rows left after filtering for phase '{phase}' "
                                 f"(source = '{source_phase}')")

            # preprocessing 
            data["gazeCI"]  = pd.to_numeric(data["gazeCI"],  errors="coerce")
            data["gazeSE"]= pd.to_numeric(data["gazeSE"],errors="coerce")
            data["phase"]       = data["phase"].astype("category")
            data["rt"]          = pd.to_numeric(data["rtime"], errors="coerce")
            data["chose_right"] = pd.to_numeric(data["chose_right"], errors="coerce")
            data["chose_left"] = pd.to_numeric(data["chose_left"], errors="coerce")
            
            data = zscore_column(data, "ES_AttentionW")
            data = zscore_column(data, "ES_InattentionW_E")
            data = zscore_column(data, "ES_InattentionW_S")

            data                = data[data["rt"] > 0.250]
            # data["response"]    = pd.to_numeric(data["corr"], errors="coerce")

            # here, it's important to be selctive depending on whether chose_right or chose_left is the upper bound
            if phase in ("ES_ZBIAS", "ES_quad", "ES_VAL", "For_paper", "Final"):

                data["response"] = pd.to_numeric(data["chose_right"], errors="coerce")
                print("head of response mapping:")
                print(data[["chose_right","corr","response"]].head(10).to_string(index=False))
                print("counts:", data["response"].value_counts(dropna=False).to_dict())
                # sanity checks ...are we actually filtering the right things
                mismatches = (data["response"] != data["chose_right"]).sum()
                assert mismatches == 0, f"{mismatches} rows where response ≠ chose_right!"
            else:
                data["response"] = pd.to_numeric(data["corr"], errors="coerce")
                print(data[["chose_right","corr","response"]].head(5).to_string(index=False))

            
            print(f"phase={phase}  response counts:\n",
            data["response"].value_counts(dropna=False).head())
            data["OVcate"]      = data["OVcate_2"].astype("category")
            data["Abscate"]     = data["Abscate_2"].astype("category")
            data["cond"]     = data["cond"].fillna(-1)
            data["cond"]     = data["cond"].astype("int")
            data["AttentionW"]  = pd.to_numeric(data["AttentionW"],  errors="coerce")
            data["InattentionW"]= pd.to_numeric(data["InattentionW"],errors="coerce")
            data["ES_AttentionW"]  = pd.to_numeric(data["ES_AttentionW"],  errors="coerce")
            data["ES_InattentionW"]= pd.to_numeric(data["ES_InattentionW"],errors="coerce")
            data["subj_idx"]    = data["sub_id"]
            #data["stimulus"] = pd.to_numeric(data["stimulus"], errors="coerce")
            data["DTA"] = pd.to_numeric(data["DTA"],errors="coerce")
            data["DwellPropAdvantage"] = pd.to_numeric(data["DwellPropAdvantage"],errors="coerce")
            data["DwellLeft"]  = pd.to_numeric(data["DwellLeft"],  errors="coerce")
            data["DwellRight"] = pd.to_numeric(data["DwellRight"], errors="coerce")
            data["AttentionW_E"]  = pd.to_numeric(data["AttentionW_E"],  errors="coerce")
            data["AttentionW_S"] = pd.to_numeric(data["AttentionW_S"], errors="coerce")
            data["InattentionW_E"] = pd.to_numeric(data["InattentionW_E"], errors="coerce")
            data["InattentionW_S"] = pd.to_numeric(data["InattentionW_S"], errors="coerce")
            data["ES_AttentionW_E"]  = pd.to_numeric(data["ES_AttentionW_E"],  errors="coerce")
            data["ES_AttentionW_S"] = pd.to_numeric(data["ES_AttentionW_S"], errors="coerce")
            data["ES_InattentionW_E"] = pd.to_numeric(data["ES_InattentionW_E"], errors="coerce")
            data["ES_InattentionW_S"] = pd.to_numeric(data["ES_InattentionW_S"], errors="coerce")
            data["ES_AttentionW"]  = pd.to_numeric(data["ES_AttentionW"],  errors="coerce")
            data["ES_InattentionW"]  = pd.to_numeric(data["ES_InattentionW"],  errors="coerce")
            data["V_E"] = pd.to_numeric(data["V_E"], errors="coerce")
            data["V_S"] = pd.to_numeric(data["V_S"], errors="coerce")
            data["Value_diff"] = pd.to_numeric(data["Value_diff"], errors="coerce")
            data["ES_AttentionW_S_dwell"] = pd.to_numeric(data["ES_AttentionW_S_dwell"], errors="coerce")
            data["ES_InattentionW_E_dwell"] = pd.to_numeric(data["ES_InattentionW_E_dwell"], errors="coerce")
            data["ES_InattentionW_S_dwell"] = pd.to_numeric(data["ES_InattentionW_S_dwell"], errors="coerce")
            
            # keep only trials with strictly positive dwell time on both sides, this can be changed; depends on the goal
            #data = data[(data["DwellLeft"] > -1) & (data["DwellRight"] > -1)]
            data = data[~data["subj_idx"].isin({1,4,5,6,14,99})]
            data = data.dropna(subset=["rt",
                                       "response",
                                       "OVcate",
                                       "Abscate",
                                       "subj_idx",
                                       "AttentionW",
                                       "InattentionW",
                                       "cond",
                                       "gazeSE",
                                       "gazeCI",
                                       "ES_AttentionW",
                                       "ES_InattentionW",
                                       "chose_left",
                                       "DTA",
                                       "AttentionW_E",
                                       "AttentionW_S",
                                       "InattentionW_E",
                                       "InattentionW_S",
                                       "z_AttentionW_E",
                                       "z_AttentionW_S",
                                       "z_InattentionW_E",
                                       "z_InattentionW_S",
                                       "V_E", "V_S", "Value_diff",
                                       "ES_AttentionW_E",
                                       "ES_AttentionW_S",
                                       "ES_InattentionW_E",
                                       "ES_InattentionW_S",
                                       "ES_AttentionW",
                                       "ES_InattentionW",
                                       "ES_AttentionW_S_dwell",
                                       "ES_InattentionW_E_dwell",
                                       "ES_InattentionW_S_dwell",
                                       ])   
            
            # quick report at the start
            quick_report(data, phase, version, model_name, phase_key)

            # fig_dir = os.path.join("figures_dir_garcia", full_model_name)
            # ensure_dir(os.path.join(fig_dir, "diagnostics"))
            
            fig_dir = FIG_DIR_ROOT / full_model_name
            ensure_dir(fig_dir / "diagnostics")

            # # run hddm function 
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
            
            #only when running RL in the LE phase #
            # drift_diffusion_hddmRL(
            #     data=data,
            #     samples=nr_samples,
            #     n_jobs=nr_models,
            #     run=RUN_ALL_MODELS,
            #     parallel=parallel,
            #     model_name=full_model_name,
            #     model_dir=BASE_MODEL_DIR,        
            #     version=version,
            #     phase=phase,
            # )
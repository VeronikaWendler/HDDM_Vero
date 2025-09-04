# adding a new column to a dataframe
import pandas as pd  
import numpy as np 


data = pd.read_csv("C:/Cluster_Github/HDDM_Vero/data_sets/data_sets_OV/OVParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv")

data['chose_right'] = data['chose_right'].astype(int)
data['chose_left']  = 1 - data['chose_right']          # 1 = left, 0 = right
data['gazeCI'] = data['DwellTime_opt'] - data['DwellTime_sub']
data['gazeSE'] = data['DwellRight'] - data['DwellLeft']
# --- first-fix location ----------------------------------------------
data['FirstFix_Left']  = (data['FirstFixLoc']  == 1).astype(int)
data['FirstFix_Right'] = (data['FirstFixLoc']  == 2).astype(int)

# --- final-fix location ----------------------------------------------
data['FinalFix_Left']  = (data['FinalFixLoc']  == 1).astype(int)
data['FinalFix_Right'] = (data['FinalFixLoc']  == 2).astype(int)

# # --- middle-dominant location ----------------------------------------
# data['MiddleDominantLoc_Left']  = (data['MiddleDominantLoc'] == 1).astype(int)
# data['MiddleDominantLoc_Right'] = (data['MiddleDominantLoc'] == 2).astype(int)

# # sequence model cols

# data['ESE'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Right'] & data['FinalFix_Left']).astype(int)
# data['ESS'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Right'] & data['FinalFix_Right']).astype(int)
# data['EEE'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Left'] & data['FinalFix_Left']).astype(int)
# data['EES'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Left'] & data['FinalFix_Right']).astype(int)

# data['SES'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Left'] & data['FinalFix_Right']).astype(int)
# data['SEE'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Left'] & data['FinalFix_Left']).astype(int)
# data['SSS'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Right'] & data['FinalFix_Right']).astype(int)
# data['SSE'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Right'] & data['FinalFix_Left']).astype(int)

# # test easy models:

# data['ES_first'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Right']).astype(int)
# data['EE_first'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Left']).astype(int)

# data['SE_first'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Left']).astype(int)
# data['SS_first'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Right']).astype(int)


# data['ES_final'] = (data['FirstFix_Left'] & data['FinalFix_Right']).astype(int)
# data['EE_final'] = (data['FirstFix_Left'] & data['FinalFix_Left']).astype(int)

# data['SE_final'] = (data['FirstFix_Right'] & data['FinalFix_Left']).astype(int)
# data['SS_final'] = (data['FirstFix_Right'] & data['FinalFix_Right']).astype(int)


# data['SS_middle'] = (data['MiddleDominantLoc_Right'] & data['FinalFix_Right']).astype(int)
# data['SE_middle'] = (data['MiddleDominantLoc_Right'] & data['FinalFix_Left']).astype(int)
# data['ES_middle'] = (data['MiddleDominantLoc_Left'] & data['FinalFix_Right']).astype(int)
# data['EE_middle'] = (data['MiddleDominantLoc_Left'] & data['FinalFix_Left']).astype(int)

# computing regressors for the addm according to Sebastian:
#driftRate = driftConstant*((fix(t,f)==1)*(choiceSet(t,1)-theta*choiceSet(t,2))+...
#                                   (fix(t,f)==0)*(theta*choiceSet(t,1)-choiceSet(t,2)));
# the Sebastian drift rate regression - p1 is value on the left and p2 is value on the right ...this is how we get teh attntional and inattentional parameters..
# v = b0 + b1(PropDwell_Right * p2 - PropDwell_Left * p1) + b2(PropDwell_Right* p1 - PropDwell_Left*p2) + e

data['ES_AttentionW'] = (data['PropDwell_Right'] * data['p2']) - (data['PropDwell_Left'] * data['p1'])
data['ES_InattentionW'] = (data['PropDwell_Left'] * data['p2']) - (data['PropDwell_Right'] * data['p1'])

#old convention gives you negative predictor
#data['ES_InattentionW_E'] = (data['PropDwell_Right'] * data['p1']) 
data['ES_InattentionW_S'] = (data['PropDwell_Left'] * data['p2']) 

#new changes:
# flip sign so “more unattended E” now codes as *more* evidence in the positive direction
data['ES_InattentionW_E'] = - (data['PropDwell_Right'] * data['p1'])

# just to experiment a final time:
data['ES_AttentionW_S'] = (data['PropDwell_Right'] * data['p2'])
# this would also need to be flipped
data['ES_AttentionW_E'] = - (data['PropDwell_Left'] * data['p1']) 


# the CCT drift rate regression 
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2 ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ϵ

# max_abs = np.nanmax(np.abs(data['DwellTimeAdvantage']))
# data['dta_norm'] = data['DwellTimeAdvantage'] / max_abs
# data['z_dynamic']   = (data['dta_norm'] + 1) / 2                            # dynamic z sscaled by gaze
# data['z_static'] = 0.55                                                     #Sebastian's idea 
#
# # Which option is the correct one?
# data['target_option'] = np.where(data['p1'] > data['p2'], 'E', 'S')
# # stimulus
# data['stimulus'] = np.where(data['target_option']=='E', 1, 0)
# # value difference
# data['val_diff'] = data['V_corr'] - data['V_sub']
# # Response code (0/1)
# data['resp'] = (data['chose_left']==data['stimulus']).astype(int)
# # flipping val_diff so it’s positive when resp=1 (when they chose the correct side)
# data['val_diff*'] = data['val_diff'] * data['resp'].map({1:1, 0:-1})
# # gaze-imbalance regressors because we can see in emp. data that much gaze to either option is disadvantageous
# data['abs_DwellPropAdv'] = data['DwellPropAdvantage'].abs()   # |Prop_S – Prop_E|
# data['gaze_quad']= data['DwellPropAdvantage'] ** 2             # (Prop_S – Prop_E)^2



# v = β0  + β_deltaV * (V_S - V_E) + β_lin * (G_S - G_E) + β_quad * (G_S - G_E)**2

# data["gaze_bal"]   = 1 - data["DwellPropAdvantage"]**2     # penalty (1 at centre, 0 at extremes)
# data["val_bal_int"] = data["val_diff"] * data["gaze_bal"]  # interaction that drives drift

# # new predictors (let's not compute quadratic - easier)


# -----------------------------------------------------------------
data['DTA']   = data['PropDwell_Right'] - data['PropDwell_Left']
data['DTA2']  = data['DTA'] ** 2
data['absDTA']= np.abs(data['DTA'])


# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2 ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ϵ
data['AttentionW'] = (data['PropDwell_opt'] * data['V_corr']) - (data['PropDwell_sub'] * data['V_sub'])
data['InattentionW'] = (data['PropDwell_sub'] * data['V_corr']) - (data['PropDwell_opt'] * data['V_sub'])
data['AttentionW'] = data['AttentionW'].round(3)
data['InattentionW'] = data['InattentionW'].round(3)
V_C = data['p2']          # chart EV
V_I = data['p1']          # image EV
# dummies which format has lower EV
data['chart_is_sub']  = (V_C < V_I).astype(int)
data['image_is_sub']  = (V_I < V_C).astype(int)

#splitting InattentionW
data['IAW_chart'] = data['InattentionW'] * data['chart_is_sub']
data['IAW_image'] = data['InattentionW'] * data['image_is_sub']
data['IAW_chart'] = data['IAW_chart'].round(3)
data['IAW_image'] = data['IAW_image'].round(3)


# #---------- Early vs Late --------------------------------------------
# data["early"] = reaction time lower than mean  as type int
# data["late"] = rection time higher than mean  as tpye int

# data["AttentionW_early"] = data["AttentionW"] * data["early"]
# data["AttentionW_late"] = data["AttentionW"] * data["late"]
# data["InattentionW_early"] = data["InattentionW"] * data["early"]
# data["InattentionW_late"] = data["InattentionW"] * data["late"]

# # R rule: threshold_rt <- mean(rtime)/2
# threshold_rt = data["rtime"].mean(skipna=True) / 2.0

# # dummies as requested (ints)
# data['early'] = (data["rtime"] <= threshold_rt).astype(int)
# data['late']  = (data["rtime"] >  threshold_rt).astype(int)

# # handy label for plotting/checks
# data['RT_group'] = np.where(data['early'] == 1, 'early', 'late')

# # --- build the four interaction-coded drift predictors for HDDM
# # assumes you already created AttentionW and InattentionW above
# data['AttentionW_early']   = data['AttentionW']   * data['early']
# data['AttentionW_late']    = data['AttentionW']   * data['late']
# data['InattentionW_early'] = data['InattentionW'] * data['early']
# data['InattentionW_late']  = data['InattentionW'] * data['late']



# in hddm v = b0 + b1*AttentionW_early + b2*InattentionW_early + AttentionW_late* InattentionW_late



#regressors
data['val_diff_corr'] = data['V_corr'] - data['V_sub']
# gaze balance weight - gaze is disadvantageous for choice --> reduces accuracy
# given teh strange u shape, we could assume that at both extremes gaze acts even more
# --> non-linear effect
# data['w'] = 1 - data['DwellPropAdvantageCorrect']**2  
# data['w_dv'] = data['w'] * data['val_diff_corr']   # --> in a second model this could also interact with OV
# data['absDPAC']= np.abs(data['DwellPropAdvantageCorrect'])



# data["abs_DwellPropAdvCorr"] = data["DwellPropAdvantageCorrect"].abs()   # is large when the difference between correct and incorrect is large
# data["balance"] = 1 - data["abs_DwellPropAdvCorr"]  # is close to 1 when gaze is balanced (linear)

# # v = b0 + b1*abs_DwellPropAdvCorr + b2*balance:C(OVcate)
# # a = 

# # v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2 ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ϵ

# data['AW_bal']  = data['AttentionW']   * data['balance']
# data['IAW_bal'] = data['InattentionW'] * data['balance']







########################################################################################################################
# original drift rate formula:
# v = β0 + β1 * (PropDwell_opt​ * V_opt​ − PropDwell_sub * V_sub) + β2 * (PropDwell_sub * V_opt​ − PropDwell_opt​ * V_sub)+ϵ

# dirft rate with two separate thetas for S and E :
# v = β0 + β1 * AttentionW_E + β2 * AttentionW_S + β3 * InattentionW_E + β4 * InattentionW_S  +ϵ

# where:
# AttentionW_E = Value_E_opt * DwellProp_E - Value_S_sub * DwellProp_S
# AttentionW_S = Value_S_opt * DwellProp_S - Value_E_sub * DwellProp_E
# InattentionW_E = Value_E_opt * DwellProp_S - Value_S_sub * DwellProp_E
# InattentionW_S = Value_S_opt * DwellProp_E - Value_E_sub * DwellProp_S

#where:

# Value_E_opt = value of E-option when E option > S option on that trial
# Value_E_sub = value of E-option when E option < S option on that trial
# Value_S_sub = value of S-option when S option < E option on that trial
# Value_S_opt = value of S-option when S option > E option on that trial
# DwellProp_E = proportion of dwell time on E option
# DwellProp_S = proportion of dwell time on S option


value_left  = data['p1']   # or data['ev1']
value_right = data['p2']   # or data['ev2']

# Dwell proportions for the identities (E on left, S on right)
data['DwellProp_E'] = data['DwellLeft']  / data['DwellTotal']
data['DwellProp_S'] = data['DwellRight'] / data['DwellTotal']
data.loc[data['DwellTotal'] == 0, ['DwellProp_E', 'DwellProp_S']] = np.nan

# which option is optimal on each trial
E_is_opt = value_left  > value_right
S_is_opt = value_right > value_left
is_tie   = value_left  == value_right   # let's exclude

data['Value_E_opt'] = np.where(E_is_opt & ~is_tie, data['p1'], 0.0)
data['Value_E_sub'] = np.where(S_is_opt & ~is_tie, data['p1'], 0.0)
data['Value_S_opt'] = np.where(S_is_opt & ~is_tie, data['p2'], 0.0)
data['Value_S_sub'] = np.where(E_is_opt & ~is_tie, data['p2'], 0.0)

# For ties all masked values set to 0 
data.loc[is_tie, ['Value_E_opt','Value_E_sub','Value_S_opt','Value_S_sub']] = np.nan
# regressors
data['AttentionW_E']   = data['Value_E_opt'] * data['DwellProp_E'] - data['Value_S_sub'] * data['DwellProp_S']
data['AttentionW_S']   = data['Value_S_opt'] * data['DwellProp_S'] - data['Value_E_sub'] * data['DwellProp_E']
data['InattentionW_E'] = data['Value_E_opt'] * data['DwellProp_S'] - data['Value_S_sub'] * data['DwellProp_E']
data['InattentionW_S'] = data['Value_S_opt'] * data['DwellProp_E'] - data['Value_E_sub'] * data['DwellProp_S']



to_z = ['AttentionW',            # symmetric, continuous
        'InattentionW',          # symmetric, continuous
        "AttentionW_E", "AttentionW_S", "InattentionW_E", "InattentionW_S",
        ]             

for c in to_z:
    mu, sd = data[c].mean(), data[c].std()
    data[f'z_{c}'] = ((data[c] - mu) / sd).round(4)  

data['z_IAW_chart'] = data['z_InattentionW'] * data['chart_is_sub']
data['z_IAW_image'] = data['z_InattentionW'] * data['image_is_sub']




# # Early vs Late per-subject median split + gated regressors 
# phase_filter = "ES"
# excluded_subjects = [1, 4, 5, 6, 14, 99]
# data = (data.query("phase == @phase_filter").loc[~data['sub_id'].isin(excluded_subjects)].copy())
# data = data.dropna(subset=['DwellTimeAdvantage', 'OVcate_2', 'chose_right', 'corr'])
# data['rt_median_subj'] = data.groupby('sub_id')['rtime'].transform('median')

# #early/late dummies
# data['early'] = (data['rtime'] <= data['rt_median_subj']).astype(int)
# data['late']  = (data['rtime'] > data['rt_median_subj']).astype(int)
# data['RT_group'] = np.where(data['early'] == 1, 'early', 'late')
# data['trial_type'] = data['rtime'] <= data['rt_median_subj']
# data['trial_type'] = data['trial_type'].map({True: 'ET', False: 'LT'})
# base_cols = ['AttentionW', 'InattentionW',
#              'ES_AttentionW', 'ES_InattentionW',
#              'AttentionW_E', 'AttentionW_S', 'InattentionW_E', 'InattentionW_S']
# base_cols = [c for c in base_cols if c in data.columns]

# for c in base_cols:
#     data[f'{c}_early'] = data[c] * data['early']
#     data[f'{c}_late']  = data[c] * data['late']




#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Value models (Sebastian):
# Response coded
# Value difference model:
#v = β0 + β1 ⋅ (V_E − V_S) + ϵ
# 
# independent value model:
#v = β0 + β1 ⋅ V_E + β2 ⋅  V_S + ϵ

data["V_E"] = data["p1"]
data["V_S"] = data["p2"]

data["Value_diff"] = data["V_E"] - data["V_S"]

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# good vs poor performers





data.to_csv("C:/Cluster_Github/HDDM_Vero/data_sets/data_sets_OV/OVParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv",index=False,)





# adding a new column to a dataframe
import pandas as pd  
import numpy as np 


data = pd.read_csv(
    "C:/Cluster_Github/HDDM_Vero/data_sets/data_sets_Garcia/"
    "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv"
)

# choice side (assumes 0/1 or True/False) -------------------------
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

# --- middle-dominant location ----------------------------------------
data['MiddleDominantLoc_Left']  = (data['MiddleDominantLoc'] == 1).astype(int)
data['MiddleDominantLoc_Right'] = (data['MiddleDominantLoc'] == 2).astype(int)

# sequence model cols

data['ESE'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Right'] & data['FinalFix_Left']).astype(int)
data['ESS'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Right'] & data['FinalFix_Right']).astype(int)
data['EEE'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Left'] & data['FinalFix_Left']).astype(int)
data['EES'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Left'] & data['FinalFix_Right']).astype(int)

data['SES'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Left'] & data['FinalFix_Right']).astype(int)
data['SEE'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Left'] & data['FinalFix_Left']).astype(int)
data['SSS'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Right'] & data['FinalFix_Right']).astype(int)
data['SSE'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Right'] & data['FinalFix_Left']).astype(int)

# test easy models:

data['ES_first'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Right']).astype(int)
data['EE_first'] = (data['FirstFix_Left'] & data['MiddleDominantLoc_Left']).astype(int)

data['SE_first'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Left']).astype(int)
data['SS_first'] = (data['FirstFix_Right'] & data['MiddleDominantLoc_Right']).astype(int)


data['ES_final'] = (data['FirstFix_Left'] & data['FinalFix_Right']).astype(int)
data['EE_final'] = (data['FirstFix_Left'] & data['FinalFix_Left']).astype(int)

data['SE_final'] = (data['FirstFix_Right'] & data['FinalFix_Left']).astype(int)
data['SS_final'] = (data['FirstFix_Right'] & data['FinalFix_Right']).astype(int)


data['SS_middle'] = (data['MiddleDominantLoc_Right'] & data['FinalFix_Right']).astype(int)
data['SE_middle'] = (data['MiddleDominantLoc_Right'] & data['FinalFix_Left']).astype(int)
data['ES_middle'] = (data['MiddleDominantLoc_Left'] & data['FinalFix_Right']).astype(int)
data['EE_middle'] = (data['MiddleDominantLoc_Left'] & data['FinalFix_Left']).astype(int)

# computing regressors for the stim coded addm according to Sebastian:
#driftRate = driftConstant*((fix(t,f)==1)*(choiceSet(t,1)-theta*choiceSet(t,2))+...
#                                   (fix(t,f)==0)*(theta*choiceSet(t,1)-choiceSet(t,2)));
# the Sebastian drift rate regression - p1 is value on the left and p2 is value on the right ...this is how we get teh attntional and inattentional parameters..
# v = b0 + b1(PropDwell_Right * p2 - PropDwell_Left * p1) + b2(PropDwell_Right* p1 - PropDwell_Left*p2) + e

data['ES_AttentionW'] = (data['PropDwell_Right'] * data['p2']) - (data['PropDwell_Left'] * data['p1'])
data['ES_InattentionW'] = (data['PropDwell_Left'] * data['p2']) - (data['PropDwell_Right'] * data['p1'])
data['ES_AttentionW'] = data['ES_AttentionW'].round(3)
data['ES_InattentionW'] = data['ES_InattentionW'].round(3)

# the CCT drift rate regression 
# v = β0 + β1 ⋅ (PropDwell_opt​ ⋅ V_opt​ − PropDwell_sub ⋅ V_sub) + β2 ⋅ (PropDwell_sub ⋅ V_opt​ − PropDwell_opt​ ⋅ V_sub)+ϵ

# --- save back to disk -----------------------------------------------
# normalize DwellTimeAdvantage and create z_reg
# normalize DwellTimeAdvantage and create z_reg
max_abs = np.nanmax(np.abs(data['DwellTimeAdvantage']))
data['dta_norm'] = data['DwellTimeAdvantage'] / max_abs
data['z_dynamic']   = (data['dta_norm'] + 1) / 2                            # dynamic z sscaled by gaze
data['z_static'] = 0.55                                                     #Sebastian's idea 

# Which option is the correct one?
data['target_option'] = np.where(data['p1'] > data['p2'], 'E', 'S')
# stimulus
data['stimulus'] = np.where(data['target_option']=='E', 1, 0)
# value difference
data['val_diff'] = data['V_corr'] - data['V_sub']
# Response code (0/1)
data['resp'] = (data['chose_left']==data['stimulus']).astype(int)
# flipping val_diff so it’s positive when resp=1 (when they chose the correct side)
data['val_diff*'] = data['val_diff'] * data['resp'].map({1:1, 0:-1})
# gaze-imbalance regressors because we can see in emp. data that much gaze to either option is disadvantageous
data['abs_DwellPropAdv'] = data['DwellPropAdvantage'].abs()   # |Prop_S – Prop_E|
data['gaze_quad']= data['DwellPropAdvantage'] ** 2             # (Prop_S – Prop_E)^2



# v = β0  + β_deltaV * (V_S - V_E) + β_lin * (G_S - G_E) + β_quad * (G_S - G_E)**2




# dwell_prop ranges roughly -1 … +1.  Centre it first.
data["gaze_bal"]   = 1 - data["DwellPropAdvantage"]**2     # penalty (1 at centre, 0 at extremes)
data["val_bal_int"] = data["val_diff"] * data["gaze_bal"]  # interaction that drives drift



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



#regressors
data['val_diff_corr'] = data['V_corr'] - data['V_sub']
# gaze balance weight - gaze is disadvantageous for choice --> reduces accuracy
# given teh strange u shape, we could assume that at both extremes gaze acts even more
# --> non-linear effect
data['w'] = 1 - data['DwellPropAdvantageCorrect']**2  
data['w_dv'] = data['w'] * data['val_diff_corr']   # --> in a second model this could also interact with OV
data['absDPAC']= np.abs(data['DwellPropAdvantageCorrect'])

# ---------- choose which columns to standardise --------------
to_z = ['AttentionW',            # symmetric, continuous
        'InattentionW',          # symmetric, continuous
        'gaze_quad',             # if you use it
        'val_diff',
        'DwellPropAdvantage',
        'abs_DwellPropAdv',
        'val_diff_corr',
        'w_dv',
        'absDPAC']              # etc. add more if needed

# ---------------------------------------------
# z-score, then round to 3 decimals (example)
# ---------------------------------------------
for c in to_z:
    mu, sd = data[c].mean(), data[c].std()
    data[f'z_{c}'] = ((data[c] - mu) / sd).round(4)  

data['z_IAW_chart'] = data['z_InattentionW'] * data['chart_is_sub']
data['z_IAW_image'] = data['z_InattentionW'] * data['image_is_sub']

data.to_csv(
    "C:/Cluster_Github/HDDM_Vero/data_sets/data_sets_Garcia/"
    "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv",
    index=False,)





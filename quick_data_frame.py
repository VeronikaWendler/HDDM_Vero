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
# the Sebastian drift rate regression
# v = b0 + b1(PropDwell_Left * p1 - PropDwell_Right * p2) + b2(PropDwell_Right* p1 - PropDwell_Left*p2) + e

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

data.to_csv(
    "C:/Cluster_Github/HDDM_Vero/data_sets/data_sets_Garcia/"
    "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv",
    index=False
)


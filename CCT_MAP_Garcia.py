# # MAP estimation similar to code from Dr Chih-Chung Ting
# Veronika Wendler
# 15.03.25
# 15.03.25
# This code calculates group maximum posterior estimates of the attentional drift diffusion parameters and their parameter comparison

#libraries as always 
import pandas as pd
import pickle
import kabuki
import scipy.stats as stats
import pickle
import kabuki

# version 1 == models varying by OV (high, medium, low)
# version 2 == models varying by phase (ES EE)
# Version 3 == models varying by phase (LE ES EE)

def run_version_1_a():
    #---------------------------------------------------------------------------------------------------------------
    # Version 1: OV-modulated models (high, medium, low)
    # load and combine OV model files (set which model)
    model_paths_OV = [
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_5_4.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_5_3.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_5_2.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_5_1.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_5_0.pkl"
    ]
    
    models_OV = []
    for path in model_paths_OV:
        with open(path, "rb") as f:
            models_OV.append(pickle.load(f))
            
    combinedModels_OV = kabuki.utils.concat_models(models_OV)
    
    # summary stats for relevant nodes:
    stats_summary_OV = combinedModels_OV.gen_stats()
    print(stats_summary_OV[stats_summary_OV.index.isin([
        'a',
        't(low)',
        't(medium)',
        't(high)',
        'v_Intercept',
        'v_AttentionW',
        'v_InattentionW:C(OVcate)[low]',
        'v_InattentionW:C(OVcate)[medium]',
        'v_InattentionW:C(OVcate)[high]'
    ])])
    print("DIC (OV):", combinedModels_OV.dic)            # some diagnostics
    print("BPIC (OV):", combinedModels_OV.mc.BPIC)
    
    # nodes for OV:
    a_OV     = combinedModels_OV.nodes_db.node['a']
    t_low    = combinedModels_OV.nodes_db.node['t(low)']
    t_med    = combinedModels_OV.nodes_db.node['t(medium)']
    t_high   = combinedModels_OV.nodes_db.node['t(high)']
    inter_OV = combinedModels_OV.nodes_db.node['v_Intercept']
    vA_OV    = combinedModels_OV.nodes_db.node['v_AttentionW']
    vIA_low  = combinedModels_OV.nodes_db.node['v_InattentionW:C(OVcate)[low]']
    vIA_med  = combinedModels_OV.nodes_db.node['v_InattentionW:C(OVcate)[medium]']
    vIA_high = combinedModels_OV.nodes_db.node['v_InattentionW:C(OVcate)[high]']
    
    # Group-level Table for OV (theta = b2 / b1 per OV level)
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()

    group_params_OV = {
        "a": a_OV.trace(),
        "t(low)": t_low.trace(),
        "t(medium)": t_med.trace(),
        "t(high)": t_high.trace(),
        "v_Intercept": inter_OV.trace(),
        "v_AttentionW": vA_OV.trace(),
        "v_InattentionW:C(OVcate)[low]": vIA_low.trace(),
        "v_InattentionW:C(OVcate)[medium]": vIA_med.trace(),
        "v_InattentionW:C(OVcate)[high]": vIA_high.trace(),
        "theta(low)": theta_low,
        "theta(medium)": theta_med,
        "theta(high)": theta_high
    }

    group_results_OV = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params_OV.items():
        group_results_OV["Parameter"].append(name)
        group_results_OV["MAP"].append(trace.mean())
        group_results_OV["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results_OV["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group_OV = pd.DataFrame(group_results_OV)
    df_group_OV.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_EE_5/group_level_MAP_table_EE_garcia_m5.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group_OV)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_InattentionW / v_AttentionW
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()
    
    rows_OV = []
    # (group-level from t)  this obviously depends on which model you are running (4 = a varies by OV, 5 = t varies by OV - code below would need to be adjusted)
    rows_OV.append({
        "Parameter": "a",
        "Group-level": format_estimate(a_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # a differences across OVcate:
    rows_OV.append({
        "Parameter": "t",
        "Group-level": "",
        "Med-Low": format_diff(t_med.trace() - t_low.trace()),
        "High-Low": format_diff(t_high.trace() - t_low.trace()),
        "High-Med": format_diff(t_high.trace() - t_med.trace())
    })
    # θ differences across OVcate:
    rows_OV.append({
        "Parameter": "θ",
        "Group-level": "",
        "Med-Low": format_diff(theta_med - theta_low),
        "High-Low": format_diff(theta_high - theta_low),
        "High-Med": format_diff(theta_high - theta_med)
    })
    # b0 and b1 (group-level):
    rows_OV.append({
        "Parameter": "b0",
        "Group-level": format_estimate(inter_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    rows_OV.append({
        "Parameter": "b1",
        "Group-level": format_estimate(vA_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # b2: differences in v_InattentionW across OVcate:
    rows_OV.append({
        "Parameter": "b2",
        "Group-level": "",
        "Med-Low": format_diff(vIA_med.trace() - vIA_low.trace()),
        "High-Low": format_diff(vIA_high.trace() - vIA_low.trace()),
        "High-Med": format_diff(vIA_high.trace() - vIA_med.trace())
    })
    
    df_combined_OV = pd.DataFrame(rows_OV, columns=["Parameter", "Group-level", "Med-Low", "High-Low", "High-Med"])
    df_combined_OV.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_EE_5/combined_parameter_comparison_table_EE_garcia_m5.csv", index=False)
    print("OV Combined Parameter Comparison Table:")
    print(df_combined_OV)
    
    #---------------------------------------------------------------------------------------------------------------
    #combine pkls for model 1 (baseline model)
    
    model_paths = [
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_1_4.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_1_3.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_1_2.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_1_1.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_1_0.pkl"
    ]
    
    models = []
    for path in model_paths:
        with open(path, "rb") as f:
            models.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models)
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        'a',
        't',
        'v_Intercept', 
        'v_AttentionW',
        'v_InattentionW',
    ])])
    print("DIC:", combinedModels.dic)
    print("BPIC:", combinedModels.mc.BPIC)
    
    # nodes for the baseline model
    a     = combinedModels.nodes_db.node['a']
    t     = combinedModels.nodes_db.node['t']
    inter = combinedModels.nodes_db.node['v_Intercept']
    vA    = combinedModels.nodes_db.node['v_AttentionW']
    vIA   = combinedModels.nodes_db.node['v_InattentionW']
    
    # group-level table
    mapping = {
        "a": "a",
        "t": "ndt",
        "v_Intercept": "b0",
        "v_AttentionW": "b1",
        "v_InattentionW": "b2"
    }
    
    group_params = {
        mapping["a"]: a.trace(),
        mapping["t"]: t.trace(),
        mapping["v_Intercept"]: inter.trace(),
        mapping["v_AttentionW"]: vA.trace(),
        mapping["v_InattentionW"]: vIA.trace()
    }
    
    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    
    def compute_stats(trace):
        return trace.mean(), stats.mstats.mquantiles(trace, [0.025])[0], stats.mstats.mquantiles(trace, [0.975])[0]
    
    for param, trace in group_params.items():
        map_val, lower, upper = compute_stats(trace)
        group_results["Parameter"].append(param)
        group_results["MAP"].append(map_val)
        group_results["HDI_lower"].append(lower)
        group_results["HDI_upper"].append(upper)
    
    theta_trace = vIA.trace() / vA.trace()
    map_val, lower, upper = compute_stats(theta_trace)
    group_results["Parameter"].append("theta")
    group_results["MAP"].append(map_val)
    group_results["HDI_lower"].append(lower)
    group_results["HDI_upper"].append(upper)
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_EE_1/group_level_MAP_table_EE_garcia_m1.csv", index=False)
    print("Group-level parameter estimates:")
    print(df_group)

def run_version_1_b():
    #---------------------------------------------------------------------------------------------------------------
    # Version 1: OV-modulated models (high, medium, low)
    # load and combine OV model files (set which model)
    model_paths_OV_4 = [
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_4_4.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_4_3.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_4_2.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_4_1.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_EE_4_0.pkl"
    ]
    
    models_OV_4 = []
    for path in model_paths_OV_4:
        with open(path, "rb") as f:
            models_OV_4.append(pickle.load(f))
            
    combinedModels_OV_4 = kabuki.utils.concat_models(models_OV_4)
    
    # summary stats for relevant nodes:
    stats_summary_OV_4 = combinedModels_OV_4.gen_stats()
    print(stats_summary_OV_4[stats_summary_OV_4.index.isin([
        't',
        'a(low)',
        'a(medium)',
        'a(high)',
        'v_Intercept',
        'v_AttentionW',
        'v_InattentionW:C(OVcate)[low]',
        'v_InattentionW:C(OVcate)[medium]',
        'v_InattentionW:C(OVcate)[high]'
    ])])
    print("DIC (OV):", combinedModels_OV_4.dic)            # some diagnostics
    print("BPIC (OV):", combinedModels_OV_4.mc.BPIC)
    
    # nodes for OV:
    t_OV     = combinedModels_OV_4.nodes_db.node['t']
    a_low    = combinedModels_OV_4.nodes_db.node['a(low)']
    a_med    = combinedModels_OV_4.nodes_db.node['a(medium)']
    a_high   = combinedModels_OV_4.nodes_db.node['a(high)']
    inter_OV = combinedModels_OV_4.nodes_db.node['v_Intercept']
    vA_OV    = combinedModels_OV_4.nodes_db.node['v_AttentionW']
    vIA_low  = combinedModels_OV_4.nodes_db.node['v_InattentionW:C(OVcate)[low]']
    vIA_med  = combinedModels_OV_4.nodes_db.node['v_InattentionW:C(OVcate)[medium]']
    vIA_high = combinedModels_OV_4.nodes_db.node['v_InattentionW:C(OVcate)[high]']
    
    # Group-level Table for OV (theta = b2 / b1 per OV level)
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()

    group_params_OV_4 = {
        "t": t_OV.trace(),
        "a(low)": a_low.trace(),
        "a(medium)": a_med.trace(),
        "a(high)": a_high.trace(),
        "v_Intercept": inter_OV.trace(),
        "v_AttentionW": vA_OV.trace(),
        "v_InattentionW:C(OVcate)[low]": vIA_low.trace(),
        "v_InattentionW:C(OVcate)[medium]": vIA_med.trace(),
        "v_InattentionW:C(OVcate)[high]": vIA_high.trace(),
        "theta(low)": theta_low,
        "theta(medium)": theta_med,
        "theta(high)": theta_high
    }

    group_results_OV_4 = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params_OV_4.items():
        group_results_OV_4["Parameter"].append(name)
        group_results_OV_4["MAP"].append(trace.mean())
        group_results_OV_4["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results_OV_4["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group_OV_4 = pd.DataFrame(group_results_OV_4)
    df_group_OV_4.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_EE_4/group_level_MAP_table_garcia_EE_m4.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group_OV_4)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_InattentionW / v_AttentionW
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()
    
    rows_OV_4 = []
    # (group-level from t)  this obviously depends on which model you are running (4 = a varies by OV, 5 = t varies by OV - code below would need to be adjusted)
    rows_OV_4.append({
        "Parameter": "t",
        "Group-level": format_estimate(t_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # a differences across OVcate:
    rows_OV_4.append({
        "Parameter": "a",
        "Group-level": "",
        "Med-Low": format_diff(a_med.trace() - a_low.trace()),
        "High-Low": format_diff(a_high.trace() - a_low.trace()),
        "High-Med": format_diff(a_high.trace() - a_med.trace())
    })
    # θ differences across OVcate:
    rows_OV_4.append({
        "Parameter": "θ",
        "Group-level": "",
        "Med-Low": format_diff(theta_med - theta_low),
        "High-Low": format_diff(theta_high - theta_low),
        "High-Med": format_diff(theta_high - theta_med)
    })
    # b0 and b1 (group-level):
    rows_OV_4.append({
        "Parameter": "b0",
        "Group-level": format_estimate(inter_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    rows_OV_4.append({
        "Parameter": "b1",
        "Group-level": format_estimate(vA_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # b2: differences in v_InattentionW across OVcate:
    rows_OV_4.append({
        "Parameter": "b2",
        "Group-level": "",
        "Med-Low": format_diff(vIA_med.trace() - vIA_low.trace()),
        "High-Low": format_diff(vIA_high.trace() - vIA_low.trace()),
        "High-Med": format_diff(vIA_high.trace() - vIA_med.trace())
    })
    
    df_combined_OV_4 = pd.DataFrame(rows_OV_4, columns=["Parameter", "Group-level", "Med-Low", "High-Low", "High-Med"])
    df_combined_OV_4.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_EE_4/combined_parameter_comparison_table_garcia_EE_m4.csv", index=False)
    print("OV Combined Parameter Comparison Table:")
    print(df_combined_OV_4)


###########################################################################################################################################################################
###########################################################################################################################################################################

def run_version_1_c():
    #---------------------------------------------------------------------------------------------------------------
    # Version 1: OV-modulated models (high, medium, low)
    # load and combine OV model files (set which model)
    model_paths_OV = [
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_5_4.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_5_3.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_5_2.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_5_1.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_5_0.pkl"
    ]
    
    models_OV = []
    for path in model_paths_OV:
        with open(path, "rb") as f:
            models_OV.append(pickle.load(f))
            
    combinedModels_OV = kabuki.utils.concat_models(models_OV)
    
    # summary stats for relevant nodes:
    stats_summary_OV = combinedModels_OV.gen_stats()
    print(stats_summary_OV[stats_summary_OV.index.isin([
        'a',
        't(low)',
        't(medium)',
        't(high)',
        'v_Intercept',
        'v_AttentionW',
        'v_InattentionW:C(OVcate)[low]',
        'v_InattentionW:C(OVcate)[medium]',
        'v_InattentionW:C(OVcate)[high]'
    ])])
    print("DIC (OV):", combinedModels_OV.dic)            # some diagnostics
    print("BPIC (OV):", combinedModels_OV.mc.BPIC)
    
    # nodes for OV:
    a_OV     = combinedModels_OV.nodes_db.node['a']
    t_low    = combinedModels_OV.nodes_db.node['t(low)']
    t_med    = combinedModels_OV.nodes_db.node['t(medium)']
    t_high   = combinedModels_OV.nodes_db.node['t(high)']
    inter_OV = combinedModels_OV.nodes_db.node['v_Intercept']
    vA_OV    = combinedModels_OV.nodes_db.node['v_AttentionW']
    vIA_low  = combinedModels_OV.nodes_db.node['v_InattentionW:C(OVcate)[low]']
    vIA_med  = combinedModels_OV.nodes_db.node['v_InattentionW:C(OVcate)[medium]']
    vIA_high = combinedModels_OV.nodes_db.node['v_InattentionW:C(OVcate)[high]']
    
    # Group-level Table for OV (theta = b2 / b1 per OV level)
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()

    group_params_OV = {
        "a": a_OV.trace(),
        "t(low)": t_low.trace(),
        "t(medium)": t_med.trace(),
        "t(high)": t_high.trace(),
        "v_Intercept": inter_OV.trace(),
        "v_AttentionW": vA_OV.trace(),
        "v_InattentionW:C(OVcate)[low]": vIA_low.trace(),
        "v_InattentionW:C(OVcate)[medium]": vIA_med.trace(),
        "v_InattentionW:C(OVcate)[high]": vIA_high.trace(),
        "theta(low)": theta_low,
        "theta(medium)": theta_med,
        "theta(high)": theta_high
    }

    group_results_OV = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params_OV.items():
        group_results_OV["Parameter"].append(name)
        group_results_OV["MAP"].append(trace.mean())
        group_results_OV["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results_OV["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group_OV = pd.DataFrame(group_results_OV)
    df_group_OV.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ES_5/group_level_MAP_table_ES_garcia_m5.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group_OV)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_InattentionW / v_AttentionW
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()
    
    rows_OV = []
    # (group-level from t)  this obviously depends on which model you are running (4 = a varies by OV, 5 = t varies by OV - code below would need to be adjusted)
    rows_OV.append({
        "Parameter": "a",
        "Group-level": format_estimate(a_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # a differences across OVcate:
    rows_OV.append({
        "Parameter": "t",
        "Group-level": "",
        "Med-Low": format_diff(t_med.trace() - t_low.trace()),
        "High-Low": format_diff(t_high.trace() - t_low.trace()),
        "High-Med": format_diff(t_high.trace() - t_med.trace())
    })
    # θ differences across OVcate:
    rows_OV.append({
        "Parameter": "θ",
        "Group-level": "",
        "Med-Low": format_diff(theta_med - theta_low),
        "High-Low": format_diff(theta_high - theta_low),
        "High-Med": format_diff(theta_high - theta_med)
    })
    # b0 and b1 (group-level):
    rows_OV.append({
        "Parameter": "b0",
        "Group-level": format_estimate(inter_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    rows_OV.append({
        "Parameter": "b1",
        "Group-level": format_estimate(vA_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # b2: differences in v_InattentionW across OVcate:
    rows_OV.append({
        "Parameter": "b2",
        "Group-level": "",
        "Med-Low": format_diff(vIA_med.trace() - vIA_low.trace()),
        "High-Low": format_diff(vIA_high.trace() - vIA_low.trace()),
        "High-Med": format_diff(vIA_high.trace() - vIA_med.trace())
    })
    
    df_combined_OV = pd.DataFrame(rows_OV, columns=["Parameter", "Group-level", "Med-Low", "High-Low", "High-Med"])
    df_combined_OV.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ES_5/combined_parameter_comparison_table_ES_garcia_m5.csv", index=False)
    print("OV Combined Parameter Comparison Table:")
    print(df_combined_OV)
    
    #---------------------------------------------------------------------------------------------------------------
    #combine pkls for model 1 (baseline model)
    
    model_paths = [
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_1_4.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_1_3.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_1_2.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_1_1.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_1_0.pkl"
    ]
    
    models = []
    for path in model_paths:
        with open(path, "rb") as f:
            models.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models)
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        'a',
        't',
        'v_Intercept', 
        'v_AttentionW',
        'v_InattentionW',
    ])])
    print("DIC:", combinedModels.dic)
    print("BPIC:", combinedModels.mc.BPIC)
    
    # nodes for the baseline model
    a     = combinedModels.nodes_db.node['a']
    t     = combinedModels.nodes_db.node['t']
    inter = combinedModels.nodes_db.node['v_Intercept']
    vA    = combinedModels.nodes_db.node['v_AttentionW']
    vIA   = combinedModels.nodes_db.node['v_InattentionW']
    
    # group-level table
    mapping = {
        "a": "a",
        "t": "ndt",
        "v_Intercept": "b0",
        "v_AttentionW": "b1",
        "v_InattentionW": "b2"
    }
    
    group_params = {
        mapping["a"]: a.trace(),
        mapping["t"]: t.trace(),
        mapping["v_Intercept"]: inter.trace(),
        mapping["v_AttentionW"]: vA.trace(),
        mapping["v_InattentionW"]: vIA.trace()
    }
    
    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    
    def compute_stats(trace):
        return trace.mean(), stats.mstats.mquantiles(trace, [0.025])[0], stats.mstats.mquantiles(trace, [0.975])[0]
    
    for param, trace in group_params.items():
        map_val, lower, upper = compute_stats(trace)
        group_results["Parameter"].append(param)
        group_results["MAP"].append(map_val)
        group_results["HDI_lower"].append(lower)
        group_results["HDI_upper"].append(upper)
    
    theta_trace = vIA.trace() / vA.trace()
    map_val, lower, upper = compute_stats(theta_trace)
    group_results["Parameter"].append("theta")
    group_results["MAP"].append(map_val)
    group_results["HDI_lower"].append(lower)
    group_results["HDI_upper"].append(upper)
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ES_1/group_level_MAP_table_ES_garcia_m1.csv", index=False)
    print("Group-level parameter estimates:")
    print(df_group)

def run_version_1_d():
    #---------------------------------------------------------------------------------------------------------------
    # Version 1: OV-modulated models (high, medium, low)
    # load and combine OV model files (set which model)
    model_paths_OV_4 = [
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_4_4.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_4_3.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_4_2.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_4_1.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ES_4_0.pkl"
    ]
    
    models_OV_4 = []
    for path in model_paths_OV_4:
        with open(path, "rb") as f:
            models_OV_4.append(pickle.load(f))
            
    combinedModels_OV_4 = kabuki.utils.concat_models(models_OV_4)
    
    # summary stats for relevant nodes:
    stats_summary_OV_4 = combinedModels_OV_4.gen_stats()
    print(stats_summary_OV_4[stats_summary_OV_4.index.isin([
        't',
        'a(low)',
        'a(medium)',
        'a(high)',
        'v_Intercept',
        'v_AttentionW',
        'v_InattentionW:C(OVcate)[low]',
        'v_InattentionW:C(OVcate)[medium]',
        'v_InattentionW:C(OVcate)[high]'
    ])])
    print("DIC (OV):", combinedModels_OV_4.dic)            # some diagnostics
    print("BPIC (OV):", combinedModels_OV_4.mc.BPIC)
    
    # nodes for OV:
    t_OV     = combinedModels_OV_4.nodes_db.node['t']
    a_low    = combinedModels_OV_4.nodes_db.node['a(low)']
    a_med    = combinedModels_OV_4.nodes_db.node['a(medium)']
    a_high   = combinedModels_OV_4.nodes_db.node['a(high)']
    inter_OV = combinedModels_OV_4.nodes_db.node['v_Intercept']
    vA_OV    = combinedModels_OV_4.nodes_db.node['v_AttentionW']
    vIA_low  = combinedModels_OV_4.nodes_db.node['v_InattentionW:C(OVcate)[low]']
    vIA_med  = combinedModels_OV_4.nodes_db.node['v_InattentionW:C(OVcate)[medium]']
    vIA_high = combinedModels_OV_4.nodes_db.node['v_InattentionW:C(OVcate)[high]']
    
    # Group-level Table for OV (theta = b2 / b1 per OV level)
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()

    group_params_OV_4 = {
        "t": t_OV.trace(),
        "a(low)": a_low.trace(),
        "a(medium)": a_med.trace(),
        "a(high)": a_high.trace(),
        "v_Intercept": inter_OV.trace(),
        "v_AttentionW": vA_OV.trace(),
        "v_InattentionW:C(OVcate)[low]": vIA_low.trace(),
        "v_InattentionW:C(OVcate)[medium]": vIA_med.trace(),
        "v_InattentionW:C(OVcate)[high]": vIA_high.trace(),
        "theta(low)": theta_low,
        "theta(medium)": theta_med,
        "theta(high)": theta_high
    }

    group_results_OV_4 = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params_OV_4.items():
        group_results_OV_4["Parameter"].append(name)
        group_results_OV_4["MAP"].append(trace.mean())
        group_results_OV_4["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results_OV_4["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group_OV_4 = pd.DataFrame(group_results_OV_4)
    df_group_OV_4.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ES_4/group_level_MAP_table_garcia_ES_m4.csv", index=False)
    print("group-level parameter estimates:")
    print(df_group_OV_4)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_InattentionW / v_AttentionW
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()
    
    rows_OV_4 = []
    # (group-level from t)  this obviously depends on which model you are running (4 = a varies by OV, 5 = t varies by OV - code below would need to be adjusted)
    rows_OV_4.append({
        "Parameter": "t",
        "Group-level": format_estimate(t_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # a differences across OVcate:
    rows_OV_4.append({
        "Parameter": "a",
        "Group-level": "",
        "Med-Low": format_diff(a_med.trace() - a_low.trace()),
        "High-Low": format_diff(a_high.trace() - a_low.trace()),
        "High-Med": format_diff(a_high.trace() - a_med.trace())
    })
    # θ differences across OVcate:
    rows_OV_4.append({
        "Parameter": "θ",
        "Group-level": "",
        "Med-Low": format_diff(theta_med - theta_low),
        "High-Low": format_diff(theta_high - theta_low),
        "High-Med": format_diff(theta_high - theta_med)
    })
    # b0 and b1 (group-level):
    rows_OV_4.append({
        "Parameter": "b0",
        "Group-level": format_estimate(inter_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    rows_OV_4.append({
        "Parameter": "b1",
        "Group-level": format_estimate(vA_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # b2: differences in v_InattentionW across OVcate:
    rows_OV_4.append({
        "Parameter": "b2",
        "Group-level": "",
        "Med-Low": format_diff(vIA_med.trace() - vIA_low.trace()),
        "High-Low": format_diff(vIA_high.trace() - vIA_low.trace()),
        "High-Med": format_diff(vIA_high.trace() - vIA_med.trace())
    })
    
    df_combined_OV_4 = pd.DataFrame(rows_OV_4, columns=["Parameter", "Group-level", "Med-Low", "High-Low", "High-Med"])
    df_combined_OV_4.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ES_4/combined_parameter_comparison_table_garcia_ES_m4.csv", index=False)
    print("OV Combined Parameter Comparison Table:")
    print(df_combined_OV_4)


    
###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################
# ESEE 

def run_version_2_a():
    # -------------------------------------------------------------------------
    # Version 2: Phase differences ES EE
    import pandas as pd
    import pickle
    import kabuki
    import scipy.stats as stats
    
    # MODEL 5 or 4 (best fitting ones) , ESEE (Phase-specific model: ES and EE)
    model_paths_m4 = [
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_4_4.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_4_3.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_4_2.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_4_1.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_4_0.pkl"

    ]
    
    models_m4 = []
    for path in model_paths_m4:
        with open(path, "rb") as f:
            models_m4.append(pickle.load(f))
            
    combinedModels_m4 = kabuki.utils.concat_models(models_m4)
    
    stats_summary_m4 = combinedModels_m4.gen_stats()
    print(stats_summary_m4[stats_summary_m4.index.isin([
        't',
        'a(ES)',
        'a(EE)',
        'v_Intercept',
        'v_AttentionW',
        'v_InattentionW:C(phase)[ES]',
        'v_InattentionW:C(phase)[EE]',
    ])])
    print("DIC (Model 4):", combinedModels_m4.dic)
    print("BPIC (Model 4):", combinedModels_m4.mc.BPIC)
    
    # get nodes (only ES and EE)
    t_m4     = combinedModels_m4.nodes_db.node['t']
    a_ES     = combinedModels_m4.nodes_db.node['a(ES)']
    a_EE     = combinedModels_m4.nodes_db.node['a(EE)']
    inter_m4 = combinedModels_m4.nodes_db.node['v_Intercept']
    vA_m4    = combinedModels_m4.nodes_db.node['v_AttentionW']
    vIA_ES   = combinedModels_m4.nodes_db.node['v_InattentionW:C(phase)[ES]']
    vIA_EE   = combinedModels_m4.nodes_db.node['v_InattentionW:C(phase)[EE]']
    
    
    # Group-level Table for OV (including theta = b2 / b1 per OV level)
    theta_ES  = vIA_ES.trace() / vA_m4.trace()
    theta_EE  = vIA_EE.trace() / vA_m4.trace()

    # Group-level Table for Model 5 or 4
    group_params_m4 = {
        "t": t_m4.trace(),
        "a(ES)": a_ES.trace(),
        "a(EE)": a_EE.trace(),
        "v_Intercept": inter_m4.trace(),
        "v_AttentionW": vA_m4.trace(),
        "v_InattentionW:C(phase)[ES]": vIA_ES.trace(),
        "v_InattentionW:C(phase)[EE]": vIA_EE.trace(),
        "theta(ES)": theta_ES,
        "theta(EE)": theta_EE,
    }
    
    group_results_m4 = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params_m4.items():
        group_results_m4["Parameter"].append(name)
        group_results_m4["MAP"].append(trace.mean())
        group_results_m4["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results_m4["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group_m4 = pd.DataFrame(group_results_m4)
    df_group_m4.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ESEE_4/group_level_MAP_table_garcia_ESEE_m4.csv", index=False)
    print("Model 4 Group-level parameter estimates (ESEE):")
    print(df_group_m4)
    
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # theta for ES and EE
    theta_ES = vIA_ES.trace() / vA_m4.trace()
    theta_EE = vIA_EE.trace() / vA_m4.trace()
    
    rows_m4 = []
    rows_m4.append({
        "Parameter": "t",
        "Group-level": format_estimate(t_m4.trace()),
        "ES-EE": ""
    })
    # a: difference between ES and EE
    rows_m4.append({
        "Parameter": "a",
        "Group-level": "",
        "ES-EE": format_diff(a_ES.trace() - a_EE.trace())
    })
    # θ difference
    rows_m4.append({
        "Parameter": "θ",
        "Group-level": "",
        "ES-EE": format_diff(theta_ES - theta_EE)
    })
    rows_m4.append({
        "Parameter": "b0",
        "Group-level": format_estimate(inter_m4.trace()),
        "ES-EE": ""
    })
    rows_m4.append({
        "Parameter": "b1",
        "Group-level": format_estimate(vA_m4.trace()),
        "ES-EE": ""
    })
    rows_m4.append({
        "Parameter": "b2",
        "Group-level": "",
        "ES-EE": format_diff(vIA_ES.trace() - vIA_EE.trace())
    })
    
    df_combined_m4 = pd.DataFrame(rows_m4, columns=["Parameter", "Group-level", "ES-EE"])
    df_combined_m4.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ESEE_4/combined_parameter_comparison_table_garcia_ESEE_m4.csv", index=False)
    print("Model 4 Combined Parameter Comparison Table (ESEE):")
    print(df_combined_m4)
    
    # MODEL 1, ESEE (baaseline model)
    model_paths_m1 = [
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_1_4.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_1_3.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_1_2.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_1_1.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_1_0.pkl",

    ]
    
    models_m1 = []
    for path in model_paths_m1:
        with open(path, "rb") as f:
            models_m1.append(pickle.load(f))
            
    combinedModels_m1 = kabuki.utils.concat_models(models_m1)
    stats_summary_m1 = combinedModels_m1.gen_stats()
    print(stats_summary_m1[stats_summary_m1.index.isin([
        'a',
        't',
        'v_Intercept',
        'v_AttentionW',
        'v_InattentionW',
    ])])
    print("DIC (Model 1):", combinedModels_m1.dic)
    print("BPIC (Model 1):", combinedModels_m1.mc.BPIC)
    
    a_m1     = combinedModels_m1.nodes_db.node['a']
    t_m1     = combinedModels_m1.nodes_db.node['t']
    inter_m1 = combinedModels_m1.nodes_db.node['v_Intercept']
    vA_m1    = combinedModels_m1.nodes_db.node['v_AttentionW']
    vIA_m1   = combinedModels_m1.nodes_db.node['v_InattentionW']
    
    mapping = {
        "a": "a",
        "t": "ndt",
        "v_Intercept": "b0",
        "v_AttentionW": "b1",
        "v_InattentionW": "b2"
    }
    
    group_params_m1 = {
        mapping["a"]: a_m1.trace(),
        mapping["t"]: t_m1.trace(),
        mapping["v_Intercept"]: inter_m1.trace(),
        mapping["v_AttentionW"]: vA_m1.trace(),
        mapping["v_InattentionW"]: vIA_m1.trace()
    }
    
    group_results_m1 = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    
    def compute_stats(trace):
        return trace.mean(), stats.mstats.mquantiles(trace, [0.025])[0], stats.mstats.mquantiles(trace, [0.975])[0]
    
    for param, trace in group_params_m1.items():
        map_val, lower, upper = compute_stats(trace)
        group_results_m1["Parameter"].append(param)
        group_results_m1["MAP"].append(map_val)
        group_results_m1["HDI_lower"].append(lower)
        group_results_m1["HDI_upper"].append(upper)
    
    theta_trace_m1 = vIA_m1.trace() / vA_m1.trace()
    map_val, lower, upper = compute_stats(theta_trace_m1)
    group_results_m1["Parameter"].append("theta")
    group_results_m1["MAP"].append(map_val)
    group_results_m1["HDI_lower"].append(lower)
    group_results_m1["HDI_upper"].append(upper)
    
    df_group_m1 = pd.DataFrame(group_results_m1)
    df_group_m1.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ESEE_1/group_level_MAP_table_garcia_ESEE_m1.csv", index=False)
    print("Model 1 Group-level parameter estimates (ESEE):")
    print(df_group_m1)

def run_version_2_b():
    # -------------------------------------------------------------------------
    # Version 2: Phase differences ES EE
    
    import pandas as pd
    import pickle
    import kabuki
    import scipy.stats as stats
    
    # MODEL 5 or 4 (best fitting ones) , ESEE (Phase-specific model: ES and EE)
    model_paths_m5 = [
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_5_4.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_5_3.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_5_2.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_5_1.pkl",
        "/home/jovyan/OfficialTutorials/For_Linux/models_dir_garcia/garcia_replication_ESEE_5_0.pkl",
    ]
    
    models_m5 = []
    for path in model_paths_m5:
        with open(path, "rb") as f:
            models_m5.append(pickle.load(f))
            
    combinedModels_m5 = kabuki.utils.concat_models(models_m5)
    
    stats_summary_m5 = combinedModels_m5.gen_stats()
    print(stats_summary_m5[stats_summary_m5.index.isin([
        'a',
        't(ES)',
        't(EE)',
        'v_Intercept',
        'v_AttentionW',
        'v_InattentionW:C(phase)[ES]',
        'v_InattentionW:C(phase)[EE]',
    ])])
    print("DIC (Model 5):", combinedModels_m5.dic)
    print("BPIC (Model 5):", combinedModels_m5.mc.BPIC)
    
    # get nodes (only ES and EE)
    a_m5     = combinedModels_m5.nodes_db.node['a']
    t_ES     = combinedModels_m5.nodes_db.node['t(ES)']
    t_EE     = combinedModels_m5.nodes_db.node['t(EE)']
    inter_m5 = combinedModels_m5.nodes_db.node['v_Intercept']
    vA_m5    = combinedModels_m5.nodes_db.node['v_AttentionW']
    vIA_ES   = combinedModels_m5.nodes_db.node['v_InattentionW:C(phase)[ES]']
    vIA_EE   = combinedModels_m5.nodes_db.node['v_InattentionW:C(phase)[EE]']
    
    
    # Group-level Table for OV (including theta = b2 / b1 per OV level)
    theta_ES  = vIA_ES.trace() / vA_m5.trace()
    theta_EE  = vIA_EE.trace() / vA_m5.trace()

    # Group-level Table for Model 5 or 4
    group_params_m5 = {
        "a": a_m5.trace(),
        "t(ES)": t_ES.trace(),
        "t(EE)": t_EE.trace(),
        "v_Intercept": inter_m5.trace(),
        "v_AttentionW": vA_m5.trace(),
        "v_InattentionW:C(phase)[ES]": vIA_ES.trace(),
        "v_InattentionW:C(phase)[EE]": vIA_EE.trace(),
        "theta(ES)": theta_ES,
        "theta(EE)": theta_EE,
    }
    
    group_results_m5 = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params_m5.items():
        group_results_m5["Parameter"].append(name)
        group_results_m5["MAP"].append(trace.mean())
        group_results_m5["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results_m5["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group_m5 = pd.DataFrame(group_results_m5)
    df_group_m5.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ESEE_5/group_level_MAP_table_garcia_ESEE_m5.csv", index=False)
    print("Model 5 Group-level parameter estimates (ESEE):")
    print(df_group_m5)
    
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # theta for ES and EE
    theta_ES = vIA_ES.trace() / vA_m5.trace()
    theta_EE = vIA_EE.trace() / vA_m5.trace()
    
    rows_m5 = []
    rows_m5.append({
        "Parameter": "a",
        "Group-level": format_estimate(a_m5.trace()),
        "ES-EE": ""
    })
    # a: difference between ES and EE
    rows_m5.append({
        "Parameter": "t",
        "Group-level": "",
        "ES-EE": format_diff(t_ES.trace() - t_EE.trace())
    })
    # θ difference
    rows_m5.append({
        "Parameter": "θ",
        "Group-level": "",
        "ES-EE": format_diff(theta_ES - theta_EE)
    })
    rows_m5.append({
        "Parameter": "b0",
        "Group-level": format_estimate(inter_m5.trace()),
        "ES-EE": ""
    })
    rows_m5.append({
        "Parameter": "b1",
        "Group-level": format_estimate(vA_m5.trace()),
        "ES-EE": ""
    })
    rows_m5.append({
        "Parameter": "b2",
        "Group-level": "",
        "ES-EE": format_diff(vIA_ES.trace() - vIA_EE.trace())
    })
    
    df_combined_m5 = pd.DataFrame(rows_m5, columns=["Parameter", "Group-level", "ES-EE"])
    df_combined_m5.to_csv("/home/jovyan/OfficialTutorials/For_Linux/figures_dir_garcia/garcia_replication_ESEE_5/combined_parameter_comparison_table_garcia_ESEE_m5.csv", index=False)
    print("Model 5 Combined Parameter Comparison Table (ESEE):")
    print(df_combined_m5)
    

import os
import os

# inside the container /workspace == /home/u04vw21/sharedscratch/HDDM_Vero on the host
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/workspace")

def run_version_14():
    #---------------------------------------------------------------------------------------------------------------
    # Version 1: OV-modulated models (high, medium, low)
    # load and combine OV model files (set which model)
    MODELS_DIR = os.path.join(PROJECT_DIR, "models_dir_garcia")
    model_paths_OV = [
        os.path.join(MODELS_DIR, "garcia_replication_ES_14_2.pkl"),
        os.path.join(MODELS_DIR, "garcia_replication_ES_14_1.pkl"),
        os.path.join(MODELS_DIR, "garcia_replication_ES_14_0.pkl"),
        ]

    models_OV = []
    for path in model_paths_OV:
        with open(path, "rb") as f:
            models_OV.append(pickle.load(f))
            
    combinedModels_OV = kabuki.utils.concat_models(models_OV)
    
    # summary stats for relevant nodes:
    stats_summary_OV = combinedModels_OV.gen_stats()
    print(stats_summary_OV[stats_summary_OV.index.isin([
        'a',
        't',
        'v_Intercept',
        'v_AttentionW',
        'v_InattentionW:C(OVcate)[low]',
        'v_InattentionW:C(OVcate)[medium]',
        'v_InattentionW:C(OVcate)[high]',
        'z(0)',
        'z(1)',
    ])])
    print("DIC (OV):", combinedModels_OV.dic)            # some diagnostics
    print("BPIC (OV):", combinedModels_OV.mc.BPIC)
    
    # nodes for OV:
    a_OV     = combinedModels_OV.nodes_db.node['a']
    t_OV     = combinedModels_OV.nodes_db.node['t']
    inter_OV = combinedModels_OV.nodes_db.node['v_Intercept']
    vA_OV    = combinedModels_OV.nodes_db.node['v_AttentionW']
    vIA_low  = combinedModels_OV.nodes_db.node['v_InattentionW:C(OVcate)[low]']
    vIA_med  = combinedModels_OV.nodes_db.node['v_InattentionW:C(OVcate)[medium]']
    vIA_high = combinedModels_OV.nodes_db.node['v_InattentionW:C(OVcate)[high]']
    z0_OV    = combinedModels_OV.nodes_db.node['z(0)']
    z1_OV    = combinedModels_OV.nodes_db.node['z(1)']
    
    # Group-level Table for OV (theta = b2 / b1 per OV level)
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()

    group_params_OV = {
        "a": a_OV.trace(),
        "t": t_OV.trace(),
        "v_Intercept": inter_OV.trace(),
        "v_AttentionW": vA_OV.trace(),
        "v_InattentionW:C(OVcate)[low]": vIA_low.trace(),
        "v_InattentionW:C(OVcate)[medium]": vIA_med.trace(),
        "v_InattentionW:C(OVcate)[high]": vIA_high.trace(),
        "theta(low)": theta_low,
        "theta(medium)": theta_med,
        "theta(high)": theta_high,
        "z(0)": z0_OV.trace(),
        "z(1)": z1_OV.trace(),
    }

    group_results_OV = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params_OV.items():
        group_results_OV["Parameter"].append(name)
        group_results_OV["MAP"].append(trace.mean())
        group_results_OV["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results_OV["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group_OV = pd.DataFrame(group_results_OV)
    FIG_DIR = os.path.join(PROJECT_DIR, "figures_dir_garcia", "garcia_replication_ES_14")
    os.makedirs(FIG_DIR, exist_ok=True)
    
    df_group_OV.to_csv(
        os.path.join(FIG_DIR, "group_level_MAP_table_ES_garcia_m14.csv"),index=False)
    print("group-level parameter estimates:")
    print(df_group_OV)
    
    #Combined Parameter Comparison Table
    def format_estimate(trace):
        m = trace.mean()
        l = stats.mstats.mquantiles(trace, [0.025])[0]
        u = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    def format_diff(diff_trace):
        m = diff_trace.mean()
        l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
    # get theta for each category: theta = v_InattentionW / v_AttentionW
    theta_low  = vIA_low.trace() / vA_OV.trace()
    theta_med  = vIA_med.trace() / vA_OV.trace()
    theta_high = vIA_high.trace() / vA_OV.trace()

    
    rows_OV = []
    # (group-level from t)  this obviously depends on which model you are running (4 = a varies by OV, 5 = t varies by OV - code below would need to be adjusted)
    rows_OV.append({
        "Parameter": "a",
        "Group-level": format_estimate(a_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # a differences across OVcate:
    rows_OV.append({
        "Parameter": "t",
        "Group-level": format_estimate(t_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # θ differences across OVcate:
    rows_OV.append({
        "Parameter": "θ",
        "Group-level": "",
        "Med-Low": format_diff(theta_med - theta_low),
        "High-Low": format_diff(theta_high - theta_low),
        "High-Med": format_diff(theta_high - theta_med)
    })
    # b0 and b1 (group-level):
    rows_OV.append({
        "Parameter": "b0",
        "Group-level": format_estimate(inter_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    rows_OV.append({
        "Parameter": "b1",
        "Group-level": format_estimate(vA_OV.trace()),
        "Med-Low": "",
        "High-Low": "",
        "High-Med": ""
    })
    # b2: differences in v_InattentionW across OVcate:
    rows_OV.append({
        "Parameter": "b2",
        "Group-level": "",
        "Med-Low": format_diff(vIA_med.trace() - vIA_low.trace()),
        "High-Low": format_diff(vIA_high.trace() - vIA_low.trace()),
        "High-Med": format_diff(vIA_high.trace() - vIA_med.trace())
    })
    
    rows_OV.append({
        "Parameter": "z",
        "Group-level": "",
        "Stim zS-zE": format_diff(z0_OV.trace() - z1_OV.trace()),
    })
    
    df_combined_OV = pd.DataFrame(rows_OV, columns=["Parameter", "Group-level", "Med-Low", "High-Low", "High-Med", "Stim zS-zE"])
    df_combined_OV.to_csv(os.path.join(FIG_DIR, "combined_parameter_comparison_table_ES_garcia_m5.csv"),index=False)
    print("OV Combined Parameter Comparison Table:")
    print(df_combined_OV)

# def run_version_35():
#     #---------------------------------------------------------------------------------------------------------------
#     # Version 1: OV-modulated models (high, medium, low)
#     # load and combine OV model files (set which model)
#     MODELS_DIR = os.path.join(PROJECT_DIR, "models_dir_garcia")
#     model_paths_OV = [
#         os.path.join(MODELS_DIR, "garcia_replication_ES_35_2.pkl"),
#         os.path.join(MODELS_DIR, "garcia_replication_ES_35_1.pkl"),
#         os.path.join(MODELS_DIR, "garcia_replication_ES_35_0.pkl"),
#         ]
 
#     models_OV = []
#     for path in model_paths_OV:
#         with open(path, "rb") as f:
#             models_OV.append(pickle.load(f))
            
#     combinedModels_OV = kabuki.utils.concat_models(models_OV)
    
#     # summary stats for relevant nodes:
#     stats_summary_OV = combinedModels_OV.gen_stats()
#     print(stats_summary_OV[stats_summary_OV.index.isin([
#         't',
#         'v_Intercept',
#         'v_z_AttentionW:C(OVcate)[low]',
#         'v_z_AttentionW:C(OVcate)[medium]',
#         'v_z_AttentionW:C(OVcate)[high]',
#         'v_z_IAW_chart',
#         'v_z_IAW_image',
#         'a_Intercept',
#         'a_OVcate[T.low]',
#         'a_OVcate[T.medium]',
#     ])])
    
#     print("DIC (OV):", combinedModels_OV.dic)            # some diagnostics
#     print("BPIC (OV):", combinedModels_OV.mc.BPIC)
    
#     # nodes for OV:
#     t_OV     = combinedModels_OV.nodes_db.node['t']
#     inter_OV = combinedModels_OV.nodes_db.node['v_Intercept']
#     vA_low  = combinedModels_OV.nodes_db.node['v_z_AttentionW:C(OVcate)[low]']
#     vA_med  = combinedModels_OV.nodes_db.node['v_z_AttentionW:C(OVcate)[medium]']
#     vA_high = combinedModels_OV.nodes_db.node['v_z_AttentionW:C(OVcate)[high]']
#     vIA_chart = combinedModels_OV.nodes_db.node['v_z_IAW_chart']
#     vIA_image = combinedModels_OV.nodes_db.node['v_z_IAW_image']
#     a_int = combinedModels_OV.nodes_db.node['a_Intercept']
#     a_OV_low    = combinedModels_OV.nodes_db.node['a_OVcate[T.low]']
#     a_OV_medium    = combinedModels_OV.nodes_db.node['a_OVcate[T.medium]']
    
#     # Group-level Table for OV (theta = b2 / b1 per OV level)
#     thetaS_low  = vIA_chart.trace() / vA_low.trace()
#     thetaS_med  = vIA_chart.trace() / vA_med.trace()
#     thetaS_high = vIA_chart.trace() / vA_high.trace()
    
#     thetaE_low  = vIA_image.trace() / vA_low.trace()
#     thetaE_med  = vIA_image.trace() / vA_med.trace()
#     thetaE_high = vIA_image.trace() / vA_high.trace()
    
    
#     group_params_OV = {        
#         "t": t_OV.trace(),
#         "v_Intercept": inter_OV.trace(),
#         "v_z_AttentionW:C(OVcate)[low]": vA_low.trace() ,
#         "v_z_AttentionW:C(OVcate)[medium]": vA_med.trace(),
#         'v_z_AttentionW:C(OVcate)[high]': vA_high.trace(),
#         'v_z_IAW_chart': vIA_chart.trace(),
#         'v_z_IAW_image': vIA_image.trace(),
#         'a_Intercept': a_int.trace(),
#         'a_OVcate[T.low]': a_OV_low.trace(),
#         'a_OVcate[T.medium]': a_OV_medium.trace(),
#     }

#     group_results_OV = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
#     for name, trace in group_params_OV.items():
#         group_results_OV["Parameter"].append(name)
#         group_results_OV["MAP"].append(trace.mean())
#         group_results_OV["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
#         group_results_OV["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
#     df_group_OV = pd.DataFrame(group_results_OV)
#     FIG_DIR = os.path.join(PROJECT_DIR, "figures_dir_garcia", "garcia_replication_ES_35")
#     os.makedirs(FIG_DIR, exist_ok=True)
    
#     df_group_OV.to_csv(
#         os.path.join(FIG_DIR, "group_level_MAP_table_ES_garcia_m35.csv"),index=False)
#     print("group-level parameter estimates:")
#     print(df_group_OV)
    
#     #Combined Parameter Comparison Table
#     def format_estimate(trace):
#         m = trace.mean()
#         l = stats.mstats.mquantiles(trace, [0.025])[0]
#         u = stats.mstats.mquantiles(trace, [0.975])[0]
#         return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
#     def format_diff(diff_trace):
#         m = diff_trace.mean()
#         l = stats.mstats.mquantiles(diff_trace, [0.025])[0]
#         u = stats.mstats.mquantiles(diff_trace, [0.975])[0]
#         return f"{m:.3f} [{l:.3f}, {u:.3f}]"
    
#     # get theta for each category: theta = v_InattentionW / v_AttentionW
#     # Group-level Table for OV (theta = b2 / b1 per OV level)
#     thetaS_low  = vIA_chart.trace() / vA_low.trace()
#     thetaS_med  = vIA_chart.trace() / vA_med.trace()
#     thetaS_high = vIA_chart.trace() / vA_high.trace()
    
#     thetaE_low  = vIA_image.trace() / vA_low.trace()
#     thetaE_med  = vIA_image.trace() / vA_med.trace()
#     thetaE_high = vIA_image.trace() / vA_high.trace()
    
#     rows_OV = []
#     # (group-level from t)  this obviously depends on which model you are running (4 = a varies by OV, 5 = t varies by OV - code below would need to be adjusted)
#     rows_OV.append({
#         "Parameter": "t",
#         "Group-level": format_estimate(t_OV.trace()),
#         "Med-Low": "",
#         "High-Low": "",
#         "High-Med": ""
#     })
#     # θ differences across OVcate:
#     rows_OV.append({
#         "Parameter": "θS",
#         "Group-level": "",
#         "Med-Low": format_diff(thetaS_med - thetaS_low),
#         "High-Low": format_diff(thetaS_high - thetaS_low),
#         "High-Med": format_diff(thetaS_high - thetaS_med)
#     })
#     rows_OV.append({
#         "Parameter": "θE",
#         "Group-level": "",
#         "Med-Low": format_diff(thetaE_med - thetaE_low),
#         "High-Low": format_diff(thetaE_high - thetaE_low),
#         "High-Med": format_diff(thetaE_high - thetaE_med)
#     })
#     # b0 and b1 (group-level):
#     rows_OV.append({
#         "Parameter": "b0",
#         "Group-level": format_estimate(inter_OV.trace()),
#         "Med-Low": "",
#         "High-Low": "",
#         "High-Med": ""
#     })
#     rows_OV.append({
#         "Parameter": "b1",
#         "Group-level": "",
#         "Med-Low": format_diff(vA_med.trace() - vA_low.trace()),
#         "High-Low": format_diff(vA_high.trace() - vA_low.trace()),
#         "High-Med": format_diff(vA_high.trace() - vA_med.trace())
#     })
#     # b2: differences in v_InattentionW across OVcate:
#     rows_OV.append({
#         "Parameter": "b2",
#         "Group-level": "",
#         "Category difference": format_diff(vIA_chart.trace() - vIA_image.trace()),
#     })
#     # a: differences in OVcate:
#     rows_OV.append({
#         "Parameter": "a",
#         "Group-level": "",
#         "Med-Low": format_diff(a_OV_medium.trace() - a_OV_low.trace()), 
#         "High-Low": format_diff(a_int.trace() - a_OV_low.trace()),
#         "High-Med": format_diff(a_int.trace() - a_OV_medium.trace())
#     })
#     df_combined_OV = pd.DataFrame(rows_OV, columns=["Parameter", "Group-level", "Med-Low", "High-Low", "High-Med", "Category difference"])
#     df_combined_OV.to_csv(os.path.join(FIG_DIR, "combined_parameter_comparison_table_ES_garcia_m35.csv"),index=False)
#     print("OV Combined Parameter Comparison Table:")
#     print(df_combined_OV)


#########################################################################################################################
#########################################################################################################################


import arviz as az
import pandas as pd
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/workspace")

def run_version_35():
    import arviz as az
    import pandas as pd
    import os
    PROJECT_DIR = os.environ.get("PROJECT_DIR", "/workspace")
    MODELS_DIR = os.path.join(PROJECT_DIR, "models_dir_garcia")    
    nc_paths = [
        os.path.join(MODELS_DIR,"garcia_replication_ES_35_2.nc"),
        os.path.join(MODELS_DIR,"garcia_replication_ES_35_1.nc"),
        os.path.join(MODELS_DIR,"garcia_replication_ES_35_0.nc"),
    ]
    idatas = [az.from_netcdf(p) for p in nc_paths]

    # 2. concatenate along a new “chain” axis
    idata = az.concat(idatas, dim="chain")

    # shortcut to pull out a flattened array of draws for any var
    def draws(varname):
        da = idata.posterior[varname]
        return da.stack(sample=["chain","draw"]).values

    # 3. extract everything you need
    t        = draws("t")
    inter    = draws("v_Intercept")
    vA_low   = draws("v_z_AttentionW:C(OVcate)[low]")
    vA_med   = draws("v_z_AttentionW:C(OVcate)[medium]")
    vA_high  = draws("v_z_AttentionW:C(OVcate)[high]")
    vIA_c    = draws("v_z_IAW_chart")
    vIA_i    = draws("v_z_IAW_image")
    a_int    = draws("a_Intercept")
    a_low    = draws("a_OVcate[T.low]")
    a_med    = draws("a_OVcate[T.medium]")

    # 4. compute the θ’s
    thetaS_low  = vIA_c / vA_low
    thetaS_med  = vIA_c / vA_med
    thetaS_high = vIA_c / vA_high

    thetaE_low  = vIA_i / vA_low
    thetaE_med  = vIA_i / vA_med
    thetaE_high = vIA_i / vA_high

    # helper for 95% HDI
    def hdi(arr):
        lo, hi = az.hdi(arr, hdi_prob=0.95)
        return lo, hi

    # 5A. build the group‐level MAP / HDI table
    group = []
    for name, arr in [
        ("t",        t),
        ("v_Intercept", inter),
        ("v_z_AttentionW:low",  vA_low),
        ("v_z_AttentionW:med",  vA_med),
        ("v_z_AttentionW:high", vA_high),
        ("v_z_IAW_chart",       vIA_c),
        ("v_z_IAW_image",       vIA_i),
        ("a_Intercept",         a_int),
        ("a_low",               a_low),
        ("a_med",               a_med),
        ("θS_low",              thetaS_low),
        ("θS_med",              thetaS_med),
        ("θS_high",             thetaS_high),
        ("θE_low",              thetaE_low),
        ("θE_med",              thetaE_med),
        ("θE_high",             thetaE_high),
    ]:
        m = arr.mean()
        lo, hi = hdi(arr)
        group.append({"Parameter": name, "MAP": m, "HDI_lower": lo, "HDI_upper": hi})

    df_group = pd.DataFrame(group)
    df_group.to_csv(
        os.path.join(MODELS_DIR, "group_level_MAP_table_ES_garcia_m35.csv"),
        index=False
        )
    print("group‐level estimates:")
    print(df_group)

    # 5B. build the paired‐difference table
    rows = []
    def diff(x, y): 
        return x.mean() - y.mean(), *hdi(x - y)

    for (label, xa, xb) in [
        ("Med−Low θS", thetaS_med, thetaS_low),
        ("High−Low θS", thetaS_high, thetaS_low),
        ("High−Med θS", thetaS_high, thetaS_med),

        ("Med−Low θE", thetaE_med, thetaE_low),
        ("High−Low θE", thetaE_high, thetaE_low),
        ("High−Med θE", thetaE_high, thetaE_med),

        ("Chart−Image b2", vIA_c, vIA_i),

        ("Med−Low a", a_med, a_low),
        ("High−Low a", a_int, a_low),
        ("High−Med a", a_int, a_med),
    ]:
        m, lo, hi = diff(xa, xb)
        rows.append({"Comparison": label, "MeanDiff": m, "HDI_lower": lo, "HDI_upper": hi})

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(
        os.path.join(MODELS_DIR, "combined_parameter_comparison_table_ES_garcia_m35.csv"),
        index=False
        )
    
    
    
########################################################################################################################
import arviz as az
import pandas as pd
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/workspace")

def run_version_36():
    import arviz as az
    import pandas as pd
    import os
    import numpy as np

    PROJECT_DIR = os.environ.get("PROJECT_DIR", "/workspace")
    MODELS_DIR  = os.path.join(PROJECT_DIR, "models_dir_OV")

    # ---------- Helpers ----------
    def load_idata(model_stem: str, chain_idxs=(0,1,2)):
        paths = [os.path.join(MODELS_DIR, f"{model_stem}_{i}.nc") for i in chain_idxs]
        idatas = [az.from_netcdf(p) for p in paths]
        return az.concat(idatas, dim="chain")

    def short_label(stem: str) -> str:
        toks = stem.split("_")
        return toks[-1] if toks else stem

    def compare_group(group_name: str, model_stems: list, chain_idxs=(0,1,2)):
        """Run LOO, stacking, pointwise LOO, and Pareto-k for a set of models"""
        # Load models
        model_idatas = {}
        for stem in model_stems:
            try:
                idata = load_idata(stem, chain_idxs=chain_idxs)
                label = short_label(stem)
                # Ensure uniqueness if same label appears
                base, k = label, 1
                while label in model_idatas:
                    k += 1
                    label = f"{base}#{k}"
                model_idatas[label] = idata
            except Exception as e:
                print(f"[WARN] Skipping '{stem}' due to load error: {e}")
    
        if len(model_idatas) < 2:
            print(f"[WARN] Group '{group_name}': need ≥2 models. Got {len(model_idatas)}. Aborting.")
            return

        # LOO compare
        cmp_loo = az.compare(model_idatas, method="BB-pseudo-BMA", ic="loo")
        print(f"\nLOO compare (group: {group_name}):")
        print(cmp_loo)
        cmp_loo_df = cmp_loo.reset_index().rename(columns={"index": "model"})
        cmp_loo_df.to_csv(os.path.join(MODELS_DIR, f"{group_name}_LOO.csv"), index=False)
    
        # Stacking weights
        cmp_stack = az.compare(model_idatas, method="stacking", ic="loo")
        print(f"\nStacking weights (group: {group_name}):")
        print(cmp_stack)
        cmp_stack_df = cmp_stack.reset_index().rename(columns={"index": "model"})
        cmp_stack_df.to_csv(os.path.join(MODELS_DIR, f"{group_name}_stacking.csv"), index=False)
    
        # Pointwise LOO + Pareto-k reliability
        loo_summ_rows, pareto_rows = [], []
        for label, idata in model_idatas.items():
            loo_pw = az.loo(idata, pointwise=True)
            loo_summ_rows.append({
                "model": label,
                "elpd_loo": float(loo_pw.elpd_loo),
                "p_loo":    float(loo_pw.p_loo),
            })
            pk = loo_pw.pareto_k.values
            pareto_rows.append({
                "model": label,
                "frac_k>0.7": float(np.mean(pk > 0.7)),
                "frac_k>1.0": float(np.mean(pk > 1.0)),
            })
    
        pd.DataFrame(loo_summ_rows).to_csv(
            os.path.join(MODELS_DIR, f"{group_name}_LOO_summary.csv"), index=False
        )
        pd.DataFrame(pareto_rows).to_csv(
            os.path.join(MODELS_DIR, f"{group_name}_pareto_k.csv"), index=False
        )
        print(f"\nPareto-k reliability (group: {group_name}):")
        print(pd.DataFrame(pareto_rows))
    
        # Quick console rule-of-thumb for top 2
        se_col = "elpd_diff_se" if "elpd_diff_se" in cmp_loo_df.columns else ("dse" if "dse" in cmp_loo_df.columns else None)
        if len(cmp_loo_df) >= 2:
            best, runner = cmp_loo_df.iloc[0], cmp_loo_df.iloc[1]
            if se_col is not None:
                se_val = runner[se_col]
                ratio = abs(runner["elpd_diff"]) / se_val if se_val != 0 else np.inf
                print(f"\nBest by LOO (group {group_name}): {best['model']} (elpd_diff=0).")
                print(f"Runner-up: {runner['model']}, elpd_diff={runner['elpd_diff']:.2f}, "
                      f"SE={se_val:.2f} → |elpd_diff|/SE = {ratio:.2f}")
            else:
                print(f"\nBest by LOO (group {group_name}): {best['model']} (elpd_diff=0). "
                      f"Runner-up: {runner['model']}, elpd_diff={runner['elpd_diff']:.2f} (SE column not found)")
    
    # ---------- Define your groups ----------
    # Group 1 (first 4)
    # group1_stems = [
    #     "garcia_replication_ES_VAL_16",  # pure DDM (z not included)
    #     "garcia_replication_ES_VAL_12",  # DDM + SP (z free)
    #     "garcia_replication_ES_VAL_17",  # pure aDDM (z not included)
    #     "garcia_replication_ES_VAL_7",   # aDDM + SP (z included)
    # ]
    
    #for EXP2 (6000 samples)
    group1_stems = [
         "OV_replication_For_paper_7",  # pure DDM (z not included)
         "OV_replication_For_paper_2",  # DDM + SP (z free)
         "OV_replication_For_paper_8",  # pure aDDM (z not included)
         "OV_replication_For_paper_3",   # aDDM + SP (z included)
     ]
    
    # # Group 2 (rest)
    # group2_stems = [
    #     "garcia_replication_ES_VAL_10",  # Value_diff + DTA (z free)
    #     "garcia_replication_ES_VAL_7",   # aDDM + SP (z free)
    #     "garcia_replication_ES_VAL_26",  # a ~ OV (z free), shared inattn
    #     "garcia_replication_ES_VAL_35",  # dual inattn (no a~OV?)
    #     "garcia_replication_ES_VAL_36",  # dual inattn, a ~ OV
    # ]
   # for EXP2 (6000 Samples) Group 2 (rest)
    group2_stems = [
        "OV_replication_For_paper_4",  # Value_diff + DTA (z free)
        "OV_replication_For_paper_3",  # aDDM + SP (z free)
        "OV_replication_For_paper_6",  # a ~ OV (z free), dual inattn
        "OV_replication_For_paper_5"  # dual inattn (no a~OV?)
#        "OV_replication_ES_VAL_36",    #  aDDM, a ~ OV
    ]
    # ---------- Run ----------
    compare_group("group1_first4_OV", group1_stems, chain_idxs=(0,1,2))
    compare_group("group2_rest_OV",   group2_stems, chain_idxs=(0,1,2))
    
    idata_8 = load_idata("OV_replication_For_paper_8", chain_idxs=(0,1,2))
    az.loo(idata_8)

    
    
    
    
#     PROJECT_DIR = os.environ.get("PROJECT_DIR", "/workspace")
#     MODELS_DIR = os.path.join(PROJECT_DIR, "models_dir_garcia")    
#     nc_paths = [
#         os.path.join(MODELS_DIR,"garcia_replication_ES_VAL_34_2.nc"),
#         os.path.join(MODELS_DIR,"garcia_replication_ES_VAL_34_1.nc"),
#         os.path.join(MODELS_DIR,"garcia_replication_ES_VAL_34_0.nc"),
#     ]
#     idatas = [az.from_netcdf(p) for p in nc_paths]

#     # 2. concatenate along a new “chain” axis
#     idata = az.concat(idatas, dim="chain")

#     # shortcut to pull out a flattened array of draws for any var
#     def draws(varname):
#         da = idata.posterior[varname]
#         return da.stack(sample=["chain","draw"]).values


#     import numpy as np

#     vIA_E = np.abs(draws("v_ES_InattentionW_E"))

#     # 3. extract everything you need
#     a    = draws("a")
#     t        = draws("t")
#     z        = draws("z")
#     #inter    = draws("v_Intercept")
#     vA   = draws("v_ES_AttentionW")
#     #vIA_E    = draws("v_ES_InattentionW_E")
#     vIA_S    = draws("v_ES_InattentionW_S")


#     # 4. compute the θ’s
#     thetaE  = vIA_E / vA
#     thetaS  = vIA_S / vA
    
#     # ----- Posterior contrasts, directional probabilities, ROPEs, and CSV -----
#     import numpy as np

#     d_theta_unsigned = thetaE - thetaS          
#     d_b2_unsigned    = vIA_E - vIA_S           

#     vIA_E_signed = draws("v_ES_InattentionW_E")   # no abs
#     thetaE_signed = vIA_E_signed / vA
#     thetaS_signed = vIA_S / vA                    
#     d_theta_signed = thetaE_signed - thetaS_signed
#     d_b2_signed    = vIA_E_signed - vIA_S

#     def summarize_diff(arr, name, rope=None):
#         m = arr.mean()
#         lo, hi = az.hdi(arr, 0.95)
#         p_pos = float(np.mean(arr > 0))
#         out = {"Contrast": name, "Mean": m, "HDI_lower": lo, "HDI_upper": hi, "P(>0)": p_pos}
#         if rope is not None:
#             out["ROPE_low"] = rope[0]
#             out["ROPE_high"] = rope[1]
#             out["ROPE_%"] = float(np.mean((arr >= rope[0]) & (arr <= rope[1])))
#         return out

#     rope_theta = (-0.02, 0.02)   # example for θ differences
#     rope_b2    = (-0.002, 0.002) # example for raw weight differences

#     summary_rows = []

#     # Unsigned (magnitude-oriented) estimands
#     summary_rows.append(summarize_diff(d_theta_unsigned, "Δθ (unsigned: |vIA_E|/vA − vIA_S/vA)", rope=rope_theta))
#     summary_rows.append(summarize_diff(d_b2_unsigned,   "Δb2 (unsigned: |vIA_E| − vIA_S)",       rope=rope_b2))

#     # Signed estimands (as estimated)
#     summary_rows.append(summarize_diff(d_theta_signed, "Δθ (signed: vIA_E/vA − vIA_S/vA)", rope=rope_theta))
#     summary_rows.append(summarize_diff(d_b2_signed,   "Δb2 (signed: vIA_E − vIA_S)",       rope=rope_b2))

#     df_contrasts = pd.DataFrame(summary_rows)

#     df_contrasts.to_csv(
#         os.path.join(MODELS_DIR, "posterior_contrast_summary_ES_VAL_34.csv"),
#         index=False
#     )

#     print("\nPosterior contrast summary:")
#     print(df_contrasts)
# # -------------------------------------------------------------------------


#     # helper for 95% HDI
#     def hdi(arr):
#         lo, hi = az.hdi(arr, hdi_prob=0.95)
#         return lo, hi

#     # 5A. build the group‐level MAP / HDI table
#     group = []
#     for name, arr in [
#         ("a", a),
#         ("t", t),
#         ("z", z),
#         ("v_ES_AttentionW",  vA),
#         ("v_ES_InattentionW_E",  vIA_E),
#         ("v_ES_InattentionW_S", vIA_S),
#         ("θE", thetaE),
#         ("θS", thetaS),
#     ]:
#         m = arr.mean()
#         lo, hi = hdi(arr)
#         group.append({"Parameter": name, "MAP": m, "HDI_lower": lo, "HDI_upper": hi})

#     df_group = pd.DataFrame(group)
#     df_group.to_csv(
#         os.path.join(MODELS_DIR, "group_level_MAP_table_ES_VAL_34.csv"),
#         index=False
#         )
#     print("group‐level estimates:")
#     print(df_group)

#     # 5B. build the paired‐difference table
#     rows = []
#     def diff(x, y): 
#         return x.mean() - y.mean(), *hdi(x - y)

#     for (label, xa, xb) in [
#         ("θE-θS", thetaE, thetaS),
#         ("E−S b2", vIA_E, vIA_S),

        
#     ]:
#         m, lo, hi = diff(xa, xb)
#         rows.append({"Comparison": label, "MeanDiff": m, "HDI_lower": lo, "HDI_upper": hi})

#     df_comp = pd.DataFrame(rows)
#     df_comp.to_csv(
#         os.path.join(MODELS_DIR, "combined_parameter_comparison_table_ES_VAL_34.csv"),
#         index=False
#         )
    
    
    # import os
    # import arviz as az
    # import pandas as pd

    # PROJECT_DIR = os.environ.get("PROJECT_DIR", "/workspace")
    # MODELS_DIR  = os.path.join(PROJECT_DIR, "models_dir_garcia")

    # # ---------- 1) Helper: load & concat chains for a model ----------
    # def load_idata(model_stem, chain_idxs=(0,1,2)):
    #     """
    #     model_stem: e.g., 'garcia_replication_ES_VAL_36'
    #     expects files: f"{model_stem}_{i}.nc" inside MODELS_DIR
    #     returns: InferenceData with chains concatenated along 'chain'
    #     """
    #     paths = [os.path.join(MODELS_DIR, f"{model_stem}_{i}.nc") for i in chain_idxs]
    #     idatas = [az.from_netcdf(p) for p in paths]
    #     # Concatenate along a new chain axis
    #     idata = az.concat(idatas, dim="chain")
    #     return idata

    # # ---------- 2) Load BOTH models (dual vs shared) ----------
    # # Adjust stems to your filenames
    # dual_stem   = "garcia_replication_ES_VAL_36"
    # shared_stem = "garcia_replication_ES_VAL_26"

    # # If you have 4 chains (0..3), set chain_idxs=(0,1,2,3)
    # dual_idata   = load_idata(dual_stem,   chain_idxs=(0,1,2))
    # shared_idata = load_idata(shared_stem, chain_idxs=(0,1,2))

    # # ---------- 3) LOO comparison (predictive fit) ----------
    # # Tip: If you get errors about 'log_likelihood', ensure your .nc includes it.
    # cmp_loo = az.compare({"dual": dual_idata, "shared": shared_idata},
    #                  method="BB-pseudo-BMA", ic="loo")
    # print("\nLOO compare (BB-pseudo-BMA):")
    # print(cmp_loo)

    # cmp_loo_df = cmp_loo.reset_index().rename(columns={"index":"model"})
    # cmp_loo_df.to_csv(os.path.join(MODELS_DIR, "VAL36_vs_VAL26_LOO.csv"), index=False)

    # # Also pointwise LOO (useful diagnostics; optional)
    # loo_dual   = az.loo(dual_idata, pointwise=True)
    # loo_shared = az.loo(shared_idata, pointwise=True)
    # pd.DataFrame({
    #     "dual_elpd":   [loo_dual.elpd_loo],
    #     "dual_p_loo":  [loo_dual.p_loo],
    #     "shared_elpd": [loo_shared.elpd_loo],
    #     "shared_p_loo":[loo_shared.p_loo]
    #     }).to_csv(os.path.join(MODELS_DIR, "VAL36_vs_VAL26_LOO_summary.csv"), index=False)

    # # ---------- 4) Stacking weights (model averaging) ----------
    # cmp_stack = az.compare({"dual": dual_idata, "shared": shared_idata},
    #                    method="stacking", ic="loo")
    # print("\nStacking weights (LOO):")
    # print(cmp_stack)

    # cmp_stack_df = cmp_stack.reset_index().rename(columns={"index":"model"})
    # cmp_stack_df.to_csv(os.path.join(MODELS_DIR, "VAL36_vs_VAL26_stacking.csv"), index=False)

    # # ---------- 5) Quick console rule-of-thumb ----------
    # best = cmp_loo_df.iloc[0]         # top row is the best by ELPD
    # other = cmp_loo_df.iloc[1]
    # print(f"\nBest by LOO: {best['model']} (elpd_diff=0 by definition).")
    # print(f"Runner-up: {other['model']}, elpd_diff={other['elpd_diff']:.2f}, "
    #       f"SE={other['elpd_diff_se']:.2f} — if |elpd_diff| > SE, that’s meaningful support for the best model.")
    #
    
#     # first table csv etc ...   
#     # first these need to be compared:
#     #1.
#     # garcia_replication_ES_VAL_16 (z not included)
#     # pure DDM
#             v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
#             reg_descr = [v_reg]
#     # 2.         
#     # garcia_repliction_ES_VAL_12  (z is free param, hence + SP)
#     # DDM + SP
#             v_reg = {'model': 'v ~ 0 + Value_diff', 'link_func': lambda x: x}
#             reg_descr = [v_reg]
#     # 3. 
#     # pure aDDM (z not incldued)
#     # garica_replication_ES_VAL_17
#             v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
#             reg_descr = [v_reg]
#     # 4. 
#     # aDDM + SP (z is included)
#     # garcia_replication_ES_VAL_7
#             v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
#             reg_descr = [v_reg]
            
    
#     # second table/csv etc....
#     # models to be compared in a seperate tabel/csv
#         #modles that I wanat to incldue to compare: 
   
#     # indpependent influences of value difference and Dwelltime advantage (z is free param)
#      # garcia_replication_ES_VAL_10
#             v_reg = {'model': 'v ~ 0 + Value_diff + DTA', 'link_func': lambda x: x}
#             reg_descr = [v_reg]
#  # normal attnetional inattentional no a varies by oV  # (z is free param)
#     #garcia_replication_ES_VAL_7
#             v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
#             reg_descr = [v_reg]
#     # garcia_replication_ES_VAL_26
#             v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW', 'link_func': lambda x: x}
#             reg_descr = [v_reg]
#             depends_on = {'a': 'OVcate'} 
#     # garcia_replication_ES_VAL_35
#             v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
#             reg_descr = [v_reg]
#     # garcia_replication_ES_VAL_36
#             v_reg = {'model': 'v ~ 0 + ES_AttentionW + ES_InattentionW_E + ES_InattentionW_S', 'link_func': lambda x: x}
#             reg_descr = [v_reg]
#             depends_on = {'a': 'OVcate'} 
    
    
    
    
    
    
    # import os
    # import arviz as az
    # import pandas as pd
    # import numpy as np

    # PROJECT_DIR = os.environ.get("PROJECT_DIR", "/workspace")
    # MODELS_DIR  = os.path.join(PROJECT_DIR, "models_dir_garcia")

    # # ---------- 1) Helper: load & concat chains for a model ----------
    # def load_idata(model_stem, chain_idxs=(0,1,2)):
        
    #     paths = [os.path.join(MODELS_DIR, f"{model_stem}_{i}.nc") for i in chain_idxs]
    #     idatas = [az.from_netcdf(p) for p in paths]
    #     idata = az.concat(idatas, dim="chain")
    #     return idata

    # # ---------- 2) Load BOTH models (dual vs shared) ----------
    # dual_stem   = "garcia_replication_ES_VAL_36"
    # shared_stem = "garcia_replication_ES_VAL_26"

    # # change chain_idxs if you truly only have 3 chains
    # dual_idata   = load_idata(dual_stem,   chain_idxs=(0,1,2))
    # shared_idata = load_idata(shared_stem, chain_idxs=(0,1,2))

    # # ---------- 3) LOO comparison (predictive fit) ----------
    # cmps = {"dual": dual_idata, "shared": shared_idata}
    # cmp_loo = az.compare(cmps, method="BB-pseudo-BMA", ic="loo")
    # print("\nLOO compare (BB-pseudo-BMA):")
    # print(cmp_loo)

    # cmp_loo_df = cmp_loo.reset_index().rename(columns={"index":"model"})
    # cmp_loo_df.to_csv(os.path.join(MODELS_DIR, "VAL36_vs_VAL26_LOO.csv"), index=False)

    # # Pointwise LOO + Pareto-k diagnostics (useful with those PSIS warnings)
    # loo_dual   = az.loo(dual_idata,   pointwise=True)
    # loo_shared = az.loo(shared_idata, pointwise=True)

    # pd.DataFrame({
    #     "dual_elpd":   [loo_dual.elpd_loo],
    #     "dual_p_loo":  [loo_dual.p_loo],
    #     "shared_elpd": [loo_shared.elpd_loo],
    #     "shared_p_loo":[loo_shared.p_loo]
    #     }).to_csv(os.path.join(MODELS_DIR, "VAL36_vs_VAL26_LOO_summary.csv"), index=False)

    # pareto_df = pd.DataFrame({
    #     "model": ["dual","shared"],
    #     "frac_k>0.7": [np.mean(loo_dual.pareto_k.values>0.7),
    #                np.mean(loo_shared.pareto_k.values>0.7)],
    #     "frac_k>1.0": [np.mean(loo_dual.pareto_k.values>1.0),
    #                np.mean(loo_shared.pareto_k.values>1.0)],
    #     })
    # pareto_df.to_csv(os.path.join(MODELS_DIR, "VAL36_vs_VAL26_pareto_k.csv"), index=False)
    # print("\nPareto-k reliability:")
    # print(pareto_df)

    # # ---------- 4) Stacking weights (model averaging) ----------
    # cmp_stack = az.compare(cmps, method="stacking", ic="loo")
    # print("\nStacking weights (LOO):")
    # print(cmp_stack)
    # cmp_stack_df = cmp_stack.reset_index().rename(columns={"index":"model"})
    # cmp_stack_df.to_csv(os.path.join(MODELS_DIR, "VAL36_vs_VAL26_stacking.csv"), index=False)

    # # ---------- 5) Quick console rule-of-thumb ----------
    # best  = cmp_loo_df.iloc[0]     # top row = best by ELPD
    # other = cmp_loo_df.iloc[1]
    # se_col = "elpd_diff_se" if "elpd_diff_se" in cmp_loo_df.columns else ("dse" if "dse" in cmp_loo_df.columns else None)

    # if se_col is not None:
    #     ratio = abs(other["elpd_diff"])/other[se_col] if other[se_col] != 0 else np.inf
    #     print(f"\nBest by LOO: {best['model']} (elpd_diff=0).")
    #     print(f"Runner-up: {other['model']}, elpd_diff={other['elpd_diff']:.2f}, "
    #           f"SE={other[se_col]:.2f} → |elpd_diff|/SE = {ratio:.2f}")
    # else:
    #     print(f"\nBest by LOO: {best['model']} (elpd_diff=0). Runner-up: {other['model']}, "
    #           f"elpd_diff={other['elpd_diff']:.2f} (SE column not found)")


################################### for LEESEE phase differences ##############################################################################
################################### for LEESEE phase differences ##############################################################################

"""def run_version3():
    # -------------------------------------------------------------------------
    # Version 3: Phase differences LE ES EE
    import pandas as pd
    import pickle
    import kabuki
    import scipy.stats as stats
    
    model_paths = [
        "/home/jovyan/OfficialTutorials/THESIS_HDDM/model_dir_OV_CCT_2/OV_replication_LEESEE_5_2.pkl",
        "/home/jovyan/OfficialTutorials/THESIS_HDDM/model_dir_OV_CCT_2/OV_replication_LEESEE_5_1.pkl",
        "/home/jovyan/OfficialTutorials/THESIS_HDDM/model_dir_OV_CCT_2/OV_replication_LEESEE_5_0.pkl"
    ]
    
    models = []
    for path in model_paths:
        with open(path, "rb") as f:
            models.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models)
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        'a',
        't(LE)',
        't(ES)',
        't(EE)', 
        'v_Intercept', 
        'v_AttentionW',
        'v_InattentionW:C(phase)[LE]',
        'v_InattentionW:C(phase)[ES]',
        'v_InattentionW:C(phase)[EE]',
    ])])
    print("DIC:", combinedModels.dic)
    print("BPIC:", combinedModels.mc.BPIC)
    
    a     = combinedModels.nodes_db.node['a']
    t_LE  = combinedModels.nodes_db.node['t(LE)']
    t_ES  = combinedModels.nodes_db.node['t(ES)']
    t_EE  = combinedModels.nodes_db.node['t(EE)']
    inter = combinedModels.nodes_db.node['v_Intercept']
    vA    = combinedModels.nodes_db.node['v_AttentionW']
    vIA_LE = combinedModels.nodes_db.node['v_InattentionW:C(phase)[LE]']
    vIA_ES = combinedModels.nodes_db.node['v_InattentionW:C(phase)[ES]']
    vIA_EE = combinedModels.nodes_db.node['v_InattentionW:C(phase)[EE]']
    
    # Group-level Table for OV (including theta = b2 / b1 per OV level)
    theta_LE  = vIA_LE.trace() / vA.trace()
    theta_ES  = vIA_ES.trace() / vA.trace()
    theta_EE  = vIA_EE.trace() / vA.trace()

    
    group_params = {
        "a": a.trace(),
        "t(LE)": t_LE.trace(),
        "t(ES)": t_ES.trace(),
        "t(EE)": t_EE.trace(),
        "v_Intercept": inter.trace(),
        "v_AttentionW": vA.trace(),
        "v_InattentionW:C(phase)[LE]": vIA_LE.trace(),
        "v_InattentionW:C(phase)[ES]": vIA_ES.trace(),
        "v_InattentionW:C(phase)[EE]": vIA_EE.trace(),
        "theta(LE)": theta_LE,
        "theta(ES)": theta_ES,
        "theta(EE)": theta_EE
    }
    
    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    for name, trace in group_params.items():
        group_results["Parameter"].append(name)
        group_results["MAP"].append(trace.mean())
        group_results["HDI_lower"].append(stats.mstats.mquantiles(trace, [0.025])[0])
        group_results["HDI_upper"].append(stats.mstats.mquantiles(trace, [0.975])[0])
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/home/jovyan/OfficialTutorials/THESIS_HDDM/figures_dir_OV_CCT_2/OV_replication_LEESEE_5/group_level_MAP_table_mod5.csv", index=False)
    print("Group-level parameter estimates:")
    print(df_group)
    
    def format_estimate(trace):
        map_val = trace.mean()
        hdi_lower = stats.mstats.mquantiles(trace, [0.025])[0]
        hdi_upper = stats.mstats.mquantiles(trace, [0.975])[0]
        return f"{map_val:.3f} [{hdi_lower:.3f}, {hdi_upper:.3f}]"
    
    def format_diff(diff_trace):
        map_val = diff_trace.mean()
        hdi_lower = stats.mstats.mquantiles(diff_trace, [0.025])[0]
        hdi_upper = stats.mstats.mquantiles(diff_trace, [0.975])[0]
        return f"{map_val:.3f} [{hdi_lower:.3f}, {hdi_upper:.3f}]"
    
    # theta for each phase
    theta_LE = vIA_LE.trace() / vA.trace()
    theta_ES = vIA_ES.trace() / vA.trace()
    theta_EE = vIA_EE.trace() / vA.trace()
    
    rows = []
    rows.append({
        "Parameter": "a",
        "Group-level": format_estimate(a.trace()),
        "LE-ES": "",
        "LE-EE": "",
        "ES-EE": ""
    })
    rows.append({
        "Parameter": "t (ndt)",
        "Group-level": "",
        "LE-ES": format_diff(t_LE.trace() - t_ES.trace()),
        "LE-EE": format_diff(t_LE.trace() - t_EE.trace()),
        "ES-EE": format_diff(t_ES.trace() - t_EE.trace())
    })
    rows.append({
        "Parameter": "θ",
        "Group-level": "",
        "LE-ES": format_diff(theta_LE - theta_ES),
        "LE-EE": format_diff(theta_LE - theta_EE),
        "ES-EE": format_diff(theta_ES - theta_EE)
    })
    rows.append({
        "Parameter": "b0",
        "Group-level": format_estimate(inter.trace()),
        "LE-ES": "",
        "LE-EE": "",
        "ES-EE": ""
    })
    rows.append({
        "Parameter": "b1",
        "Group-level": format_estimate(vA.trace()),
        "LE-ES": "",
        "LE-EE": "",
        "ES-EE": ""
    })
    rows.append({
        "Parameter": "b2",
        "Group-level": "",
        "LE-ES": format_diff(vIA_LE.trace() - vIA_ES.trace()),
        "LE-EE": format_diff(vIA_LE.trace() - vIA_EE.trace()),
        "ES-EE": format_diff(vIA_ES.trace() - vIA_EE.trace())
    })
    
    df_combined = pd.DataFrame(rows, columns=["Parameter", "Group-level", "LE-ES", "LE-EE", "ES-EE"])
    df_combined.to_csv("/home/jovyan/OfficialTutorials/THESIS_HDDM/figures_dir_OV_CCT_2/OV_replication_LEESEE_5/combined_parameter_comparison_table_m5.csv", index=False)
    print("Combined Parameter Comparison Table:")
    print(df_combined)
    
    # For model 1 (baseline model) with LE, ES, EE
    model_paths = [
        "/home/jovyan/OfficialTutorials/THESIS_HDDM/model_dir_OV_CCT_2/OV_replication_LEESEE_1_2.pkl",
        "/home/jovyan/OfficialTutorials/THESIS_HDDM/model_dir_OV_CCT_2/OV_replication_LEESEE_1_1.pkl",
        "/home/jovyan/OfficialTutorials/THESIS_HDDM/model_dir_OV_CCT_2/OV_replication_LEESEE_1_0.pkl"
    ]
    
    models = []
    for path in model_paths:
        with open(path, "rb") as f:
            models.append(pickle.load(f))
            
    combinedModels = kabuki.utils.concat_models(models)
    stats_summary = combinedModels.gen_stats()
    print(stats_summary[stats_summary.index.isin([
        'a',
        't',
        'v_Intercept', 
        'v_AttentionW',
        'v_InattentionW',
    ])])
    print("DIC:", combinedModels.dic)
    print("BPIC:", combinedModels.mc.BPIC)
    
    a     = combinedModels.nodes_db.node['a']
    t     = combinedModels.nodes_db.node['t']
    inter = combinedModels.nodes_db.node['v_Intercept']
    vA    = combinedModels.nodes_db.node['v_AttentionW']
    vIA   = combinedModels.nodes_db.node['v_InattentionW']
    
    mapping = {
        "a": "a",
        "t": "ndt",
        "v_Intercept": "b0",
        "v_AttentionW": "b1",
        "v_InattentionW": "b2"
    }
    
    group_params = {
        mapping["a"]: a.trace(),
        mapping["t"]: t.trace(),
        mapping["v_Intercept"]: inter.trace(),
        mapping["v_AttentionW"]: vA.trace(),
        mapping["v_InattentionW"]: vIA.trace()
    }
    
    group_results = {"Parameter": [], "MAP": [], "HDI_lower": [], "HDI_upper": []}
    def compute_stats(trace):
        return trace.mean(), stats.mstats.mquantiles(trace, [0.025])[0], stats.mstats.mquantiles(trace, [0.975])[0]
    
    for param, trace in group_params.items():
        map_val, lower, upper = compute_stats(trace)
        group_results["Parameter"].append(param)
        group_results["MAP"].append(map_val)
        group_results["HDI_lower"].append(lower)
        group_results["HDI_upper"].append(upper)
    
    theta_trace = vIA.trace() / vA.trace()
    map_val, lower, upper = compute_stats(theta_trace)
    group_results["Parameter"].append("theta")
    group_results["MAP"].append(map_val)
    group_results["HDI_lower"].append(lower)
    group_results["HDI_upper"].append(upper)
    
    df_group = pd.DataFrame(group_results)
    df_group.to_csv("/home/jovyan/OfficialTutorials/THESIS_HDDM/figures_dir_OV_CCT_2/OV_replication_LEESEE_1/group_level_MAP_table_mod1.csv", index=False)
    print("Group-level parameter estimates:")
    print(df_group) """

    
if __name__ == "__main__":
    
    version = 36
    if version == 10:
        run_version_1_a()
    elif version == 11:
        run_version_1_b()
    elif version == 12: 
        run_version_1_c()
    elif version == 13: 
        run_version_1_d()
    elif version == 21:
        run_version_2_a()
    elif version == 22:
        run_version_2_b()
    elif version == 23:
        run_version_14()
    elif version == 35:
        run_version_35()
    elif version == 36:
        run_version_36()
    
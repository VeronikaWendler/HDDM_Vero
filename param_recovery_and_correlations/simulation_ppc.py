# #Veronika Wendler

# Posterior Predictive Checks for the aDDM (this produces only RT & Accuracy, we programmed anther version in Matlab)
# can be used for both experiments, the 'garcia' quasi-replication (Exp1) and the 'OV' experiment, in which we manipulated overall value levels during learning
# you just need to set the paths accordingly


import os
import gc
import warnings
import random

import hddm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# USER SETTINGS
# =========================================================

model_paths = [
    "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/models/garcia_replication_For_paper_23_2.hddm",
    "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/models/garcia_replication_For_paper_23_1.hddm",
    "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/models/garcia_replication_For_paper_23_0.hddm"
]

output_dir = "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/figures/garcia_replication_For_paper_23"
os.makedirs(output_dir, exist_ok=True)

# IMPORTANT:
# Replace this with the actual column name in your original data
# Example possibilities: "dwell_adv", "dwellTimeAdvantage", "gaze_advantage"
DWELL_COL = "DwellTimeAdvantage"

# Posterior predictive settings
PPC_SAMPLES = 2000          # total posterior predictive replications
N_DISPLAY_REPLICATIONS = 8  # how many simulated lines to draw in quintile plots
N_QUANTILES = 5             # quintiles

# Random seed for reproducible displayed PPC lines
RANDOM_SEED = 123


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def style_ax(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.tick_params(axis="both", which="major", labelsize=11)


def save_close(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def detect_sample_column(ppc_df):
    """
    Tries to detect the posterior predictive replication/sample column
    after reset_index().
    """
    candidate_cols = ["sample", "level_1", "draw", "replication"]
    for col in candidate_cols:
        if col in ppc_df.columns:
            return col

    # fallback: look for a low-cardinality index-like column
    for col in ppc_df.columns:
        if col.startswith("level_"):
            return col

    raise ValueError(
        "Could not detect posterior predictive sample column. "
        f"Available columns are: {list(ppc_df.columns)}"
    )


def get_observed_and_ppc_data(best_model, ppc_samples=1000):
    """
    Returns:
      obs_df : original observed data
      ppc_df : posterior predictive trial-level dataframe with appended original columns
      sample_col : detected posterior sample column
    """
    print(f"Generating posterior predictive data with {ppc_samples} samples per node...")
    ppc_df = hddm.utils.post_pred_gen(best_model, samples=ppc_samples, append_data=True)
    ppc_df = ppc_df.reset_index()

    obs_df = best_model.data.copy().reset_index(drop=True)

    print("\nObserved data columns:")
    print(obs_df.columns.tolist())

    print("\nPosterior predictive data columns:")
    print(ppc_df.columns.tolist())

    sample_col = detect_sample_column(ppc_df)
    print(f"\nDetected PPC sample column: {sample_col}")

    if "response_sampled" not in ppc_df.columns:
        raise ValueError(
            "Expected column 'response_sampled' was not found in posterior predictive data."
        )
    if "rt_sampled" not in ppc_df.columns:
        raise ValueError(
            "Expected column 'rt_sampled' was not found in posterior predictive data."
        )

    return obs_df, ppc_df, sample_col


def compute_quantile_edges(series, n_quantiles=5):
    """
    Compute value-based quantile cut edges from observed data.
    Uses tiny jitter to reduce failures due to ties.
    """
    s = pd.Series(series).dropna().astype(float).copy()
    if s.empty:
        raise ValueError("Series is empty after dropping NaNs.")

    jitter = np.linspace(0, 1e-10, len(s))
    s_jittered = s + jitter

    _, edges = pd.qcut(
        s_jittered,
        q=n_quantiles,
        retbins=True,
        duplicates="drop"
    )

    edges = np.unique(edges)
    return edges


def assign_bins_from_edges(series, edges, labels):
    """
    Apply fixed edges to any series.
    """
    return pd.cut(
        series.astype(float),
        bins=edges,
        labels=labels,
        include_lowest=True
    )


def assign_rt_quantiles_within_group(df, value_col, group_col=None, n_quantiles=5):
    """
    Assign quantiles based on RT rank.
    If group_col is None, quantiles are assigned to the whole dataframe.
    If group_col is given, quantiles are assigned separately within each group.
    This is useful for simulated datasets, where each PPC sample gets its own RT quintiles.
    """
    labels = [str(i) for i in range(1, n_quantiles + 1)]

    if group_col is None:
        ranks = df[value_col].rank(method="first")
        return pd.qcut(ranks, q=n_quantiles, labels=labels, duplicates="drop")

    def _assign_one_group(x):
        ranks = x.rank(method="first")
        return pd.qcut(ranks, q=n_quantiles, labels=labels, duplicates="drop")

    return df.groupby(group_col)[value_col].transform(_assign_one_group)


def summarise_choice_by_bin(df, bin_col, response_col, order):
    """
    Returns mean choice per bin, reindexed to desired order.
    """
    out = (
        df.groupby(bin_col, observed=False)[response_col]
        .mean()
        .reindex(order)
    )
    return out


def summarise_choice_by_bin_and_sample(df, sample_col, bin_col, response_col, order):
    """
    Returns long-format sample-by-bin mean choice.
    """
    out = (
        df.groupby([sample_col, bin_col], observed=False)[response_col]
        .mean()
        .reset_index()
    )
    out[bin_col] = pd.Categorical(out[bin_col], categories=order, ordered=True)
    return out


def plot_choice_bars_and_ppc_lines(
    observed_series,
    simulated_long,
    sample_col,
    bin_col,
    x_labels,
    title,
    xlabel,
    ylabel,
    save_path,
    n_display_replications=8,
    y_limits=(0.30, 0.70)
):
    """
    Black bars = observed
    Thin colored lines = selected PPC replications
    Thick black dashed line = posterior predictive mean across replications
    """
    rng = np.random.default_rng(RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(x_labels))

    # Observed bars
    ax.bar(x, observed_series.values, width=0.8, color="black", label="Observed")

    # Draw selected PPC replication lines
    unique_samples = simulated_long[sample_col].dropna().unique()
    n_to_draw = min(n_display_replications, len(unique_samples))
    chosen_samples = rng.choice(unique_samples, size=n_to_draw, replace=False)

    first_line = True
    for s in chosen_samples:
        tmp = (
            simulated_long.loc[simulated_long[sample_col] == s]
            .sort_values(bin_col)
            .set_index(bin_col)
            .reindex(x_labels)
        )
        ax.plot(
            x,
            tmp["p_choose_S"].values,
            marker="x",
            linewidth=1.2,
            alpha=0.85,
            label="Posterior predictive samples" if first_line else None
        )
        first_line = False

    # PPC mean line
    ppc_mean = (
        simulated_long.groupby(bin_col, observed=False)["p_choose_S"]
        .mean()
        .reindex(x_labels)
    )
    ax.plot(
        x,
        ppc_mean.values,
        linestyle="--",
        linewidth=2.0,
        label="Posterior predictive mean"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(*y_limits)

    style_ax(ax)
    ax.legend(fontsize=10, facecolor="white", framealpha=1, edgecolor="black")

    save_close(fig, save_path)


def plot_rt_distribution(obs_df, ppc_df, model_name, save_path):
    """
    Observed RT distribution vs sampled RT distribution.
    """
    all_rt = pd.concat([obs_df["rt"], ppc_df["rt_sampled"]], axis=0).dropna()
    bins = np.histogram_bin_edges(all_rt, bins=50)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        obs_df["rt"],
        bins=bins,
        alpha=0.6,
        density=True,
        edgecolor="black",
        linewidth=0.5,
        label="Observed RTs"
    )
    ax.hist(
        ppc_df["rt_sampled"],
        bins=bins,
        alpha=0.6,
        density=True,
        edgecolor="black",
        linewidth=0.5,
        label="Simulated RTs"
    )

    ax.set_xlabel("Reaction time (s)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Posterior predictive check: RT distribution\n{model_name}", fontsize=14)

    style_ax(ax)
    ax.legend(fontsize=11, facecolor="white", framealpha=1, edgecolor="black")
    save_close(fig, save_path)


def plot_response_distribution(obs_df, ppc_df, model_name, save_path):
    """
    Response proportion plot with explicit S/E labels.
    """
    obs_prop = obs_df["response"].value_counts(normalize=True).sort_index()
    sim_prop = ppc_df["response_sampled"].value_counts(normalize=True).sort_index()

    all_levels = [0, 1]
    obs_prop = obs_prop.reindex(all_levels, fill_value=0)
    sim_prop = sim_prop.reindex(all_levels, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(all_levels))

    ax.bar(x - 0.2, obs_prop.values, width=0.4, label="Observed")
    ax.bar(x + 0.2, sim_prop.values, width=0.4, label="Simulated")

    ax.set_xticks(x)
    ax.set_xticklabels(["E (response = 0, lower bound)", "S (response = 1, upper bound)"], fontsize=11)
    ax.set_ylabel("Proportion", fontsize=13)
    ax.set_title(f"Posterior predictive check: response proportions\n{model_name}", fontsize=14)

    style_ax(ax)
    ax.legend(fontsize=11, facecolor="white", framealpha=1, edgecolor="black")
    save_close(fig, save_path)


def save_parameter_summary(best_model, save_path):
    """
    Save HDDM parameter summary statistics.
    """
    try:
        stats_df = best_model.print_stats()
        stats_df.to_csv(save_path)
        print(f"Parameter summary saved to {save_path}")
    except Exception as e:
        print(f"Could not save parameter summary: {e}")


def save_builtin_hddm_plots(best_model, model_name, output_dir):
    """
    Try to save built-in HDDM PPC plots.
    Availability depends on HDDM version/model class.
    """
    # Built-in posterior predictive plot
    try:
        best_model.plot_posterior_predictive(figsize=(12, 10))
        plt.suptitle(f"Built-in HDDM posterior predictive plot\n{model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"BuiltIn_PosteriorPredictive_{model_name}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        print("Saved built-in posterior predictive plot.")
    except Exception as e:
        print(f"Could not save built-in posterior predictive plot: {e}")

    # Built-in posterior quantile plot
    try:
        best_model.plot_posterior_quantiles(columns=3, hexbin=True)
        plt.suptitle(f"Built-in HDDM posterior quantiles\n{model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"BuiltIn_PosteriorQuantiles_{model_name}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        print("Saved built-in posterior quantiles plot.")
    except Exception as e:
        print(f"Could not save built-in posterior quantiles plot: {e}")


def save_post_pred_stats(best_model, obs_df, ppc_samples, model_name, output_dir):
    """
    Save HDDM posterior predictive summary stats.
    """
    try:
        print(f"Generating summary statistics with {ppc_samples} samples per node...")
        ppc_data_stats = hddm.utils.post_pred_gen(best_model, samples=ppc_samples)
        ppc_stats = hddm.utils.post_pred_stats(obs_df, ppc_data_stats)

        print("\nPosterior predictive summary statistics:")
        print(ppc_stats)

        save_path = os.path.join(output_dir, f"posterior_predictive_summary_{model_name}.csv")
        ppc_stats.to_csv(save_path)
        print(f"Summary statistics saved to {save_path}")

    except Exception as e:
        print(f"Could not generate posterior predictive summary statistics: {e}")


# =========================================================
# MODEL SELECTION
# =========================================================

best_model = None
best_model_path = None
best_model_name = None
best_dic = float("inf")

for path in model_paths:
    print(f"Loading model from: {path}")
    m = hddm.load(path)
    current_dic = m.dic
    print(f"Model DIC: {current_dic}")

    if current_dic < best_dic:
        if best_model is not None:
            del best_model
            gc.collect()

        best_dic = current_dic
        best_model = m
        best_model_path = path
        best_model_name = os.path.basename(path).replace(".hddm", "")
    else:
        del m
        gc.collect()

print(f"\nBest model selected: {best_model_name} with DIC = {best_dic}")


# =========================================================
# LOAD OBSERVED + PPC DATA
# =========================================================

obs_df, ppc_df, sample_col = get_observed_and_ppc_data(best_model, ppc_samples=PPC_SAMPLES)

# Basic checks
if DWELL_COL not in obs_df.columns:
    raise ValueError(
        f"DWELL_COL = '{DWELL_COL}' was not found in the observed data.\n"
        f"Available observed columns: {list(obs_df.columns)}"
    )

if DWELL_COL not in ppc_df.columns:
    raise ValueError(
        f"DWELL_COL = '{DWELL_COL}' was not found in the PPC dataframe.\n"
        "Because append_data=True was used, this usually means the column name needs to be corrected."
    )


# =========================================================
# 1) STANDARD PPC PLOTS
# =========================================================

plot_rt_distribution(
    obs_df=obs_df,
    ppc_df=ppc_df,
    model_name=best_model_name,
    save_path=os.path.join(output_dir, f"RT_Distribution_{best_model_name}.png")
)

plot_response_distribution(
    obs_df=obs_df,
    ppc_df=ppc_df,
    model_name=best_model_name,
    save_path=os.path.join(output_dir, f"Response_Proportions_{best_model_name}.png")
)

save_builtin_hddm_plots(
    best_model=best_model,
    model_name=best_model_name,
    output_dir=output_dir
)

save_parameter_summary(
    best_model=best_model,
    save_path=os.path.join(output_dir, f"Parameter_Summary_{best_model_name}.csv")
)

save_post_pred_stats(
    best_model=best_model,
    obs_df=obs_df,
    ppc_samples=PPC_SAMPLES,
    model_name=best_model_name,
    output_dir=output_dir
)


# =========================================================
# 2) P(choose S) BY DWELL QUINTILE
# =========================================================
# Uses ORIGINAL dwell data
# response = 1 means choose S (upper bound)

dwell_labels = ["E>>S", "E>S", "S~E", "S>E", "S>>E"]

# Fixed quintile edges from observed dwell advantage
dwell_edges = compute_quantile_edges(obs_df[DWELL_COL], n_quantiles=N_QUANTILES)

# If ties caused fewer than 5 bins, fall back to generic labels
n_dwell_bins = len(dwell_edges) - 1
if n_dwell_bins != 5:
    warnings.warn(
        f"Dwell variable produced {n_dwell_bins} bins instead of 5 because of ties. "
        "Using generic labels."
    )
    dwell_labels = [f"Q{i}" for i in range(1, n_dwell_bins + 1)]

obs_df["dwell_quintile"] = assign_bins_from_edges(obs_df[DWELL_COL], dwell_edges, dwell_labels)
ppc_df["dwell_quintile"] = assign_bins_from_edges(ppc_df[DWELL_COL], dwell_edges, dwell_labels)

# Observed P(choose S)
obs_dwell = summarise_choice_by_bin(
    df=obs_df,
    bin_col="dwell_quintile",
    response_col="response",
    order=dwell_labels
)

# Simulated P(choose S)
sim_dwell = summarise_choice_by_bin_and_sample(
    df=ppc_df,
    sample_col=sample_col,
    bin_col="dwell_quintile",
    response_col="response_sampled",
    order=dwell_labels
).rename(columns={"response_sampled": "p_choose_S"})

plot_choice_bars_and_ppc_lines(
    observed_series=obs_dwell,
    simulated_long=sim_dwell,
    sample_col=sample_col,
    bin_col="dwell_quintile",
    x_labels=dwell_labels,
    title=f"P(choose S) by dwell quintile\n{best_model_name}",
    xlabel="Dwell-time advantage quintile",
    ylabel="P(choose S = upper bound = response 1)",
    save_path=os.path.join(output_dir, f"P_ChooseS_by_DwellQuintile_{best_model_name}.png"),
    n_display_replications=N_DISPLAY_REPLICATIONS,
    y_limits=(0.30, 0.70)
)


# =========================================================
# 3) P(choose S) BY RT QUINTILE
# =========================================================
# Observed bars:
#   RT quintiles from observed RT
# Simulated lines:
#   RT quintiles computed WITHIN EACH simulated PPC sample using rt_sampled
# This is the most natural RT-quintile PPC.

rt_labels = [str(i) for i in range(1, N_QUANTILES + 1)]

obs_df["rt_quintile"] = assign_rt_quantiles_within_group(
    df=obs_df,
    value_col="rt",
    group_col=None,
    n_quantiles=N_QUANTILES
)

ppc_df["rt_quintile"] = assign_rt_quantiles_within_group(
    df=ppc_df,
    value_col="rt_sampled",
    group_col=sample_col,
    n_quantiles=N_QUANTILES
)

obs_rt = summarise_choice_by_bin(
    df=obs_df,
    bin_col="rt_quintile",
    response_col="response",
    order=rt_labels
)

sim_rt = summarise_choice_by_bin_and_sample(
    df=ppc_df,
    sample_col=sample_col,
    bin_col="rt_quintile",
    response_col="response_sampled",
    order=rt_labels
).rename(columns={"response_sampled": "p_choose_S"})

plot_choice_bars_and_ppc_lines(
    observed_series=obs_rt,
    simulated_long=sim_rt,
    sample_col=sample_col,
    bin_col="rt_quintile",
    x_labels=rt_labels,
    title=f"P(choose S) by RT quintile\n{best_model_name}",
    xlabel="RT quintile",
    ylabel="P(choose S = upper bound = response 1)",
    save_path=os.path.join(output_dir, f"P_ChooseS_by_RTQuintile_{best_model_name}.png"),
    n_display_replications=N_DISPLAY_REPLICATIONS,
    y_limits=(0.30, 0.70)
)


# =========================================================
# 4) SAVE THE UNDERLYING SUMMARIES AS CSV
# =========================================================

obs_dwell.to_csv(os.path.join(output_dir, f"Observed_PChooseS_by_DwellQuintile_{best_model_name}.csv"))
sim_dwell.to_csv(os.path.join(output_dir, f"Simulated_PChooseS_by_DwellQuintile_{best_model_name}.csv"), index=False)

obs_rt.to_csv(os.path.join(output_dir, f"Observed_PChooseS_by_RTQuintile_{best_model_name}.csv"))
sim_rt.to_csv(os.path.join(output_dir, f"Simulated_PChooseS_by_RTQuintile_{best_model_name}.csv"), index=False)

print("\nAll requested PPC plots and summaries were saved successfully.")


# =========================================================
# CLEANUP
# =========================================================

del best_model, obs_df, ppc_df
gc.collect()








# # libraries
# import os
# import gc
# import hddm
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import arviz as az

# model_paths = [
#     "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_4.hddm",
#     "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_3.hddm",
#     "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_2.hddm",
#     "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_1.hddm",
#     "/home/jovyan/OfficialTutorials/For_Linux/models_dir_OV/OV_replication_EE_5_0.hddm"
# ]

# # initialize variables for selecting the best model (lowest DIC) 
# best_model = None
# best_model_path = None
# best_dic = float('inf')
# best_model_name = None

# for path in model_paths:
#     print(f"Loading model from: {path}")
#     m = hddm.load(path)
#     current_dic = m.dic
#     print(f"Model DIC: {current_dic}")
    
#     if current_dic < best_dic:
#         best_dic = current_dic
#         best_model = m
#         best_model_path = path  
#         best_model_name = os.path.basename(path).replace(".hddm", "")
#     else:
#         del m
#         gc.collect()

# print("Best model selected:", best_model_name, "with DIC =", best_dic)


# # Posterior Predictive Data
# print("Generating posterior predictive data with (nr of samples) samples per node...")
# ppc_data = hddm.utils.post_pred_gen(best_model, samples=2000, append_data=True)     #samples=500
# print("Posterior predictive data (first few rows):")
# print(ppc_data.head())
# output_dir = "/home/jovyan/OfficialTutorials/For_Linux/figures_dir_OV/OV_replication_EE_5/diagnostics"
# os.makedirs(output_dir, exist_ok=True)

# # RT Distribution
# bins = np.histogram_bin_edges(best_model.data['rt'], bins=50)
# fig, ax = plt.subplots(figsize=(8,6))
# ax.hist(best_model.data['rt'], bins=bins, alpha=0.8, color='blue', label='Real RTs',
#         density=True, edgecolor='black', linewidth=0.5)
# ax.hist(ppc_data['rt_sampled'], bins=bins, alpha=0.8, color='red', label='Simulated RTs',
#         density=True, edgecolor='black', linewidth=0.5)
# ax.set_xlim(-10, 10)
# ax.set_xlabel("Reaction Time (RT)", fontsize=13)
# ax.set_ylabel("Frequency", fontsize=13)
# ax.set_title(f"Posterior Predictive Check - {best_model_name} - RT Distribution", fontsize=15)
# ax.tick_params(axis='both', which='major', labelsize=11)
# ax.legend(fontsize=11, facecolor='white', framealpha=1, edgecolor='black')
# ax.set_facecolor("white")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_color('black')
# rt_plot_path = os.path.join(output_dir, f"RT_Distribution_{best_model_name}.png")
# plt.savefig(rt_plot_path, dpi=300, bbox_inches='tight')
# plt.close(fig) 

# # Response Distribution
# real_response_counts = best_model.data['response'].value_counts(normalize=True).sort_index()
# simulated_response_counts = ppc_data['response_sampled'].value_counts(normalize=True).sort_index()

# fig, ax = plt.subplots(figsize=(8,6))
# ax.bar(real_response_counts.index - 0.2, real_response_counts.values, width=0.4, color='blue', label='Real Responses')
# ax.bar(simulated_response_counts.index + 0.2, simulated_response_counts.values, width=0.4, color='red', label='Simulated Responses')
# ax.legend(fontsize=11, facecolor='white', framealpha=1, edgecolor='black')
# ax.tick_params(axis='both', which='major', labelsize=11)
# ax.set_facecolor("white")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_color('black')
# ax.spines['left'].set_color('black')
# ax.set_xticks([0, 1])
# ax.set_xticklabels(["Response 0", "Response 1"], fontsize=13)
# ax.set_ylabel("Proportion", fontsize=13)
# ax.set_title(f"Posterior Predictive Check - {best_model_name} - Response Proportions", fontsize=14)
# response_plot_path = os.path.join(output_dir, f"Response_Proportions_{best_model_name}.png")
# plt.savefig(response_plot_path, dpi=300, bbox_inches='tight')
# plt.close(fig)

# # Generate and Save Summary Statistics
# print("Generating summary statistics with 800 samples per node...")
# ppc_data_2 = hddm.utils.post_pred_gen(best_model, samples=2000)        # , samples=500
# ppc_stats = hddm.utils.post_pred_stats(best_model.data, ppc_data_2)

# print("Posterior predictive summary statistics:")
# print(ppc_stats)

# summary_stats_path = os.path.join(output_dir, f"posterior_predictive_summary_{best_model_name}.csv")
# ppc_stats.to_csv(summary_stats_path)
# print(f"Summary statistics saved to {summary_stats_path}")

# del best_model, ppc_data, ppc_data_2
# gc.collect()




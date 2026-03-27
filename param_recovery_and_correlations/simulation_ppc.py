# #Veronika Wendler
#
# Posterior Predictive Checks for the aDDM
# Can be used for both experiments, the 'garcia' quasi-replication (Exp1)
# and the 'OV' experiment, in which overall value levels were manipulated.
# Set the paths accordingly.


import os
import gc
import warnings
import math
# ---------------------------------------------------------
# Matplotlib cache fix for cluster environments
# ---------------------------------------------------------
mpl_dir = os.path.join("/tmp", os.getenv("USER", "user"), "matplotlib")
os.makedirs(mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = mpl_dir

import hddm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning


warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
# =========================================================
# USER SETTINGS
# =========================================================

model_paths = [
    "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/models/garcia_replication_Final_1_2.hddm",
    "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/models/garcia_replication_Final_1_1.hddm",
    "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/models/garcia_replication_Final_1_0.hddm"
]

output_dir = "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/figures/garcia_replication_Final_1/ppc"
os.makedirs(output_dir, exist_ok=True)

# Name of dwell-time advantage column in original data
DWELL_COL = "DwellTimeAdvantage"

# Posterior predictive settings
PPC_SAMPLES = 2000
N_DISPLAY_REPLICATIONS = 8
N_QUANTILES = 5
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
    Detect posterior predictive sample column after reset_index().
    """
    candidate_cols = ["sample", "draw", "level_1", "replication"]
    for col in candidate_cols:
        if col in ppc_df.columns:
            return col

    for col in ppc_df.columns:
        if col.startswith("level_"):
            return col

    raise ValueError(
        "Could not detect posterior predictive sample column. "
        f"Available columns are: {list(ppc_df.columns)}"
    )


def safe_reset_index(df):
    """
    Reset index without crashing when an index level name already exists
    as a dataframe column (e.g., trial_idx).
    """
    df = df.copy()

    old_index_names = list(df.index.names)
    new_index_names = []

    for i, name in enumerate(old_index_names):
        base_name = name if name is not None else f"index_level_{i}"
        new_name = base_name

        if new_name in df.columns or new_name in new_index_names:
            new_name = f"idx_{base_name}"
            counter = 1
            while new_name in df.columns or new_name in new_index_names:
                new_name = f"idx_{base_name}_{counter}"
                counter += 1

        new_index_names.append(new_name)

    df.index = df.index.set_names(new_index_names)
    return df.reset_index()


def get_observed_and_ppc_data(best_model, ppc_samples=1000):
    """
    Returns:
      obs_df : original observed data
      ppc_df : posterior predictive trial-level dataframe with appended original columns
      sample_col : detected posterior sample column
    """
    print(f"Generating posterior predictive data with {ppc_samples} samples per node...")
    ppc_df = hddm.utils.post_pred_gen(best_model, samples=ppc_samples, append_data=True)

    print("\nPPC index names before reset:")
    print(ppc_df.index.names)

    print("\nPPC columns before reset:")
    print(ppc_df.columns.tolist())

    ppc_df = safe_reset_index(ppc_df)
    obs_df = best_model.data.copy().reset_index(drop=True)

    print("\nObserved data columns:")
    print(obs_df.columns.tolist())

    print("\nPosterior predictive data columns after reset:")
    print(ppc_df.columns.tolist())

    sample_col = detect_sample_column(ppc_df)
    print(f"\nDetected PPC sample column: {sample_col}")

    if "response_sampled" not in ppc_df.columns:
        raise ValueError("Expected column 'response_sampled' not found in posterior predictive data.")
    if "rt_sampled" not in ppc_df.columns:
        raise ValueError("Expected column 'rt_sampled' not found in posterior predictive data.")

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
    Assign RT quantiles based on rank.
    If group_col is None, quantiles are assigned to the full dataframe.
    If group_col is given, quantiles are assigned separately within each group.
    """
    labels = [str(i) for i in range(1, n_quantiles + 1)]

    if group_col is None:
        ranks = df[value_col].rank(method="first")
        return pd.qcut(ranks, q=n_quantiles, labels=labels, duplicates="drop")

    def _assign_one_group(x):
        ranks = x.rank(method="first")
        return pd.qcut(ranks, q=n_quantiles, labels=labels, duplicates="drop")

    return df.groupby(group_col)[value_col].transform(_assign_one_group)


def summarise_observed_by_bin(df, bin_col, response_col, order):
    """
    Observed summary by bin for a binary response column.
    Returns mean, sd, sem, n.
    """
    out = (
        df.groupby(bin_col, observed=False)[response_col]
        .agg(mean="mean", sd="std", n="count")
        .reindex(order)
    )
    out["sem"] = out["sd"] / np.sqrt(out["n"].replace(0, np.nan))
    out = out.reset_index().rename(columns={bin_col: "bin"})
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
    out = out.rename(columns={response_col: "p_choose_S"})
    return out


def build_ppc_comparison_table(observed_summary, simulated_long, bin_col, x_labels):
    """
    Build a comparison table between empirical summaries and PPC summaries.

    Includes:
    - observed mean/sd/sem/n
    - model mean/sd/sem/95% interval
    - overlap checks
    - approximate z-style difference test
    - PPC two-sided tail probability
    """
    obs = observed_summary.copy().set_index("bin").reindex(x_labels)

    sim_stats = (
        simulated_long.groupby(bin_col, observed=False)["p_choose_S"]
        .agg(
            model_mean="mean",
            model_sd="std",
            model_n="count",
            model_sem=lambda s: s.std(ddof=1) / np.sqrt(s.notna().sum()),
            model_pi95_low=lambda s: s.quantile(0.025),
            model_pi95_high=lambda s: s.quantile(0.975)
        )
        .reindex(x_labels)
    )

    out = obs.join(sim_stats)

    # SEM bands
    out["obs_sem_low"] = (out["mean"] - out["sem"]).clip(0, 1)
    out["obs_sem_high"] = (out["mean"] + out["sem"]).clip(0, 1)
    out["model_sem_low"] = (out["model_mean"] - out["model_sem"]).clip(0, 1)
    out["model_sem_high"] = (out["model_mean"] + out["model_sem"]).clip(0, 1)

    # Simple overlap diagnostics
    out["sem_band_overlap"] = (
        (out["obs_sem_low"] <= out["model_sem_high"]) &
        (out["model_sem_low"] <= out["obs_sem_high"])
    )

    out["obs_mean_in_model_sem_band"] = (
        (out["mean"] >= out["model_sem_low"]) &
        (out["mean"] <= out["model_sem_high"])
    )

    out["model_mean_in_obs_sem_band"] = (
        (out["model_mean"] >= out["obs_sem_low"]) &
        (out["model_mean"] <= out["obs_sem_high"])
    )

    out["obs_mean_in_model_pi95"] = (
        (out["mean"] >= out["model_pi95_low"]) &
        (out["mean"] <= out["model_pi95_high"])
    )

    # Approximate z-style comparison using combined SEM
    out["diff_model_minus_obs"] = out["model_mean"] - out["mean"]
    out["combined_sem"] = np.sqrt(out["sem"]**2 + out["model_sem"]**2)
    out["z_approx"] = out["diff_model_minus_obs"] / out["combined_sem"]

    def two_sided_norm_p(z):
        if pd.isna(z):
            return np.nan
        return math.erfc(abs(z) / np.sqrt(2.0))

    out["p_approx_2sided"] = out["z_approx"].apply(two_sided_norm_p)

    # PPC two-sided tail probability:
    # how extreme is the observed mean relative to the simulated distribution?
    ppc_pvals = []
    for b in x_labels:
        vals = simulated_long.loc[simulated_long[bin_col] == b, "p_choose_S"].dropna().values
        obs_mean = out.loc[b, "mean"]

        if len(vals) == 0 or pd.isna(obs_mean):
            ppc_pvals.append(np.nan)
            continue

        p_lower = np.mean(vals <= obs_mean)
        p_upper = np.mean(vals >= obs_mean)
        p_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))
        ppc_pvals.append(p_two_sided)

    out["ppc_p_2sided"] = ppc_pvals

    out = out.reset_index().rename(columns={"index": "bin"})
    return out


def plot_choice_bars_and_ppc_lines(
    observed_summary,
    simulated_long,
    sample_col,
    bin_col,
    x_labels,
    title,
    xlabel,
    ylabel,
    save_path,
    n_display_replications=8,
    y_limits=None,
    error_mode="sem",      # "sem", "sd", or "pi95"
    y_as_percent=True
):
    """
    Black bars = observed mean
    Black error bars = observed SEM
    Thin blue lines = selected PPC replications
    Solid red line = PPC mean
    Orange band = uncertainty around PPC mean
    """
    rng = np.random.default_rng(RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    x = np.arange(len(x_labels))

    observed_summary = observed_summary.copy().set_index("bin").reindex(x_labels)

    simulated_long = simulated_long.copy()
    simulated_long[bin_col] = pd.Categorical(
        simulated_long[bin_col],
        categories=x_labels,
        ordered=True
    )

    # Summary across all PPC samples
    ppc_summary = (
        simulated_long.groupby(bin_col, observed=False)["p_choose_S"]
        .agg(
            mean="mean",
            sd="std",
            sem=lambda s: s.std(ddof=1) / np.sqrt(s.notna().sum()),
            lo=lambda s: s.quantile(0.025),
            hi=lambda s: s.quantile(0.975)
        )
        .reindex(x_labels)
    )

    if error_mode == "sem":
        lower = ppc_summary["mean"] - ppc_summary["sem"]
        upper = ppc_summary["mean"] + ppc_summary["sem"]
        band_label = "Posterior predictive mean ± SEM"
    elif error_mode == "sd":
        lower = ppc_summary["mean"] - ppc_summary["sd"]
        upper = ppc_summary["mean"] + ppc_summary["sd"]
        band_label = "Posterior predictive mean ± SD"
    elif error_mode == "pi95":
        lower = ppc_summary["lo"]
        upper = ppc_summary["hi"]
        band_label = "Posterior predictive 95% interval"
    else:
        raise ValueError("error_mode must be 'sem', 'sd', or 'pi95'")

    lower = lower.clip(0, 1)
    upper = upper.clip(0, 1)

    scale = 100.0 if y_as_percent else 1.0

    obs_mean_vals = observed_summary["mean"].values.astype(float) * scale
    obs_sem_vals = observed_summary["sem"].values.astype(float) * scale

    mean_vals = ppc_summary["mean"].values.astype(float) * scale
    lower_vals = lower.values.astype(float) * scale
    upper_vals = upper.values.astype(float) * scale

    # Observed bars
    ax.bar(
        x,
        obs_mean_vals,
        width=0.8,
        color="black",
        alpha=0.95,
        label="Observed mean",
        zorder=1
    )

    # Observed SEM
    ax.errorbar(
        x,
        obs_mean_vals,
        yerr=obs_sem_vals,
        fmt="none",
        ecolor="black",
        elinewidth=2.0,
        capsize=5,
        capthick=2.0,
        zorder=5,
        label="Observed mean ± SEM"
    )

    # Selected PPC sample lines
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

        yvals = tmp["p_choose_S"].values.astype(float) * scale

        ax.plot(
            x,
            yvals,
            marker="o",
            markersize=4,
            linewidth=1.3,
            alpha=0.35,
            color="tab:blue",
            zorder=3,
            label="Posterior predictive samples" if first_line else None
        )
        first_line = False

    # Model uncertainty band
    ax.fill_between(
        x,
        lower_vals,
        upper_vals,
        color="tab:orange",
        alpha=0.30,
        zorder=2,
        label=band_label
    )

    # Model mean as solid red line
    ax.plot(
        x,
        mean_vals,
        linestyle="-",
        linewidth=3.0,
        color="tab:red",
        zorder=4,
        label="Posterior predictive mean"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        all_vals = np.concatenate([obs_mean_vals, mean_vals, lower_vals, upper_vals])
        ymin = max(0, np.nanmin(all_vals) - 5)
        ymax = min(100 if y_as_percent else 1, np.nanmax(all_vals) + 5)
        ax.set_ylim(ymin, ymax)

    if y_as_percent:
        ax.set_yticks(np.arange(0, 101, 10))

    style_ax(ax)
    ax.legend(fontsize=10, facecolor="white", framealpha=1, edgecolor="black", loc="best")
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

    ax.set_xlabel("Reaction time (signed RT, s)", fontsize=13)
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
    ax.set_xticklabels(
        ["E (response = 0, lower bound)", "S (response = 1, upper bound)"],
        fontsize=11
    )
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
    """
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


def attach_observed_bin_by_row_order(obs_df, ppc_df, sample_col, bin_col):
    """
    Attach an observed trial-wise bin label (e.g. dwell quintile) to PPC rows
    by within-draw row order.

    This avoids relying on subj_idx/trial_idx being unique.
    Assumes append_data=True preserved original row order within each PPC draw.
    """
    obs_map = obs_df[[bin_col]].copy().reset_index(drop=True)
    obs_map["_orig_row"] = np.arange(len(obs_map))

    ppc_df = ppc_df.copy()
    ppc_df["_orig_row"] = ppc_df.groupby(sample_col, sort=False).cumcount()

    # sanity check: each PPC draw should contain exactly len(obs_df) rows
    rows_per_draw = ppc_df.groupby(sample_col, sort=False)["_orig_row"].max() + 1
    expected_n = len(obs_map)

    if not (rows_per_draw == expected_n).all():
        raise ValueError(
            "PPC draws do not all have the same number of rows as the observed dataset.\n"
            f"Expected {expected_n} rows per draw, got:\n{rows_per_draw.describe()}"
        )

    ppc_df = ppc_df.drop(columns=[bin_col], errors="ignore").merge(
        obs_map,
        on="_orig_row",
        how="left",
        validate="many_to_one"
    )

    return ppc_df

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
# response = 1 means choose S (upper bound)

dwell_labels = ["E>>S", "E>S", "S~E", "S>E", "S>>E"]

# Compute observed dwell quintile edges
dwell_edges = compute_quantile_edges(obs_df[DWELL_COL], n_quantiles=N_QUANTILES)
n_dwell_bins = len(dwell_edges) - 1

if n_dwell_bins != 5:
    warnings.warn(
        f"Dwell variable produced {n_dwell_bins} bins instead of 5 because of ties. "
        "Using generic labels."
    )
    dwell_labels = [f"Q{i}" for i in range(1, n_dwell_bins + 1)]

# Assign dwell bins in observed data
obs_df["dwell_quintile"] = assign_bins_from_edges(
    obs_df[DWELL_COL],
    dwell_edges,
    dwell_labels
)

# Attach observed dwell bins to PPC rows by original row position within each draw
ppc_df = attach_observed_bin_by_row_order(
    obs_df=obs_df,
    ppc_df=ppc_df,
    sample_col=sample_col,
    bin_col="dwell_quintile"
)

print("\nObserved dwell quintile counts:")
print(obs_df["dwell_quintile"].value_counts(dropna=False).sort_index())

print("\nPPC dwell quintile counts after row-order mapping:")
print(ppc_df["dwell_quintile"].value_counts(dropna=False).sort_index())

print("\nNumber of missing PPC dwell quintiles:")
print(ppc_df["dwell_quintile"].isna().sum())

# Observed P(choose S) summary
obs_dwell_summary = summarise_observed_by_bin(
    df=obs_df,
    bin_col="dwell_quintile",
    response_col="response",
    order=dwell_labels
)

# Simulated P(choose S) by sample
sim_dwell = summarise_choice_by_bin_and_sample(
    df=ppc_df,
    sample_col=sample_col,
    bin_col="dwell_quintile",
    response_col="response_sampled",
    order=dwell_labels
)

# Comparison table
dwell_comparison = build_ppc_comparison_table(
    observed_summary=obs_dwell_summary,
    simulated_long=sim_dwell,
    bin_col="dwell_quintile",
    x_labels=dwell_labels
)

print("\nObserved dwell summary:")
print(obs_dwell_summary)

print("\nFirst rows of simulated dwell summary:")
print(sim_dwell.head(15))

print("\nDwell comparison table:")
print(dwell_comparison)

plot_choice_bars_and_ppc_lines(
    observed_summary=obs_dwell_summary,
    simulated_long=sim_dwell,
    sample_col=sample_col,
    bin_col="dwell_quintile",
    x_labels=dwell_labels,
    title=f"P(choose S) by dwell quintile\n{best_model_name}",
    xlabel="Dwell-time advantage quintile",
    ylabel="P(choose S) in %",
    save_path=os.path.join(output_dir, f"P_ChooseS_by_DwellQuintile_{best_model_name}.png"),
    n_display_replications=N_DISPLAY_REPLICATIONS,
    y_limits=(30, 70),
    error_mode="sem",
    y_as_percent=True
)

# =========================================================
# 3) P(choose S) BY RT QUINTILE
# =========================================================
# Use absolute RT so quintiles reflect speed, not decision boundary sign.

rt_labels = [str(i) for i in range(1, N_QUANTILES + 1)]

obs_df["rt_abs"] = obs_df["rt"].abs()
ppc_df["rt_abs_sampled"] = ppc_df["rt_sampled"].abs()

obs_df["rt_quintile"] = assign_rt_quantiles_within_group(
    df=obs_df,
    value_col="rt_abs",
    group_col=None,
    n_quantiles=N_QUANTILES
)

ppc_df["rt_quintile"] = assign_rt_quantiles_within_group(
    df=ppc_df,
    value_col="rt_abs_sampled",
    group_col=sample_col,
    n_quantiles=N_QUANTILES
)

# Observed P(choose S) summary
obs_rt_summary = summarise_observed_by_bin(
    df=obs_df,
    bin_col="rt_quintile",
    response_col="response",
    order=rt_labels
)

# Simulated P(choose S) by sample
sim_rt = summarise_choice_by_bin_and_sample(
    df=ppc_df,
    sample_col=sample_col,
    bin_col="rt_quintile",
    response_col="response_sampled",
    order=rt_labels
)

# Comparison table
rt_comparison = build_ppc_comparison_table(
    observed_summary=obs_rt_summary,
    simulated_long=sim_rt,
    bin_col="rt_quintile",
    x_labels=rt_labels
)

print("\nObserved RT summary:")
print(obs_rt_summary)

print("\nFirst rows of simulated RT summary:")
print(sim_rt.head(15))

print("\nRT comparison table:")
print(rt_comparison)

plot_choice_bars_and_ppc_lines(
    observed_summary=obs_rt_summary,
    simulated_long=sim_rt,
    sample_col=sample_col,
    bin_col="rt_quintile",
    x_labels=rt_labels,
    title=f"P(choose S) by RT quintile\n{best_model_name}",
    xlabel="RT quintile",
    ylabel="P(choose S) in %",
    save_path=os.path.join(output_dir, f"P_ChooseS_by_RTQuintile_{best_model_name}.png"),
    n_display_replications=N_DISPLAY_REPLICATIONS,
    y_limits=(0, 100),
    error_mode="sem",
    y_as_percent=True
)

# =========================================================
# 4) SAVE UNDERLYING SUMMARIES AS CSV
# =========================================================

obs_dwell_summary.to_csv(
    os.path.join(output_dir, f"Observed_Dwell_Summary_{best_model_name}.csv"),
    index=False
)

sim_dwell.to_csv(
    os.path.join(output_dir, f"Simulated_Dwell_BySample_{best_model_name}.csv"),
    index=False
)

dwell_comparison.to_csv(
    os.path.join(output_dir, f"Dwell_Model_vs_Observed_Comparison_{best_model_name}.csv"),
    index=False
)

obs_rt_summary.to_csv(
    os.path.join(output_dir, f"Observed_RT_Summary_{best_model_name}.csv"),
    index=False
)

sim_rt.to_csv(
    os.path.join(output_dir, f"Simulated_RT_BySample_{best_model_name}.csv"),
    index=False
)

rt_comparison.to_csv(
    os.path.join(output_dir, f"RT_Model_vs_Observed_Comparison_{best_model_name}.csv"),
    index=False
)

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




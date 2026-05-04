# Posterior predictive checks for the aDDM
# Can be used for both experiments, the 'garcia' quasi-replication (exp1)
# and the 'ov' experiment, in which overall value levels were manipulated.
#
# This version selects the best .hddm file by lowest DIC and only uses that
# selected model for posterior predictive checks and plotting.

import os
import gc
import warnings

mpl_dir = os.path.join("/tmp", os.getenv("USER", "user"), "matplotlib")
os.makedirs(mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = mpl_dir

import hddm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# ==========================================================
# paths
# ==========================================================

model_paths = [
    "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/models/OV_replication_Final_7_2.hddm",
    "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/models/OV_replication_Final_7_1.hddm",
    "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/models/OV_replication_Final_7_0.hddm"
]

output_dir = "/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm/figures/OV_replication_Final_7/ppc"
os.makedirs(output_dir, exist_ok=True)

analysis_name = "best_chain"

dwell_col = "DwellTimeAdvantage"
sub_col = "subj_idx"

ppc_sample = 2000
quintiles = 5

bootstrap_samp = 5000
bootstrap_ci = 0.95
rand_seed = 123

# ==========================================================
# plot style
# ==========================================================

font_tick = 20
font_label = 24
font_title = 26
font_legend = 16

plt.rcParams.update({
    "font.size": font_tick,
    "axes.labelsize": font_label,
    "axes.titlesize": font_title,
    "xtick.labelsize": font_tick,
    "ytick.labelsize": font_tick,
    "legend.fontsize": font_legend,
    "figure.titlesize": font_title,
})

observed_color = "black"
simulated_color = "red"
bar_face_color = "white"

# ==========================================================
# helpers
# ==========================================================

def style_ax(ax):
    ax.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=font_tick,
        width=1.8,
        length=7
    )

    ax.xaxis.label.set_size(font_label)
    ax.yaxis.label.set_size(font_label)
    ax.title.set_size(font_title)


def save_close(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def detect_sample_column(ppc_df):
    cols = ["sample", "draw"]

    for col in cols:
        if col in ppc_df.columns:
            return col

    for col in ppc_df.columns:
        if str(col).startswith("level_"):
            return col

    raise ValueError(
        "Could not detect posterior predictive sample column."
    )


def safe_reset_index(df):
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


def select_best_model_by_dic(model_paths):
    best_model = None
    best_model_path = None
    best_model_name = None
    best_dic = float("inf")

    for path in model_paths:
        print(f"Loading model from: {path}")

        model = hddm.load(path)
        current_dic = model.dic

        print(f"Model DIC: {current_dic}")

        if current_dic < best_dic:
            if best_model is not None:
                del best_model
                gc.collect()

            best_model = model
            best_model_path = path
            best_model_name = os.path.basename(path).replace(".hddm", "")
            best_dic = current_dic

        else:
            del model
            gc.collect()

    print(f"\nBest model selected: {best_model_name}")
    print(f"Best DIC: {best_dic}")
    print(f"Best model path: {best_model_path}")

    return best_model, best_model_name, best_dic


def get_observed_and_ppc_data(model, ppc_samples=1000):
    print(f"Generating posterior predictive data with {ppc_samples} samples per node...")

    ppc_df = hddm.utils.post_pred_gen(
        model,
        samples=ppc_samples,
        append_data=True
    )

    print("\nPPC index names before reset:")
    print(ppc_df.index.names)

    print("\nPPC columns before reset:")
    print(ppc_df.columns.tolist())

    ppc_df = safe_reset_index(ppc_df)
    obs_df = model.data.copy().reset_index(drop=True)

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
    s = pd.Series(series).dropna().astype(float).copy()

    if s.empty:
        raise ValueError("Series is empty.")

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
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=edges,
        labels=labels,
        include_lowest=True
    )


def assign_rt_quantiles_within_group(df, value_col, group_col=None, n_quantiles=5):
    labels = [str(i) for i in range(1, n_quantiles + 1)]

    if group_col is None:
        vals = pd.to_numeric(df[value_col], errors="coerce")
        out = pd.Series(pd.NA, index=df.index, dtype="object")
        valid = vals.notna()

        if valid.sum() > 0:
            ranks = vals[valid].rank(method="first")
            out.loc[valid] = pd.qcut(
                ranks,
                q=n_quantiles,
                labels=labels,
                duplicates="drop"
            ).astype(str)

        return pd.Categorical(out, categories=labels, ordered=True)

    def assign_one_group(x):
        x = pd.to_numeric(x, errors="coerce")
        out = pd.Series(pd.NA, index=x.index, dtype="object")
        valid = x.notna()

        if valid.sum() > 0:
            ranks = x[valid].rank(method="first")
            out.loc[valid] = pd.qcut(
                ranks,
                q=n_quantiles,
                labels=labels,
                duplicates="drop"
            ).astype(str)

        return out

    result = df.groupby(group_col)[value_col].transform(assign_one_group)

    return pd.Categorical(result, categories=labels, ordered=True)


def attach_observed_columns_by_row_order(obs_df, ppc_df, sample_col, cols_to_attach):
    obs_map = obs_df[cols_to_attach].copy().reset_index(drop=True)
    obs_map["orig_row"] = np.arange(len(obs_map))

    ppc_df = ppc_df.copy()
    ppc_df["orig_row"] = ppc_df.groupby(sample_col, sort=False).cumcount()

    rows_per_draw = ppc_df.groupby(sample_col, sort=False)["orig_row"].max() + 1
    expected_n = len(obs_map)

    if not (rows_per_draw == expected_n).all():
        raise ValueError(
            "PPC draws do not have the same number of rows as the observed data.\n"
            f"Expected {expected_n} rows per draw, got:\n{rows_per_draw.describe()}"
        )

    ppc_df = ppc_df.drop(columns=cols_to_attach, errors="ignore").merge(
        obs_map,
        on="orig_row",
        how="left",
        validate="many_to_one"
    )

    return ppc_df


def bootstrap_mean_ci(values, n_boot=5000, ci=0.95, seed=123):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    if len(values) == 1:
        return values[0], values[0], values[0]

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)

    n = len(values)

    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower = np.quantile(boot_means, alpha / 2.0)
    upper = np.quantile(boot_means, 1.0 - alpha / 2.0)
    mean_val = np.mean(values)

    return mean_val, lower, upper


def summarise_observed_by_bin_subjectwise_bootstrap(
    df,
    subject_col,
    bin_col,
    response_col,
    order,
    n_boot=5000,
    ci=0.95,
    seed=123
):
    tmp = df[[subject_col, bin_col, response_col]].copy()
    tmp = tmp.dropna(subset=[subject_col, bin_col, response_col])
    tmp[bin_col] = pd.Categorical(tmp[bin_col], categories=order, ordered=True)

    subj_bin = (
        tmp.groupby([subject_col, bin_col], observed=False)[response_col]
        .mean()
        .reset_index()
        .rename(columns={response_col: "subject_mean"})
    )

    subj_bin[bin_col] = pd.Categorical(
        subj_bin[bin_col],
        categories=order,
        ordered=True
    )

    rows = []

    for i, b in enumerate(order):
        vals = subj_bin.loc[subj_bin[bin_col] == b, "subject_mean"].dropna().values

        mean_val, boot_low, boot_high = bootstrap_mean_ci(
            vals,
            n_boot=n_boot,
            ci=ci,
            seed=seed + i
        )

        sd_val = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
        sem_val = sd_val / np.sqrt(len(vals)) if len(vals) > 1 else np.nan

        rows.append({
            "bin": b,
            "mean": mean_val,
            "sd": sd_val,
            "n_subjects": len(vals),
            "sem": sem_val,
            "boot_ci_low": boot_low,
            "boot_ci_high": boot_high,
        })

    summary = pd.DataFrame(rows)

    return summary, subj_bin


def summarise_simulated_by_bin_sample_subjectwise(
    df,
    sample_col,
    subject_col,
    bin_col,
    response_col,
    order
):
    tmp = df[[sample_col, subject_col, bin_col, response_col]].copy()
    tmp = tmp.dropna(subset=[sample_col, subject_col, bin_col, response_col])
    tmp[bin_col] = pd.Categorical(tmp[bin_col], categories=order, ordered=True)

    subj_bin = (
        tmp.groupby([sample_col, subject_col, bin_col], observed=False)[response_col]
        .mean()
        .reset_index()
        .rename(columns={response_col: "subject_mean"})
    )

    subj_bin[bin_col] = pd.Categorical(
        subj_bin[bin_col],
        categories=order,
        ordered=True
    )

    draw_bin = (
        subj_bin.groupby([sample_col, bin_col], observed=False)["subject_mean"]
        .agg(
            p_choose_s="mean",
            draw_subject_sd="std",
            n_subjects="count"
        )
        .reset_index()
    )

    draw_bin["draw_subject_sem"] = draw_bin["draw_subject_sd"] / np.sqrt(
        draw_bin["n_subjects"].replace(0, np.nan)
    )

    draw_bin[bin_col] = pd.Categorical(
        draw_bin[bin_col],
        categories=order,
        ordered=True
    )

    return draw_bin, subj_bin


def build_ppc_comparison_table(observed_summary, simulated_long, bin_col, x_labels):
    obs = observed_summary.copy().set_index("bin").reindex(x_labels)

    sim_stats = (
        simulated_long.groupby(bin_col, observed=False)
        .agg(
            model_mean=("p_choose_s", "mean"),
            model_sd=("p_choose_s", "std"),
            model_n_draws=("p_choose_s", "count"),
            model_pi95_low=("p_choose_s", lambda s: s.quantile(0.025)),
            model_pi95_high=("p_choose_s", lambda s: s.quantile(0.975)),
            mean_draw_subject_sd=("draw_subject_sd", "mean"),
            mean_draw_subject_sem=("draw_subject_sem", "mean"),
            pi95_draw_subject_sd_low=("draw_subject_sd", lambda s: s.quantile(0.025)),
            pi95_draw_subject_sd_high=("draw_subject_sd", lambda s: s.quantile(0.975)),
            pi95_draw_subject_sem_low=("draw_subject_sem", lambda s: s.quantile(0.025)),
            pi95_draw_subject_sem_high=("draw_subject_sem", lambda s: s.quantile(0.975)),
        )
        .reindex(x_labels)
    )

    out = obs.join(sim_stats)

    out["obs_mean_in_model_pi95"] = (
        (out["mean"] >= out["model_pi95_low"])
        & (out["mean"] <= out["model_pi95_high"])
    )

    out["model_mean_in_obs_boot95"] = (
        (out["model_mean"] >= out["boot_ci_low"])
        & (out["model_mean"] <= out["boot_ci_high"])
    )

    out["obs_boot95_overlaps_model_pi95"] = (
        (out["boot_ci_low"] <= out["model_pi95_high"])
        & (out["model_pi95_low"] <= out["boot_ci_high"])
    )

    out["diff_model_minus_obs"] = out["model_mean"] - out["mean"]

    ppc_pvals = []

    for b in x_labels:
        vals = simulated_long.loc[
            simulated_long[bin_col] == b,
            "p_choose_s"
        ].dropna().values

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
    bin_col,
    x_labels,
    title,
    xlabel,
    ylabel,
    save_path,
    y_limits=None,
    y_as_percent=True
):
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    x = np.arange(len(x_labels))

    observed_summary = observed_summary.copy().set_index("bin").reindex(x_labels)

    scale = 100.0 if y_as_percent else 1.0

    obs_mean_vals = observed_summary["mean"].values.astype(float) * scale
    obs_low_vals = observed_summary["boot_ci_low"].values.astype(float) * scale
    obs_high_vals = observed_summary["boot_ci_high"].values.astype(float) * scale

    obs_yerr = np.vstack([
        obs_mean_vals - obs_low_vals,
        obs_high_vals - obs_mean_vals
    ])

    ax.bar(
        x,
        obs_mean_vals,
        width=0.8,
        facecolor=bar_face_color,
        edgecolor=observed_color,
        linewidth=2.6,
        zorder=1
    )

    ax.errorbar(
        x,
        obs_mean_vals,
        yerr=obs_yerr,
        fmt="none",
        ecolor=observed_color,
        elinewidth=2.6,
        capsize=7,
        capthick=2.6,
        zorder=5
    )

    legend_handles = []
    legend_labels = []

    empirical_handle = ax.errorbar(
        [],
        [],
        yerr=[[1], [1]],
        fmt="s",
        mfc=bar_face_color,
        mec=observed_color,
        mew=2.2,
        ms=12,
        ecolor=observed_color,
        elinewidth=2.4,
        capsize=7,
        capthick=2.4
    )

    legend_handles.append(empirical_handle)
    legend_labels.append("Empirical mean ± 95% bootstrap CI")

    if simulated_long is not None and len(simulated_long) > 0:
        simulated_long = simulated_long.copy()

        simulated_long[bin_col] = pd.Categorical(
            simulated_long[bin_col],
            categories=x_labels,
            ordered=True
        )

        model_summary = (
            simulated_long.groupby(bin_col, observed=False)["p_choose_s"]
            .agg(
                mean="mean",
                pi95_low=lambda s: s.quantile(0.025),
                pi95_high=lambda s: s.quantile(0.975),
            )
            .reindex(x_labels)
        )

        model_mean_vals = model_summary["mean"].values.astype(float) * scale
        model_pi95_low = model_summary["pi95_low"].values.astype(float) * scale
        model_pi95_high = model_summary["pi95_high"].values.astype(float) * scale

        ppc_band = ax.fill_between(
            x,
            model_pi95_low,
            model_pi95_high,
            color=simulated_color,
            alpha=0.18,
            zorder=2
        )

        ppc_line, = ax.plot(
            x,
            model_mean_vals,
            linestyle="-",
            linewidth=3.5,
            color=simulated_color,
            zorder=6
        )

        legend_handles.append(ppc_line)
        legend_labels.append("Posterior predictive mean")
        legend_handles.append(ppc_band)
        legend_labels.append("Posterior predictive 95% interval")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=font_tick)
    ax.set_xlabel(xlabel, fontsize=font_label)
    ax.set_ylabel(ylabel, fontsize=font_label)
    ax.set_title(title, fontsize=font_title)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        all_vals = np.concatenate([
            obs_mean_vals,
            obs_low_vals,
            obs_high_vals
        ])

        if simulated_long is not None and len(simulated_long) > 0:
            all_vals = np.concatenate([
                all_vals,
                model_mean_vals,
                model_pi95_low,
                model_pi95_high
            ])

        ymin = max(0, np.nanmin(all_vals) - 5)
        ymax = min(100 if y_as_percent else 1, np.nanmax(all_vals) + 5)

        ax.set_ylim(ymin, ymax)

    if y_as_percent:
        ax.set_yticks(np.arange(0, 101, 10))

    style_ax(ax)

    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=font_legend,
        facecolor="white",
        framealpha=1,
        edgecolor="black",
        loc="best"
    )

    save_close(fig, save_path)


def plot_rt_distribution(obs_df, ppc_df, save_path):
    all_rt = pd.concat([obs_df["rt"], ppc_df["rt_sampled"]], axis=0).dropna()
    bins = np.histogram_bin_edges(all_rt, bins=50)

    fig, ax = plt.subplots(figsize=(10, 7.5))

    ax.hist(
        obs_df["rt"],
        bins=bins,
        density=True,
        histtype="step",
        color=observed_color,
        linewidth=2.6,
        label="Observed RTs"
    )

    ax.hist(
        ppc_df["rt_sampled"],
        bins=bins,
        density=True,
        histtype="step",
        color=simulated_color,
        linewidth=2.6,
        label="Simulated RTs"
    )

    ax.set_xlabel("Reaction time (signed RT, s)", fontsize=font_label)
    ax.set_ylabel("Density", fontsize=font_label)
    ax.set_title("Posterior predictive check: RT distribution", fontsize=font_title)

    style_ax(ax)

    ax.legend(
        fontsize=font_legend,
        facecolor="white",
        framealpha=1,
        edgecolor="black"
    )

    save_close(fig, save_path)


def plot_response_distribution(obs_df, ppc_df, save_path):
    obs_prop = obs_df["response"].value_counts(normalize=True).sort_index()
    sim_prop = ppc_df["response_sampled"].value_counts(normalize=True).sort_index()

    all_levels = [0, 1]

    obs_prop = obs_prop.reindex(all_levels, fill_value=0)
    sim_prop = sim_prop.reindex(all_levels, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    x = np.arange(len(all_levels))

    ax.bar(
        x - 0.2,
        obs_prop.values,
        width=0.4,
        facecolor=bar_face_color,
        edgecolor=observed_color,
        linewidth=2.6,
        label="Observed"
    )

    ax.bar(
        x + 0.2,
        sim_prop.values,
        width=0.4,
        facecolor=bar_face_color,
        edgecolor=simulated_color,
        linewidth=2.6,
        label="Simulated"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        ["E\nresponse = 0\nlower bound", "S\nresponse = 1\nupper bound"],
        fontsize=font_tick
    )

    ax.set_ylabel("Proportion", fontsize=font_label)
    ax.set_title("Posterior predictive check: response proportions", fontsize=font_title)

    style_ax(ax)

    ax.legend(
        fontsize=font_legend,
        facecolor="white",
        framealpha=1,
        edgecolor="black"
    )

    save_close(fig, save_path)


def save_parameter_summary(model, save_path):
    try:
        stats_df = model.gen_stats()
        stats_df.to_csv(save_path)
        print(f"Parameter summary saved to {save_path}")

    except Exception as e:
        print(f"Could not save parameter summary: {e}")


def save_builtin_hddm_plots(model, output_dir):
    try:
        model.plot_posterior_predictive(figsize=(12, 10))

        plt.suptitle(
            "Built-in HDDM posterior predictive plot",
            fontsize=font_title
        )

        for ax in plt.gcf().axes:
            style_ax(ax)

        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, f"builtin_posterior_predictive_{analysis_name}.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()
        print("Saved built-in posterior predictive plot.")

    except Exception as e:
        print(f"Could not save built-in posterior predictive plot: {e}")

    try:
        model.plot_posterior_quantiles(columns=3, hexbin=True)

        plt.suptitle(
            "Built-in HDDM posterior quantiles",
            fontsize=font_title
        )

        for ax in plt.gcf().axes:
            style_ax(ax)

        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, f"builtin_posterior_quantiles_{analysis_name}.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()
        print("Saved built-in posterior quantiles plot.")

    except Exception as e:
        print(f"Could not save built-in posterior quantiles plot: {e}")


def save_post_pred_stats(model, obs_df, ppc_samples, output_dir):
    try:
        print(f"Generating summary statistics with {ppc_samples} samples per node...")

        ppc_data_stats = hddm.utils.post_pred_gen(
            model,
            samples=ppc_samples
        )

        ppc_stats = hddm.utils.post_pred_stats(obs_df, ppc_data_stats)

        print("\nPosterior predictive summary statistics:")
        print(ppc_stats)

        save_path = os.path.join(
            output_dir,
            f"posterior_predictive_summary_{analysis_name}.csv"
        )

        ppc_stats.to_csv(save_path)

        print(f"Summary statistics saved to {save_path}")

    except Exception as e:
        print(f"Could not generate posterior predictive summary statistics: {e}")


# ==========================================================
# model selection
# ==========================================================

best_model, best_model_name, best_dic = select_best_model_by_dic(model_paths)

# ==========================================================
# load observed data and posterior predictive data for best model
# ==========================================================

obs_df, ppc_df, sample_col = get_observed_and_ppc_data(
    model=best_model,
    ppc_samples=ppc_sample
)

if sub_col not in obs_df.columns:
    raise ValueError(
        f"sub_col = '{sub_col}' was not found in the observed data.\n"
        f"Available observed columns: {list(obs_df.columns)}"
    )

if dwell_col not in obs_df.columns:
    raise ValueError(
        f"dwell_col = '{dwell_col}' was not found in the observed data.\n"
        f"Available observed columns: {list(obs_df.columns)}"
    )

ppc_df = attach_observed_columns_by_row_order(
    obs_df=obs_df,
    ppc_df=ppc_df,
    sample_col=sample_col,
    cols_to_attach=[sub_col]
)

# ==========================================================
# general PPC plots using best model only
# ==========================================================

plot_rt_distribution(
    obs_df=obs_df,
    ppc_df=ppc_df,
    save_path=os.path.join(output_dir, f"rt_distribution_{analysis_name}.png")
)

plot_response_distribution(
    obs_df=obs_df,
    ppc_df=ppc_df,
    save_path=os.path.join(output_dir, f"response_proportions_{analysis_name}.png")
)

save_builtin_hddm_plots(
    model=best_model,
    output_dir=output_dir
)

save_parameter_summary(
    model=best_model,
    save_path=os.path.join(output_dir, f"parameter_summary_{analysis_name}.csv")
)

save_post_pred_stats(
    model=best_model,
    obs_df=obs_df,
    ppc_samples=ppc_sample,
    output_dir=output_dir
)

# ==========================================================
# p(choose s) by dwell-time bin
# ==========================================================

dwell_labels = ["E>>S", "E>S", "S~E", "S>E", "S>>E"]

dwell_edges = compute_quantile_edges(
    obs_df[dwell_col],
    n_quantiles=quintiles
)

n_dwell_bins = len(dwell_edges) - 1

if n_dwell_bins != 5:
    warnings.warn(
        f"Dwell variable produced {n_dwell_bins} bins."
    )

    dwell_labels = [f"Q{i}" for i in range(1, n_dwell_bins + 1)]

obs_df["dwell_quintile"] = assign_bins_from_edges(
    obs_df[dwell_col],
    dwell_edges,
    dwell_labels
)

ppc_df = attach_observed_columns_by_row_order(
    obs_df=obs_df,
    ppc_df=ppc_df,
    sample_col=sample_col,
    cols_to_attach=["dwell_quintile"]
)

obs_dwell_summary, obs_dwell_subjectwise = summarise_observed_by_bin_subjectwise_bootstrap(
    df=obs_df,
    subject_col=sub_col,
    bin_col="dwell_quintile",
    response_col="response",
    order=dwell_labels,
    n_boot=bootstrap_samp,
    ci=bootstrap_ci,
    seed=rand_seed
)

sim_dwell, sim_dwell_subjectwise = summarise_simulated_by_bin_sample_subjectwise(
    df=ppc_df,
    sample_col=sample_col,
    subject_col=sub_col,
    bin_col="dwell_quintile",
    response_col="response_sampled",
    order=dwell_labels
)

dwell_comparison = build_ppc_comparison_table(
    observed_summary=obs_dwell_summary,
    simulated_long=sim_dwell,
    bin_col="dwell_quintile",
    x_labels=dwell_labels
)

plot_choice_bars_and_ppc_lines(
    observed_summary=obs_dwell_summary,
    simulated_long=sim_dwell,
    bin_col="dwell_quintile",
    x_labels=dwell_labels,
    title="P(choose S) by dwell quintile",
    xlabel="Dwell-time advantage quintile",
    ylabel="P(choose S) in %",
    save_path=os.path.join(output_dir, f"p_choose_s_by_dwell_quintile_{analysis_name}.png"),
    y_limits=(30, 70),
    y_as_percent=True
)

# ==========================================================
# p(choose s) by RT quintile
# ==========================================================

rt_labels = [str(i) for i in range(1, quintiles + 1)]

obs_df["rt_abs"] = pd.to_numeric(
    obs_df["rt"],
    errors="coerce"
).abs()

ppc_df["rt_abs_sampled"] = pd.to_numeric(
    ppc_df["rt_sampled"],
    errors="coerce"
).abs()

obs_df["rt_quintile"] = assign_rt_quantiles_within_group(
    df=obs_df,
    value_col="rt_abs",
    group_col=None,
    n_quantiles=quintiles
)

ppc_df["rt_quintile"] = assign_rt_quantiles_within_group(
    df=ppc_df,
    value_col="rt_abs_sampled",
    group_col=sample_col,
    n_quantiles=quintiles
)

obs_rt_summary, obs_rt_subjectwise = summarise_observed_by_bin_subjectwise_bootstrap(
    df=obs_df,
    subject_col=sub_col,
    bin_col="rt_quintile",
    response_col="response",
    order=rt_labels,
    n_boot=bootstrap_samp,
    ci=bootstrap_ci,
    seed=rand_seed
)

sim_rt, sim_rt_subjectwise = summarise_simulated_by_bin_sample_subjectwise(
    df=ppc_df,
    sample_col=sample_col,
    subject_col=sub_col,
    bin_col="rt_quintile",
    response_col="response_sampled",
    order=rt_labels
)

rt_comparison = build_ppc_comparison_table(
    observed_summary=obs_rt_summary,
    simulated_long=sim_rt,
    bin_col="rt_quintile",
    x_labels=rt_labels
)

plot_choice_bars_and_ppc_lines(
    observed_summary=obs_rt_summary,
    simulated_long=sim_rt,
    bin_col="rt_quintile",
    x_labels=rt_labels,
    title="P(choose S) by RT quintile",
    xlabel="RT quintile",
    ylabel="P(choose S) in %",
    save_path=os.path.join(output_dir, f"p_choose_s_by_rt_quintile_{analysis_name}.png"),
    y_limits=(0, 100),
    y_as_percent=True
)

# ==========================================================
# save summaries as csv
# ==========================================================

obs_dwell_summary.to_csv(
    os.path.join(output_dir, f"observed_dwell_summary_{analysis_name}.csv"),
    index=False
)

obs_dwell_subjectwise.to_csv(
    os.path.join(output_dir, f"observed_dwell_subjectwise_{analysis_name}.csv"),
    index=False
)

sim_dwell.to_csv(
    os.path.join(output_dir, f"simulated_dwell_by_draw_{analysis_name}.csv"),
    index=False
)

sim_dwell_subjectwise.to_csv(
    os.path.join(output_dir, f"simulated_dwell_subjectwise_{analysis_name}.csv"),
    index=False
)

dwell_comparison.to_csv(
    os.path.join(output_dir, f"dwell_model_vs_observed_comparison_{analysis_name}.csv"),
    index=False
)

obs_rt_summary.to_csv(
    os.path.join(output_dir, f"observed_rt_summary_{analysis_name}.csv"),
    index=False
)

obs_rt_subjectwise.to_csv(
    os.path.join(output_dir, f"observed_rt_subjectwise_{analysis_name}.csv"),
    index=False
)

sim_rt.to_csv(
    os.path.join(output_dir, f"simulated_rt_by_draw_{analysis_name}.csv"),
    index=False
)

sim_rt_subjectwise.to_csv(
    os.path.join(output_dir, f"simulated_rt_subjectwise_{analysis_name}.csv"),
    index=False
)

rt_comparison.to_csv(
    os.path.join(output_dir, f"rt_model_vs_observed_comparison_{analysis_name}.csv"),
    index=False
)

print("\nAll requested PPC plots and summaries for the best chain were saved successfully.")
print(f"Selected best model: {best_model_name}")
print(f"Selected best DIC: {best_dic}")

# ==========================================================
# close
# ==========================================================

del best_model, obs_df, ppc_df
gc.collect()
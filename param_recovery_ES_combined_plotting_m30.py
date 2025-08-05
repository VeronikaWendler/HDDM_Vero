
import os
import types
import sys

# === Environment setup (must happen before importing matplotlib / arviz / numba-using libs) ===
# Local writable cache for matplotlib
cache_dir = os.path.abspath("./.matplotlib_cache")
os.environ["MPLCONFIGDIR"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# Disable numba JIT/caching to avoid the histogram locator error in ArviZ
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Dummy modules to avoid import errors (keep if needed on this cluster)
sys.modules.setdefault('winreg', types.ModuleType('winreg'))
sys.modules.setdefault('_gdbm', types.ModuleType('_gdbm'))

import pandas as pd
import numpy as np
import hddm
import os, sys, pickle, time
import datetime
import math
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")                   # for backend (does not require GUI)
import os, pathlib
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
# patch: make a dummy _gdbm module so “import _gdbm” never fails
import types, sys

import dill as pickle
from copy import deepcopy   # for modfiying z to be 0.55 (like in Sebastian's Matlab)
import argparse
# warning settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Plotting
# Stats 
from statsmodels.distributions.empirical_distribution import ECDF
# HDDM
from hddm.simulators.hddm_dataset_generators import simulator_h_c
from pathlib import Path
# scitnific computing and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
import statsmodels.api as sm

# HDDM related packages
import pymc as pm
import hddm
import kabuki
import arviz as az
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import statsmodels.api as sm




#------------------------------------------------------------------------------------------------------------------------------------------
PROJECT_DIR   = pathlib.Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

BASE_MODEL_DIR = PROJECT_DIR / "models_dir_combined"
FIG_DIR_ROOT   = PROJECT_DIR / "figures_dir_combined/combined_replication_ES_30/diagnostics"

chain0 = az.from_netcdf(BASE_MODEL_DIR / "combined_replication_ES_30_0.nc")
chain1 = az.from_netcdf(BASE_MODEL_DIR / "combined_replication_ES_30_1.nc")
chain2 = az.from_netcdf(BASE_MODEL_DIR / "combined_replication_ES_30_2.nc")

es27_infdata = az.concat([ chain0, chain1, chain2], dim="chain")

# getting the recovered data
recovered_nc = os.path.join(BASE_MODEL_DIR, "mES_combined_30_recovery.nc")
m_recovery_infdata = az.from_netcdf(recovered_nc)


# list of parameters 
param_list = [
    't',
    'v_Intercept',
    'v_z_IAW_chart',
    'v_z_IAW_image',
    'v_z_AttentionW:C(OVcate)[high]',
    'v_z_AttentionW:C(OVcate)[medium]',
    'v_z_AttentionW:C(OVcate)[low]',
    'a(low)',
    'a(medium)',
    'a(high)',
]


def save_diagnostics(idata, label, outdir, var_names=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Summary including r_hat and ESS
    summary = az.summary(idata, var_names=var_names, round_to=3)  # default includes r_hat, ess_bulk, ess_tail
    summary_file = outdir / f"{label}_summary_rhat_ess.csv"
    summary.to_csv(summary_file)
    print(f"Saved summary for {label} to {summary_file}")

    # Warn about potential convergence issues
    if 'r_hat' in summary.columns:
        high_rhat = summary['r_hat'] > 1.05
        if high_rhat.any():
            print(f"Warning: {label} has R-hat >1.05 for: {list(summary.index[high_rhat])}")
    if 'ess_bulk' in summary.columns:
        low_ess = summary['ess_bulk'] < 100  # heuristic; depends on total draws
        if low_ess.any():
            print(f"Warning: {label} has low bulk ESS for: {list(summary.index[low_ess])}")

    # Trace plots
    try:
        trace_fig = az.plot_trace(idata, var_names=var_names)
        trace_path = outdir / f"{label}_trace.png"
        trace_fig.savefig(trace_path, bbox_inches='tight', dpi=200)
        plt.close(trace_fig)
        print(f"Saved trace plot for {label} to {trace_path}")
    except Exception as e:
        print(f"Could not make trace plot for {label}: {e}")

    # Optional: LOO (may fail if model isn't compatible)
    try:
        loo_res = az.loo(idata, scale="deviance")
        loo_df = pd.DataFrame({
            "loo": [loo_res.loo],
            "p_loo": [loo_res.p_loo],
            "shape_k": [np.mean(loo_res.pareto_k.values)]
        })
        loo_path = outdir / f"{label}_loo.csv"
        loo_df.to_csv(loo_path, index=False)
        print(f"Saved LOO for {label} to {loo_path}")
    except Exception as e:
        print(f"LOO computation failed for {label}: {e}")

    return summary

# Run diagnostics and persist
fitted_summary = save_diagnostics(es27_infdata, "fitted", FIG_DIR_ROOT, var_names=param_list)
recovered_summary = save_diagnostics(m_recovery_infdata, "recovered", FIG_DIR_ROOT, var_names=param_list)
#-------------------------------------------------------------------------------------------------------------------------------------------

def regplot_with_r2(x,y,ax=None,scatter_kws=None,line_kws=None,margin=0.05,annot_kws=None):
    
    # Default settings
    if ax is None:
        ax = plt.gca()
    scatter_kws = {**{'s': 50, 'alpha': 0.6}, **(scatter_kws or {})}
    line_kws = {**{'color': 'red', 'linewidth': 2}, **(line_kws or {})}
    annot_kws = {**{'fontsize': 12, 'ha': 'left', 'va': 'top'}, **(annot_kws or {})}

    # Convert to pandas Series
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    # Drop NaNs
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    # Fit linear model
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    intercept, slope = model.params
    y_pred = model.predict(X)

    # Metrics
    r2 = r2_score(y, y_pred)
    p_value = model.pvalues.get(x.name if hasattr(x, 'name') else 'x', model.f_pvalue)

    # Scatter and regression line
    ax.scatter(x, y, **scatter_kws)
    ax.plot(x, y_pred, **line_kws)

    # Axis limits with equal scale
    all_vals = np.concatenate([x, y])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    span = max_val - min_val
    ax.set_xlim(min_val - margin * span, max_val + margin * span)
    ax.set_ylim(min_val - margin * span, max_val + margin * span)

    # Annotation text
    p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
    text = f"$R^2$={r2:.2f}\n{p_text}\n$\beta_1$={slope:.2f}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.8),
        **annot_kws
    )

    return ax



#REG PLOT FUNCTION
def regplot_with_corr(
    data=None,
    x="x",
    y="y",
    cor_anonot=True,
    reg_anonot=True,
    annot_kws={
        "fontsize": 8,
        "xy": (0.95, 0.05),
        "ha": 'right',
        "va": 'bottom'
    },
    scatter_kws={
        's': 40,
        "alpha": 0.4
    },
    ax=None,
    **kwargs
):
    if ax is None:
        ax = plt.gca()

    # Extract / coerce x and y into pandas Series
    if data is not None:
        data_x = data[x]
        data_y = data[y]
    else:
        # If they passed raw arrays or Series
        data_x = x if isinstance(x, pd.Series) else pd.Series(x)
        data_y = y if isinstance(y, pd.Series) else pd.Series(y)

    # Align the two series: keep only indices present in both
    try:
        data_x, data_y = data_x.align(data_y, join='inner')
    except Exception:  # fallback if not alignment-compatible (e.g., different types)
        data_x = pd.Series(data_x).reset_index(drop=True)
        data_y = pd.Series(data_y).reset_index(drop=True)
        minlen = min(len(data_x), len(data_y))
        data_x = data_x.iloc[:minlen]
        data_y = data_y.iloc[:minlen]

    # Drop any remaining NaNs
    mask = data_x.notna() & data_y.notna()
    data_x = data_x[mask]
    data_y = data_y[mask]

    # Plot regression line and scatter
    sns.regplot(
        x=data_x,
        y=data_y,
        ci=None if len(np.unique(data_y)) == 1 else 95,
        scatter_kws=scatter_kws,
        ax=ax
    )

    annot_text = ""
    if cor_anonot:
        # Pearson correlation
        if len(data_x) > 1 and len(data_y) > 1:
            correlation, p_value = pearsonr(data_x, data_y)
            p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
            annot_text += f"$r={correlation:.2f}$\n${p_str}$"
    if reg_anonot:
        # Linear regression coefficients
        X = sm.add_constant(data_x)
        model = sm.OLS(data_y, X).fit()
        intercept, slope = model.params
        annot_text += f"\n$\\beta_0={intercept:.2f}$\n$\\beta_1={slope:.2f}$"

    if annot_text:
        ax.annotate(
            annot_text,
            **annot_kws,
            xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
        )
    return ax

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def az_summary(infdata=None, half_a=False, param_names_order=None, **kwargs):

    param_df = az.summary(infdata, kind="stats",
                            **kwargs).reset_index(names="param_name")
    # col_values = ['mean', 'sd', "hdi_3%", "hdi_97%"]
    col_values = list(param_df.columns[1:5])

    pattern = r'(.*)_subj\.(\d+)'

    param_df[['param',
                'subj_idx']] = param_df['param_name'].str.extract(pattern)
    # param_df['param'] = param_df['param'].apply(lambda x: f'${x}$')
    param_df = param_df.dropna(subset=['subj_idx'])
    param_df['subj_idx'] = param_df['subj_idx'].astype(int)

    if half_a:
        param_df.loc[param_df['param'] == 'a',
                        col_values] = param_df.loc[param_df['param'] == 'a',
                                                col_values] / 2
    param_df = param_df.pivot(
        index='subj_idx', columns='param', values=col_values)
    if param_names_order is not None:
        new_index = pd.MultiIndex.from_tuples(
            [(level_0, param) for level_0 in col_values for param in param_names_order],names=[None, 'param'])
        param_df = param_df.reindex(columns=new_index)
    param_df.reset_index(inplace=True)
    param_df.columns.names = [None, None]

    return param_df


#-----------------------------------------------------------------------------------------------------------------------------------------

summary_df = az_summary(es27_infdata)['mean']
print(summary_df.columns.tolist())

# read in model
data_ES_27 = es27_infdata.observed_data.to_dataframe().reset_index(drop=True)
data_ES_27 = data_ES_27[data_ES_27['subj_idx'] <= 26].copy()

# read in model and coerce subj_idx
data_ES_27 = es27_infdata.observed_data.to_dataframe().reset_index(drop=True)
data_ES_27 = data_ES_27.copy()
data_ES_27['subj_idx'] = pd.to_numeric(data_ES_27['subj_idx'], errors='coerce')
data_ES_27 = data_ES_27.dropna(subset=['subj_idx'])
data_ES_27['subj_idx'] = data_ES_27['subj_idx'].astype(int)

# Keep only subj_idx <= 20
orig_subjects = sorted(data_ES_27['subj_idx'].unique())
filtered_subjects = sorted(data_ES_27['subj_idx'].unique())

print(f"Subjects before filtering: {orig_subjects}")
print(f"Subjects after keeping subj_idx <= 20: {filtered_subjects}")
assert all(s <= 26 for s in filtered_subjects), "Filtering failed: found subj_idx > 20."

# Subject-level summary restricted to kept subjects
subject_summary = az_summary(es27_infdata)['mean'].reset_index(names=['subj_idx'])
subject_summary = subject_summary[subject_summary['subj_idx'].isin(filtered_subjects)]

# Individual summary
df_ind_summary = (
    data_ES_27.groupby(['subj_idx', 'OVcate'])['rt']
    .describe()
    .reset_index()
)
df_ind_summary = (
    df_ind_summary.set_index('subj_idx')
    .join(subject_summary.set_index('subj_idx'))
    .reset_index()
)
def az_summary_group(infdata, **kwargs):
    # full summary as a DataFrame
    summary_df = az.summary(infdata, kind="stats", **kwargs).reset_index()
    # Check the actual column name for parameters (might be 'index' instead of 'param_name')
    param_col = 'index' if 'index' in summary_df.columns else 'param_name'
    # Set the index to the parameter names and select the mean estimates
    return summary_df.set_index(param_col)["mean"]

# check what's in the infdata
group_params = az_summary_group(es27_infdata)
print(group_params.index.tolist())

group_params = az.summary(es27_infdata, var_names=['~subj', '~std'], filter_vars='regex')
subject_params = az_summary(es27_infdata)['mean']



##################################################################################################################################
#PLOTTING

n_params = len(param_list)
n_cols = 4
figsize = (12, 6)
n_rows = (n_params + n_cols - 1) // n_cols

f, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
ax = ax.flatten()

# plot posterior distributions from original (fitted) model and the recovered model
# ee5_infdata is the original model's InferenceData
# m2_recovery_infdata is the one from the recovered model
for i, param in enumerate(param_list):
    az.plot_posterior(                         # fitted
        es27_infdata.posterior[param],
        ax=ax[i],
        color="darkorchid",                 #darkorchid
        linestyle="dashed",
        label="Fitted",
        hdi_prob='hide',
        point_estimate=None
    )
    az.plot_posterior(                         # recovered
        m_recovery_infdata.posterior[param],
        ax=ax[i],
        color="orange",
        linestyle="dotted",
        label="Recovered"
    )
    ax[i].set_ylabel('Density' if i % n_cols == 0 else '')
    ax[i].set_title(param)
    if i == 0:
        handles, labels = ax[0].get_legend_handles_labels()
    if ax[i].get_legend():
        ax[i].legend_.remove()

for j in range(i + 1, len(ax)):
    f.delaxes(ax[j])

f.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plot_path = os.path.join(FIG_DIR_ROOT, "Posterior_dens_fitted_recov.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------------------------------------------

# Forest Plot for Group-level Parameters
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_forest(
    [es27_infdata, m_recovery_infdata],
    model_names=["Fitted", "Recovered"],
    var_names=param_list,
    combined=True,
    ridgeplot_alpha=0.5,
    colors=["darkorchid", "orange"],               #darkorchid
    hdi_prob=0.95,
    ax=ax
)
ax.set_title("Comparison of Fitted and Recovered Parameters")
# IMPORTANT - make sure this fits the model and path
plot_path2 = os.path.join(FIG_DIR_ROOT, "Forest_plot_group.png")

plt.savefig(plot_path2, dpi=300, bbox_inches='tight')

plt.show()

#-------------------------------------------------------------------------------------------------------------
# HDI overlap summary 
# function to compute HDI overlap
def hdi_overlap(idata1, idata2, var_name, hdi_prob=0.95):
    """
    Compute the percent overlap of the HDIs between two posterior distributions
    """
    hdi1 = az.hdi(idata1.posterior[var_name], hdi_prob=hdi_prob).to_array().values
    hdi2 = az.hdi(idata2.posterior[var_name], hdi_prob=hdi_prob).to_array().values
    lower1, upper1 = hdi1.min(), hdi1.max()
    lower2, upper2 = hdi2.min(), hdi2.max()
    overlap = max(0, min(upper1, upper2) - max(lower1, lower2))
    total_range = max(upper1, upper2) - min(lower1, lower2)
    return overlap / total_range if total_range > 0 else 0

# function to compute difference HDI and check for overlap with 0
def difference_hdi(idata1, idata2, var_name, hdi_prob=0.94):
    """
    Compute the posterior difference (fitted minus recovered) for a parameter,
    then calculate the mean difference and its HDI.
    
    Returns:
        diff_mean: Mean difference
        diff_lower: Lower bound of the HDI
        diff_upper: Upper bound of the HDI
        equal: Boolean; True if the HDI includes 0 (interpreted as "Equal")
    """
    diff = idata1.posterior[var_name] - idata2.posterior[var_name]
    # Get HDI of the difference
    hdi_diff = az.hdi(diff, hdi_prob=hdi_prob).to_array().values
    lower, upper = hdi_diff.min(), hdi_diff.max()
    diff_mean = diff.mean().values
    # If 0 is within the interval, flag the difference as "Equal"
    equal = (lower <= 0) and (upper >= 0)
    return diff_mean, lower, upper, equal

# list of parameters 


# plotting and summary collection
n_params = len(param_list)
n_cols = 4
figsize = (12, 6)
n_rows = (n_params + n_cols - 1) // n_cols

f, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
ax = ax.flatten()

summary_data = []


for i, param in enumerate(param_list):
    try:
        # fitted posterior (specific colour for the phase)  darkorchid = ES, deepksyblue = EE, grey = ESEE
        az.plot_posterior(
            es27_infdata.posterior[param],
            ax=ax[i],
            color="darkorchid",
            linestyle="dashed",
            label="Fitted",
            hdi_prob='hide',
            point_estimate=None
        )
        az.plot_posterior(
            m_recovery_infdata.posterior[param],
            ax=ax[i],
            color="orange",                                     # recovred always orange
            linestyle="dotted",
            label="Recovered"
        )
        ax[i].set_ylabel('Density' if i % n_cols == 0 else '')
        ax[i].set_title(param)
        
        # get HDI overlap between the posteriors
        hdi_ol = hdi_overlap(es27_infdata, m_recovery_infdata, param)
        # get the difference (fitted - recovered) and its HDI
        diff_mean, diff_lower, diff_upper, equal = difference_hdi(es27_infdata, m_recovery_infdata, param)
        equality_str = "Equal" if equal else "Not Equal"
        annotation = (f"HDI overlap: {hdi_ol:.1%}\n"
                      f"Diff: {diff_mean:.2f} [{diff_lower:.2f}, {diff_upper:.2f}]\n"
                      f"{equality_str}")
        ax[i].annotate(annotation, xy=(0.5, 0.9), xycoords='axes fraction',
                       ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
        summary_data.append({
            'Parameter': param,
            'HDI_Overlap': hdi_ol,
            'Diff_Mean': diff_mean,
            'Diff_Lower': diff_lower,
            'Diff_Upper': diff_upper,
            'Equal': equal
        })
    except Exception as e:
        print(f"Could not process param '{param}': {e}")
        ax[i].axis('off')

for j in range(i + 1, len(ax)):
    f.delaxes(ax[j])

# legend
handles, labels = ax[0].get_legend_handles_labels()
f.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
#plt.show()

# CSV
summary_df = pd.DataFrame(summary_data)
csv_filename = FIG_DIR_ROOT / "parameter_recovery_summary_ES_m30.csv"
summary_df.to_csv(csv_filename, index=False)
print(f"Summary exported to {csv_filename}")

#---------------------------------------------------------------------------------------------------------------------------
# Individual-level Comparison, forest plot

ind_param_list = [param for param in es27_infdata.posterior.data_vars if 'subj' in param and 'std' not in param]
fig, ax = plt.subplots(figsize=(10, 20))
az.plot_forest(
    [es27_infdata, m_recovery_infdata],
    model_names=["Fitted", "Recovered"],
    var_names=ind_param_list,
    coords={'subj_idx': np.arange(1,27)},
    combined=True,
    ridgeplot_alpha=0.5,
    hdi_prob=0.95,
    ax=ax
)
ax.set_title("Individual-level Comparison of Fitted and Recovered Parameters")
plot_path2 = os.path.join(FIG_DIR_ROOT, "Forest_plot_ind.png")
plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
plt.show()

#----------------------------------------------------------------------------------------------------------------------------

#'REG PLOTS'
def get_subject_means(df):
    df = df.reset_index(names="subj_idx")
    df["subj_idx"] = df["subj_idx"].astype(int)
    return df.set_index("subj_idx")

param_fitted = az_summary(es27_infdata)["mean"]
param_recovery = az_summary(m_recovery_infdata)["mean"]

fitted_subj = get_subject_means(param_fitted)
recovered_subj = get_subject_means(param_recovery)

common = fitted_subj.index.intersection(recovered_subj.index)
if len(common) < max(len(fitted_subj), len(recovered_subj)):
    print(f"Warning: only comparing {len(common)} common subjects "
          f"(fitted had {len(fitted_subj)}, recovered had {len(recovered_subj)})")

fitted_aligned = fitted_subj.loc[common]
recovered_aligned = recovered_subj.loc[common]

# regression plots: one panel per parameter
fig, ax = plt.subplots(ncols=len(param_list), figsize=(3 * len(param_list), 3))
for i, param in enumerate(param_list):
    x = fitted_aligned[param]
    y = recovered_aligned[param]
    regplot_with_corr(x=x, y=y, ax=ax[i])
    if i == 0:
        ax[i].set_ylabel('Recovered')
    else:
        ax[i].set_ylabel('')
    ax[i].set_title(param)

plt.tight_layout()
plot_path3 = os.path.join(FIG_DIR_ROOT, "Reg_plots.png")
plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
plt.close(fig)



for i, param in enumerate(param_list):
    ax = regplot_with_r2(
        fitted_aligned[param],
        recovered_aligned[param],
        ax=ax[i],
        scatter_kws={'s':30, 'alpha':0.5},
        line_kws={'color':'green', 'linewidth':2},
        margin=0.1
    )
    ax.set_title(param)




















# def regplot_with_corr(
#     data=None,
#     x="x",
#     y="y",
#     cor_anonot=True,
#     reg_anonot=True,
#     show_bootstrap_ci=False,
#     bootstrap_iters=500,
#     annot_kws={
#         "fontsize": 8,
#         "xy": (0.95, 0.05),
#         "ha": "right",
#         "va": "bottom"
#     },
#     scatter_kws={"s": 40, "alpha": 0.4},
#     deming_lambda=1.0,
#     ax=None,
#     **kwargs
# ):
#     """
#     Scatter + regression with multiple correlation/agreement metrics.
#     Shows Pearson, Spearman, Kendall, Lin’s CCC, R², and Deming slope/intercept.
#     Optionally bootstraps Pearson & Spearman for CIs.
#     """
#     if ax is None:
#         ax = plt.gca()

#     # Extract / coerce x and y into pandas Series
#     if data is not None:
#         data_x = data[x]
#         data_y = data[y]
#     else:
#         data_x = x if isinstance(x, pd.Series) else pd.Series(x)
#         data_y = y if isinstance(y, pd.Series) else pd.Series(y)

#     # Align indices
#     try:
#         data_x, data_y = data_x.align(data_y, join="inner")
#     except Exception:
#         data_x = pd.Series(data_x).reset_index(drop=True)
#         data_y = pd.Series(data_y).reset_index(drop=True)
#         minlen = min(len(data_x), len(data_y))
#         data_x = data_x.iloc[:minlen]
#         data_y = data_y.iloc[:minlen]

#     # Drop NaNs
#     mask = data_x.notna() & data_y.notna()
#     data_x = data_x[mask]
#     data_y = data_y[mask]

#     if len(data_x) < 2 or len(data_y) < 2:
#         raise ValueError("Not enough data after filtering to compute correlations.")

#     # Scatter + OLS regression line
#     sns.regplot(
#         x=data_x,
#         y=data_y,
#         ci=None if len(np.unique(data_y)) == 1 else 95,
#         scatter_kws=scatter_kws,
#         ax=ax,
#         **kwargs
#     )

#     # Compute metrics
#     pearson_r, pearson_p = pearsonr(data_x, data_y)
#     spearman_r, spearman_p = spearmanr(data_x, data_y)
#     kendall_r, kendall_p = kendalltau(data_x, data_y)
#     ccc = concordance_ccc(data_x.values, data_y.values)

#     # OLS for R^2 and slope/intercept
#     X = sm.add_constant(data_x)
#     ols_model = sm.OLS(data_y, X).fit()
#     intercept_ols, slope_ols = ols_model.params
#     r2 = ols_model.rsquared

#     # Deming regression
#     deming_slope, deming_intercept = deming_regression(data_x.values, data_y.values, lambda_ratio=deming_lambda)

#     # Bootstrap CIs if requested (Pearson and Spearman)
#     pearson_ci = None
#     spearman_ci = None
#     if show_bootstrap_ci:
#         rng = np.random.default_rng()
#         prs = []
#         sps = []
#         n = len(data_x)
#         for _ in range(bootstrap_iters):
#             idx = rng.integers(0, n, size=n)
#             sample_x = data_x.iloc[idx].values
#             sample_y = data_y.iloc[idx].values
#             try:
#                 pr, _ = pearsonr(sample_x, sample_y)
#             except Exception:
#                 pr = np.nan
#             try:
#                 sr, _ = spearmanr(sample_x, sample_y)
#             except Exception:
#                 sr = np.nan
#             prs.append(pr)
#             sps.append(sr)
#         prs = np.array(prs)
#         sps = np.array(sps)
#         pearson_ci = np.nanpercentile(prs, [2.5, 97.5])
#         spearman_ci = np.nanpercentile(sps, [2.5, 97.5])

#     # Build annotation text
#     lines = []
#     if cor_anonot:
#         line1 = f"Pearson r={pearson_r:.2f}"
#         if pearson_ci is not None:
#             line1 += f" [{pearson_ci[0]:.2f},{pearson_ci[1]:.2f}]"
#         line2 = f"Spearman ρ={spearman_r:.2f}"
#         if spearman_ci is not None:
#             line2 += f" [{spearman_ci[0]:.2f},{spearman_ci[1]:.2f}]"
#         line3 = f"Kendall τ={kendall_r:.2f}"
#         line4 = f"CCC={ccc:.2f}"
#         lines.extend([line1, line2, line3, line4])
#     if reg_anonot:
#         lines.append(f"OLS: β₀={intercept_ols:.2f}, β₁={slope_ols:.2f}, $R^2$={r2:.2f}")
#         if not np.isnan(deming_slope):
#             lines.append(f"Deming: slope={deming_slope:.2f}, intercept={deming_intercept:.2f}")

#     annot_text = "\n".join(lines)

#     if annot_text:
#         ax.annotate(
#             annot_text,
#             **annot_kws,
#             xycoords="axes fraction",
#             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
#         )

#     return ax


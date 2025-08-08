# ============================================================
# Parameter-recovery loop – group- and individual-level
# ============================================================
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import hddm

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import trange

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- configuration -----------------------------------------
PROJECT_DIR    = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
BASE_MODEL_DIR = PROJECT_DIR / "models_dir_garcia"
FIG_DIR        = PROJECT_DIR / "figures_dir_garcia/garcia_replication_ES_35/recovery_m35"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EMPIRICAL_POST_PATHS = [
    BASE_MODEL_DIR / "garcia_replication_ES_35_0.nc",
    BASE_MODEL_DIR / "garcia_replication_ES_35_1.nc",
    BASE_MODEL_DIR / "garcia_replication_ES_35_2.nc",
]

N_REPS    = 5        # as in the paper
N_SAMPLES = 700
BURN      = 100

# group-level parameters (means)
PARAM_LIST = [
    't',
    'v_Intercept',
    'v_z_AttentionW:C(OVcate)[low]',
    'v_z_AttentionW:C(OVcate)[medium]',
    'v_z_AttentionW:C(OVcate)[high]',
    'v_z_IAW_chart',
    'v_z_IAW_image',
    'a_Intercept',
    'a_OVcate[T.low]',
    'a_OVcate[T.medium]',
]

# corresponding group-level SD names (as they appear in idata)
PARAM_LIST_SD = [
    't_std',
    'v_Intercept_std',
    'v_z_AttentionW:C(OVcate)[low]_std',
    'v_z_AttentionW:C(OVcate)[medium]_std',
    'v_z_AttentionW:C(OVcate)[high]_std',
    'v_z_IAW_chart_std',
    'v_z_IAW_image_std',
    'a_Intercept_std',
    'a_OVcate[T.low]_std',
    'a_OVcate[T.medium]_std',
]
PARAM_SD_MAP = dict(zip(PARAM_LIST, PARAM_LIST_SD))

# HDDM model spec
v_reg = {'model': 'v ~ 1 + z_AttentionW:C(OVcate) + z_IAW_chart + z_IAW_image', 'link_func': lambda x: x}
a_reg = {'model': 'a ~ 1 + OVcate', 'link_func': lambda x: x}
reg_descr = [v_reg, a_reg]

# ------------- helpers ---------------------------------------------------
### NEW: safe/atomic CSV writer
def atomic_to_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def extract_group_sample(idata, *, seed=None):
    rng   = np.random.default_rng(seed)
    draw  = rng.integers(idata.posterior.dims["draw"])
    chain = rng.integers(idata.posterior.dims["chain"])
    return {p: float(idata.posterior[p].isel(chain=chain, draw=draw)) for p in PARAM_LIST}

def extract_group_sd(idata, *, seed=None):
    rng   = np.random.default_rng(seed)
    draw  = rng.integers(idata.posterior.dims["draw"])
    chain = rng.integers(idata.posterior.dims["chain"])
    out = {}
    for p in PARAM_LIST:
        sd_name = PARAM_SD_MAP.get(p)
        if (sd_name is not None) and (sd_name in idata.posterior):
            out[p] = float(idata.posterior[sd_name].isel(chain=chain, draw=draw))
        else:
            out[p] = 0.0
    return out

def sample_true_subjects(mu_dict, sd_dict, subjects, *, seed=None):
    """Draw θ_i for each subject: Normal(mu, sd)."""
    rng = np.random.default_rng(seed)
    true_individuals = {}
    for s in subjects:
        true_individuals[s] = {}
        for p in PARAM_LIST:
            mu, sd = mu_dict[p], sd_dict[p]
            true_individuals[s][p] = float(rng.normal(mu, sd)) if sd > 0 else float(mu)
    return true_individuals

### NEW: helper to flatten per-subject 'true' draws (for saving)
def flatten_true_subjects(true_individuals, rep):
    rows = []
    for subj, pmap in true_individuals.items():
        for p, val in pmap.items():
            rows.append(dict(rep=rep, subj=subj, parameter=p, true=val))
    return rows

def simulate_dataset(true_individuals, raw_df):
    """Use subject-specific parameters to simulate trials."""
    sim_rows = []
    for _, tr in raw_df.iterrows():
        subj = int(tr["subj_idx"])
        ov   = tr["OVcate"]  # 'low'/'medium'/'high'
        pars = true_individuals[subj]

        # v per trial
        v_trial = (
            pars["v_Intercept"]
            + pars[f"v_z_AttentionW:C(OVcate)[{ov}]"] * tr["z_AttentionW"]
            + pars["v_z_IAW_chart"]  * tr["z_IAW_chart"]
            + pars["v_z_IAW_image"]  * tr["z_IAW_image"]
        )

        # a per trial (reference coding: high is reference -> no dummy)
        a_trial = pars["a_Intercept"]
        if ov == "low":
            a_trial += pars["a_OVcate[T.low]"]
        elif ov == "medium":
            a_trial += pars["a_OVcate[T.medium]"]

        trial_df, _ = hddm.generate.gen_rand_data(
            {"v": v_trial, "a": a_trial, "t": pars["t"]},
            size=1, subjs=1
        )
        for col in ["subj_idx", "OVcate", "z_AttentionW", "z_IAW_chart", "z_IAW_image"]:
            trial_df[col] = tr[col]
        sim_rows.append(trial_df)
    return pd.concat(sim_rows, ignore_index=True)

def refit_and_get_means(sim_df, seed):
    np.random.seed(seed)
    mdl = hddm.HDDMRegressor(
        sim_df, reg_descr,
        include=["a","t","v"],
        p_outlier=0.05,
        keep_regressor_trace=True,
        group_only_regressors=False,
    )
    mdl.find_starting_values()
    mdl.sample(N_SAMPLES, burn=BURN, db='ram', dbname=f'ram_{seed}')
    means = {p: mdl.nodes_db.loc[p, 'node'].trace().mean() for p in PARAM_LIST}
    return means

def extract_individual_means(mdl):
    """Posterior mean for subject-specific intercepts only."""
    out = {}
    for node in mdl.nodes_db.index:
        if "_subj." not in node:
            continue
        param, subj_str = node.split("_subj.")
        # keep *only* intercepts (and t)
        if param not in ("v_Intercept", "a_Intercept", "t"):
            continue
        subj = int(subj_str)
        out[(subj, param)] = mdl.nodes_db.loc[node, 'node'].trace().mean()
    return out
# -----------------------------------------------------------------------

# load empirical fit & predictors
empirical = az.concat([az.from_netcdf(p) for p in EMPIRICAL_POST_PATHS], dim="chain")
raw_df    = empirical.observed_data.to_dataframe().reset_index(drop=True)
raw_df["subj_idx"] = raw_df["subj_idx"].astype(int)
subjects = sorted(raw_df["subj_idx"].unique())

# storage
group_records = []
indiv_records = []
true_draw_records = []  ### NEW: keep a running log of per-subject "true" draws

# paths for partial saves
GROUP_PARTIAL_CSV = FIG_DIR / "partial_group.csv"
INDIV_PARTIAL_CSV = FIG_DIR / "partial_individual.csv"
TRUE_PARTIAL_CSV  = FIG_DIR / "partial_true_subject_draws.csv"  ### NEW

for rep in trange(N_REPS, desc="parameter-recovery", unit="rep"):
    try:
        mu = extract_group_sample(empirical, seed=rep)
        sd = extract_group_sd(empirical, seed=rep+999)

        true_individuals = sample_true_subjects(mu, sd, subjects, seed=42+rep)
        true_draw_records.extend(flatten_true_subjects(true_individuals, rep))
        atomic_to_csv(pd.DataFrame(true_draw_records), TRUE_PARTIAL_CSV)  # save ASAP

        sim_df = simulate_dataset(true_individuals, raw_df)

        θ_hat_group = refit_and_get_means(sim_df, seed=10_000 + rep)
        for p in PARAM_LIST:
            group_records.append(dict(rep=rep, parameter=p, true=mu[p], recovered=θ_hat_group[p]))

        np.random.seed(20_000 + rep)
        mdl = hddm.HDDMRegressor(sim_df, reg_descr, include=["a","t","v"],
                                 p_outlier=0.05, keep_regressor_trace=True,
                                 group_only_regressors=False)
        mdl.find_starting_values()
        mdl.sample(N_SAMPLES, burn=BURN, db='ram', dbname=f'indiv_{rep}')

        indiv_means = extract_individual_means(mdl)
        for (subj, param), rec_val in indiv_means.items():
            true_val = true_individuals[subj][param]
            indiv_records.append(dict(rep=rep, subj=subj, parameter=param,
                                      true=true_val, recovered=rec_val))
    finally:
        # always write what we have so far
        atomic_to_csv(pd.DataFrame(group_records), GROUP_PARTIAL_CSV)
        atomic_to_csv(pd.DataFrame(indiv_records), INDIV_PARTIAL_CSV)


# final CSVs
pd.DataFrame(group_records).to_csv(FIG_DIR/"true_vs_recovered_group.csv", index=False)
pd.DataFrame(indiv_records).to_csv(FIG_DIR/"true_vs_recovered_individual.csv", index=False)
pd.DataFrame(true_draw_records).to_csv(FIG_DIR/"true_subject_draws_all.csv", index=False)  ### NEW

# ---------- plotting ----------
sns.set_style("white")

# group plot
grp = pd.DataFrame(group_records)
g = sns.FacetGrid(grp, col="parameter", col_wrap=3, sharex=False, sharey=False, height=3.0)
g.map_dataframe(sns.scatterplot, x="true", y="recovered", s=28, alpha=.85)
for ax in g.axes.ravel():
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--k", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
g.set_axis_labels("true value", "posterior mean (recovered)")
g.tight_layout()
g.savefig(FIG_DIR/"scatter_group.png", dpi=300)

# individual plot (means only, all reps & subs)
ind = pd.DataFrame(indiv_records)
h = sns.FacetGrid(ind, col="parameter", col_wrap=3, sharex=False, sharey=False, height=3.0)
h.map_dataframe(sns.scatterplot, x="true", y="recovered", s=16, alpha=.5)
for ax in h.axes.ravel():
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--k", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
h.set_axis_labels("true value", "posterior mean (recovered)")
h.tight_layout()
h.savefig(FIG_DIR/"scatter_individual.png", dpi=300)

print("Done.")

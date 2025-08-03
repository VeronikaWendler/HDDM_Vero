# ============================================================
# Parameter-recovery loop – true vs recovered (GROUP parameters)
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
warnings.filterwarnings("ignore", category=FutureWarning)
# very top of the script
from tqdm.auto import trange, tqdm    


# ---------- config ------------------------------------------------------
PROJECT_DIR   = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
BASE_MODEL_DIR = PROJECT_DIR / "models_dir_garcia"
FIG_DIR        = PROJECT_DIR / "figures_dir_garcia/recovery_ES31"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EMPIRICAL_POST_PATHS = [
    BASE_MODEL_DIR / "garcia_replication_ES_31_0.nc",
    BASE_MODEL_DIR / "garcia_replication_ES_31_1.nc",
    BASE_MODEL_DIR / "garcia_replication_ES_31_2.nc",
]

N_REPS    = 10         # ≥ 500 recommended for a paper
N_SAMPLES = 1000        # ↑ when you have cluster time
BURN      = 100

PARAM_LIST = [
    "t",
    "a(low)", "a(medium)", "a(high)",
    "v_Intercept",
    "v_z_AttentionW",
    "v_z_IAW_chart:C(OVcate)[low]",
    "v_z_IAW_chart:C(OVcate)[medium]",
    "v_z_IAW_chart:C(OVcate)[high]",
    "v_z_IAW_image:C(OVcate)[low]",
    "v_z_IAW_image:C(OVcate)[medium]",
    "v_z_IAW_image:C(OVcate)[high]",
]


# ---------- HDDM model specification ------------------------------------
v_reg = {'model': 'v ~ 1 + z_AttentionW + z_IAW_chart:C(OVcate) + z_IAW_image:C(OVcate)', 'link_func': lambda x: x}
reg_descr = [v_reg]
depends_on={'a': 'OVcate'} 

# ---------- helper functions -------------------------------------------
def extract_group_sample(idata, *, seed=None):
    """Draw **one** joint posterior sample of group-level parameters."""
    rng   = np.random.default_rng(seed)
    draw  = rng.integers(idata.posterior.dims["draw"])
    chain = rng.integers(idata.posterior.dims["chain"])
    return {p: float(idata.posterior[p].isel(chain=chain, draw=draw))
            for p in PARAM_LIST}

def simulate_dataset(true_pars: dict, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a synthetic dataframe that keeps every original row
    but replaces RT / choice with simulated DDM data generated
    from the **GROUP** parameters in `true_pars`.
    """
    sim_rows = []
    for _, tr in raw_df.iterrows():
        ov     = tr["OVcate"]          
        a_val  = true_pars[f"a({ov})"]

        # compute trial-wise drift
        v_trial = (true_pars["v_Intercept"]
                   + true_pars["v_z_AttentionW"] * tr["z_AttentionW"]
                   + true_pars[f"v_z_IAW_chart:C(OVcate)[{ov}]"] * tr["z_IAW_chart"]
                + true_pars[f"v_z_IAW_image:C(OVcate)[{ov}]"] * tr["z_IAW_image"]
                )


        trial_df, _ = hddm.generate.gen_rand_data(
            {"v": v_trial, "a": a_val, "t": true_pars["t"]},
            size=1, subjs=1
        )
        # keep predictors & subj_idx for the refit
        for col in ["subj_idx", "OVcate",
                    "z_AttentionW", "z_IAW_chart", "z_IAW_image"]:
            trial_df[col] = tr[col]
        sim_rows.append(trial_df)

    return pd.concat(sim_rows, ignore_index=True)

def refit(sim_df: pd.DataFrame, seed: int) -> az.InferenceData:
    np.random.seed(seed)                         # reproducible chains

    m = hddm.HDDMRegressor(
        sim_df, reg_descr,
        include=['a', 't', 'v'],
        p_outlier=0.05,
        keep_regressor_trace=True,
        group_only_regressors=False,
        depends_on=depends_on,
        is_group_model=True,
    )
    m.find_starting_values()

    # ---------- sample ----------
    m.sample(
        N_SAMPLES,          # draws
        burn=BURN,
        chains=4,
        db     = 'ram',                 # keep everything in memory
        dbname = f'ram_{seed}',         # <- MUST be a string, not None
        progressbar=True,
        ppc=False,    
        loglik=True,
    )
    
    # convert to ArviZ. Works in every HDDM version ≥ 0.8 
    idata = hddm.utils.model_to_inference_data(m, include_ppc=False)
    return idata

def group_means(idata):
    s = az.summary(idata, var_names=PARAM_LIST, stat_funcs=None)
    return s["mean"].to_dict()

# ---------- load empirical fit & raw predictors -------------------------
empirical = az.concat(
    [az.from_netcdf(p) for p in EMPIRICAL_POST_PATHS], dim="chain"
)
raw_df = empirical.observed_data.to_dataframe().reset_index(drop=True)
raw_df["subj_idx"] = raw_df["subj_idx"].astype(int)


# ---------- main loop ---------------------------------------------------
records = []
for rep in trange(N_REPS, desc="parameter-recovery reps", unit="rep"):

    θ_true = extract_group_sample(empirical, seed=rep)       
    sim_df = simulate_dataset(θ_true, raw_df)                

    # call refit **with only a seed** (we removed the tmp-pickle argument)
    idata  = refit(sim_df, seed=10_000 + rep)                
    θ_hat  = group_means(idata)

    for p in PARAM_LIST:
        records.append(
            dict(rep=rep, parameter=p,
                 true=θ_true[p], recovered=θ_hat[p])
        )



print(idata.posterior.data_vars)

# ---------- save CSV & scatter grid -------------------------------------
results = pd.DataFrame(records)
csv_out = FIG_DIR / "true_vs_recovered_ES31.csv"
results.to_csv(csv_out, index=False)
print(f"CSV saved {csv_out}")

sns.set_style("white")
g = sns.FacetGrid(
    results, col="parameter", col_wrap=3,
    sharex=False, sharey=False, height=3.2
)
g.map_dataframe(sns.scatterplot, x="true", y="recovered", s=28, alpha=.85)
for ax in g.axes.ravel():
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--k", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
g.set_axis_labels("true value (one posterior draw)",
                  "posterior mean after refit")
g.tight_layout()
png_out = FIG_DIR / "scatter_ES31.png"
g.savefig(png_out, dpi=300)
print(f"scatter grid saved {png_out}")

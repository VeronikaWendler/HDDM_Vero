# plot_recovery_individual_all.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
FIG_DIR     = PROJECT_DIR / "figures_dir_garcia/garcia_replication_ES_35/recovery_m35"
IN_CSV      = FIG_DIR / "true_vs_recovered_individual.csv"
OUT_PNG     = FIG_DIR / "scatter_individual_ALL_params.png"

df = pd.read_csv(IN_CSV)
# Defensive: drop rows with missing/inf
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["true", "recovered", "parameter"])

if df.empty:
    raise SystemExit("No rows in true_vs_recovered_individual.csv. "
                     "Re-run the main script with the updated extract_individual_means().")

# nice ordering: intercepts/t first, then others alphabetically
priority = ["t", "v_Intercept", "a_Intercept"]
params = sorted(df["parameter"].unique(), key=lambda p: (p not in priority, priority.index(p) if p in priority else 0, p))

sns.set_style("white")
n_params = len(params)
col_wrap = 3 if n_params <= 9 else 4

def facet_scatter(data, color=None, **kws):
    ax = plt.gca()
    x = data["true"].to_numpy()
    y = data["recovered"].to_numpy()
    n = len(x)

    # Scatter
    ax.scatter(x, y, s=18, alpha=0.7)

    # identity line
    xlo, xhi = np.nanmin(x), np.nanmax(x)
    ylo, yhi = np.nanmin(y), np.nanmax(y)
    lo = np.nanmin([xlo, ylo])
    hi = np.nanmax([xhi, yhi])
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="k")

    # OLS fit + stats (guard for tiny n)
    if n >= 3 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and (np.std(x) > 0):
        res = st.linregress(x, y)
        # Regression line
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, res.intercept + res.slope * xs, lw=1.2)

        # Stats
        r2   = res.rvalue**2
        rmse = float(np.sqrt(np.mean((y - (res.intercept + res.slope*x))**2)))
        dfree = max(n - 2, 1)
        tcrit = st.t.ppf(0.975, df=dfree)
        ci_lo = res.slope - tcrit * res.stderr
        ci_hi = res.slope + tcrit * res.stderr

        # Test slope == 1
        if res.stderr > 0:
            t_eq1 = (res.slope - 1.0) / res.stderr
            p_eq1 = 2 * st.t.sf(abs(t_eq1), df=dfree)
        else:
            p_eq1 = np.nan

        # annotate
        txt = (f"R²={r2:.2f}\n"
               f"r={res.rvalue:.2f}, p={res.pvalue:.3g}\n"
               f"β={res.slope:.2f} [{ci_lo:.2f},{ci_hi:.2f}]\n"
               f"p(β=1)={p_eq1:.3g}\n"
               f"RMSE={rmse:.3f}\n"
               f"N={n}")
    else:
        txt = f"N={n}"

    # put the text in the upper-left inside the axes
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))

    ax.set_xlabel("true value")
    ax.set_ylabel("posterior mean (recovered)")

g = sns.FacetGrid(df, col="parameter", col_order=params,
                  col_wrap=col_wrap, sharex=False, sharey=False, height=3.1, despine=True)
g.map_dataframe(facet_scatter)

# tidy axes ranges to same min/max per panel
for ax in g.axes.ravel():
    if ax is None: 
        continue
    # sync to square range around data + identity
    x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
    lo = min(x0,y0); hi = max(x1,y1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

g.figure.subplots_adjust(top=0.92)
g.figure.suptitle("Individual-level parameter recovery (all parameters)")
g.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")

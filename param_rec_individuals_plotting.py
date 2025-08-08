import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
FIG_DIR     = PROJECT_DIR / "figures_dir_garcia/garcia_replication_ES_35/recovery_m35"
IN_CSV      = FIG_DIR / "true_vs_recovered_individual.csv"
OUT_PNG     = FIG_DIR / "scatter_individual_ALL_params.png"

df = pd.read_csv(IN_CSV).replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["true", "recovered", "parameter"])

if df.empty:
    raise SystemExit("The individual CSV is empty. Re-run the main script after updating extract_individual_means().")

# order panels: key params first, then alphabetical
priority = ["t", "v_Intercept", "a_Intercept"]
params = sorted(df["parameter"].unique(), key=lambda p: (p not in priority, priority.index(p) if p in priority else 0, p))

sns.set_style("white")
col_wrap = 3 if len(params) <= 9 else 4

def facet_scatter(data, **k):
    ax = plt.gca()
    x = data["true"].to_numpy()
    y = data["recovered"].to_numpy()
    ax.scatter(x, y, s=18, alpha=0.7)

    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color="k")

    n = len(x)
    if n >= 3 and np.std(x) > 0 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
        res = st.linregress(x, y)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, res.intercept + res.slope*xs, lw=1.2)

        r2   = res.rvalue**2
        rmse = float(np.sqrt(np.mean((y - (res.intercept + res.slope*x))**2)))
        dfree = max(n-2, 1)
        tcrit = st.t.ppf(0.975, df=dfree)
        ci_lo = res.slope - tcrit*res.stderr
        ci_hi = res.slope + tcrit*res.stderr
        p_eq1 = 2*st.t.sf(abs((res.slope-1.0)/res.stderr), df=dfree) if res.stderr>0 else np.nan

        txt = (f"R²={r2:.2f}\n"
               f"r={res.rvalue:.2f}, p={res.pvalue:.3g}\n"
               f"β={res.slope:.2f} [{ci_lo:.2f},{ci_hi:.2f}]\n"
               f"p(β=1)={p_eq1:.3g}\n"
               f"RMSE={rmse:.3f}\n"
               f"N={n}")
    else:
        txt = f"N={n}"

    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))
    # square-ish limits
    x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
    lo2, hi2 = min(x0,y0), max(x1,y1)
    ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2)
    ax.set_xlabel("true value"); ax.set_ylabel("posterior mean (recovered)")

g = sns.FacetGrid(df, col="parameter", col_order=params, col_wrap=col_wrap,
                  sharex=False, sharey=False, height=3.1, despine=True)
g.map_dataframe(facet_scatter)
g.figure.subplots_adjust(top=0.92)
g.figure.suptitle("Individual-level parameter recovery (all parameters)")
g.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")

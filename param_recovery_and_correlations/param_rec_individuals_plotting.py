import os
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

# ---------- paths ----------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
FIG_DIR     = PROJECT_DIR / "figures_dir_OV/OV_replication_For_paper_6/recovery_For_paper_m6"
IN_CSV      = FIG_DIR / "true_vs_recovered_individual2.csv"
OUT_PNG     = FIG_DIR / "scatter_individual_ALL_params_3.png"
OUT_STATS   = FIG_DIR / "scatter_individual_ALL_params_3.csv"

# ---------- load ----------
df = pd.read_csv(IN_CSV).replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["true", "recovered", "parameter"])
if df.empty:
    raise SystemExit("The individual CSV is empty.")

# order panels: key ones first, then alphabetical
priority = ['a(high)',
            'a(low)',
            'a(medium)',
            't',
            'z',
            'v_ES_AttentionW',
            'v_ES_InattentionW_E',
            'v_ES_InattentionW_S']
params = sorted(
    df["parameter"].unique(),
    key=lambda p: (p not in priority, priority.index(p) if p in priority else 0, p)
)

# ---------- stats helper (only R2, p, RMSE) ----------
def regress_stats(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    ok = (n >= 3) and np.isfinite(x).all() and np.isfinite(y).all() and (np.std(x) > 0)
    if not ok:
        return dict(ok=False, N=n, R2=np.nan, p=np.nan, RMSE=np.nan)
    res  = st.linregress(x, y)
    yhat = res.intercept + res.slope * x
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    r2   = float(res.rvalue**2)
    return dict(ok=True, N=n, R2=r2, p=float(res.pvalue), RMSE=rmse)


AX_LIMS = {"t": (0.0, 1.0)}

def _p_text(p):
    if not np.isfinite(p):
        return "p=NA"
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"


def facet_scatter(data, **k):
    ax = plt.gca()
    x = data["true"].to_numpy()
    y = data["recovered"].to_numpy()
    pname = str(data["parameter"].iloc[0])

    # scatter all points (out-of-range ones will be clipped by axis limits)
    ax.scatter(x, y, s=18, alpha=0.7)

    # enforce axes for t and v_Intercept; leave others alone
    if pname in AX_LIMS:
        lo, hi = AX_LIMS[pname]
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        # stats on in-frame points only
        mask = (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
        x_s, y_s = x[mask], y[mask]
    else:
        x_s, y_s = x, y

    # reference line using the final limits
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    lo_line, hi_line = min(x0, y0), max(x1, y1)
    ax.plot([lo_line, hi_line], [lo_line, hi_line], "--", lw=1, color="k")

    s = regress_stats(x_s, y_s)
    txt = (f"R²={s['R2']:.2f}\n{_p_text(s['p'])}\nRMSE={s['RMSE']:.3f}") if s["ok"] else f"N={s['N']}"
    ax.text(
        0.97, 0.03, txt,
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9)
    )
    
    ax.set_xlabel("true value")
    ax.set_ylabel("posterior mean (recovered)")


rows = []
for p in params:
    sub = df[df["parameter"] == p]
    s = regress_stats(sub["true"], sub["recovered"])
    rows.append({"parameter": p, "R2": s["R2"], "p": s["p"], "RMSE": s["RMSE"]})
pd.DataFrame(rows, columns=["parameter","R2","p","RMSE"]).to_csv(OUT_STATS, index=False)

sns.set_style("white")
col_wrap = 3 if len(params) <= 9 else 4


g = sns.FacetGrid(df, col="parameter", col_order=params, col_wrap=col_wrap,
                  sharex=False, sharey=False, height=3.1, despine=True)
g.set_titles("{col_name}")
g.map_dataframe(facet_scatter)
g.figure.subplots_adjust(top=0.92)
g.figure.suptitle("Individual-level parameter recovery (R², p, RMSE)")
g.savefig(OUT_PNG, dpi=300, bbox_inches="tight")

print(f"Saved figure: {OUT_PNG}")
print(f"Saved stats:  {OUT_STATS}")
























# plot_individual_all_params.py
# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd

# import matplotlib; matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats as st

# # ---------- paths ----------
# PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
# FIG_DIR     = PROJECT_DIR / "figures_dir_garcia/garcia_replication_ES_35/recovery_m35"
# IN_CSV      = FIG_DIR / "partial_individual2.csv"
# OUT_PNG     = FIG_DIR / "scatter_individual_ALL_params_6REP.png"
# OUT_STATS   = FIG_DIR / "scatter_individual_ALL_params_stats_6REP.csv"

# # ---------- load ----------
# df = pd.read_csv(IN_CSV).replace([np.inf, -np.inf], np.nan)
# df = df.dropna(subset=["true", "recovered", "parameter"])
# if df.empty:
#     raise SystemExit("The individual CSV is empty.")

# # order: key ones first, then everything else alphabetically
# priority = ["t", "v_Intercept", "a_Intercept"]
# params = sorted(df["parameter"].unique(),
#                 key=lambda p: (p not in priority, priority.index(p) if p in priority else 0, p))

# # ---------- stats helper ----------
# def regress_stats(x, y):
#     x = np.asarray(x); y = np.asarray(y)
#     n = len(x)
#     ok = (n >= 3) and np.isfinite(x).all() and np.isfinite(y).all() and (np.std(x) > 0)
#     out = dict(N=n, ok=ok)
#     if not ok:
#         return {**out, "R2": np.nan, "p": np.nan, "RMSE": np.nan}
    
#     #"r": np.nan, "p": np.nan, "beta": np.nan, "beta_lo": np.nan, "beta_hi": np.nan, "p_beta_eq1": np.nan, "intercept": np.nan,
#     res   = st.linregress(x, y)
#     yhat  = res.intercept + res.slope*x
#     rmse  = float(np.sqrt(np.mean((y - yhat)**2)))
#     r2    = float(res.rvalue**2)
#     dfree = max(n-2, 1)
#     tcrit = st.t.ppf(0.975, df=dfree)
#     beta_lo = res.slope - tcrit*res.stderr
#     beta_hi = res.slope + tcrit*res.stderr
#     p_eq1   = 2*st.t.sf(abs((res.slope-1.0)/res.stderr), df=dfree) if res.stderr>0 else np.nan
#     return dict(R2=r2, p=float(res.pvalue), RMSE=rmse)

# #N=n, ok=True, r=float(res.rvalue), p=float(res.pvalue),beta=float(res.slope), beta_lo=float(beta_lo), beta_hi=float(beta_hi),p_beta_eq1=float(p_eq1), intercept=float(res.intercept),
# # save per-parameter stats
# stats_rows = []
# for p in params:
#     sub = df[df["parameter"] == p]
#     s = regress_stats(sub["true"], sub["recovered"])
#     s["parameter"] = p
#     stats_rows.append(s)
# pd.DataFrame(stats_rows, columns=[
#     "parameter","R2","p","RMSE"
# ]).to_csv(OUT_STATS, index=False)

# #"N","r","beta","beta_lo","beta_hi","p_beta_eq1","intercept",
# # ---------- plotting ----------
# sns.set_style("white")
# col_wrap = 3 if len(params) <= 9 else 4

# def facet_scatter(data, **k):
#     ax = plt.gca()
#     x = data["true"].to_numpy()
#     y = data["recovered"].to_numpy()

#     ax.scatter(x, y, s=18, alpha=0.7)
#     lo = np.nanmin([x.min(), y.min()])
#     hi = np.nanmax([x.max(), y.max()])
#     ax.plot([lo, hi], [lo, hi], "--", lw=1, color="k")

#     s = regress_stats(x, y)
#     if s["ok"]:
#         xs = np.linspace(lo, hi, 100)
#         ax.plot(xs, s["intercept"] + s["beta"]*xs, lw=1.2)
#         txt = (f"R²={s['R2']:.2f}\n"
#                f"r={s['r']:.2f}, p={s['p']:.3g}\n"
#                f"β={s['beta']:.2f} [{s['beta_lo']:.2f},{s['beta_hi']:.2f}]\n"
#                f"p(β=1)={s['p_beta_eq1']:.3g}\n"
#                f"RMSE={s['RMSE']:.3f}\n"
#                f"N={s['N']}")
#     else:
#         txt = f"N={s['N']}"

#     ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9,
#             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))

#     # keep square-ish axes
#     x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
#     lo2, hi2 = min(x0,y0), max(x1,y1)
#     ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2)
#     ax.set_xlabel("true value"); ax.set_ylabel("posterior mean (recovered)")

# g = sns.FacetGrid(df, col="parameter", col_order=params, col_wrap=col_wrap,
#                   sharex=False, sharey=False, height=3.1, despine=True)
# g.map_dataframe(facet_scatter)
# g.figure.subplots_adjust(top=0.92)
# g.figure.suptitle("Individual-level parameter recovery (all parameters present in CSV)")
# g.savefig(OUT_PNG, dpi=300, bbox_inches="tight")

# print(f"Saved figure: {OUT_PNG}")
# print(f"Saved stats:  {OUT_STATS}")

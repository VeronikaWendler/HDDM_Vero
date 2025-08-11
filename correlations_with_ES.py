from pathlib import Path
import os, re, math
import numpy as np
import pandas as pd
import arviz as az
from scipy import stats
import matplotlib.pyplot as plt

# ---------------- paths ----------------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
MODELS_DIR  = PROJECT_DIR / "models_dir_garcia"
M35_DIAG    = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "diagnostics"
OUT_DIR     = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_NETCDFS = [
    MODELS_DIR / "garcia_replication_ES_35_0.nc",
    MODELS_DIR / "garcia_replication_ES_35_1.nc",
    MODELS_DIR / "garcia_replication_ES_35_2.nc",
]

# --------------- helpers ---------------
def _read_results(path):
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.lower() in {"", "unnamed: 0"} or not np.issubdtype(df[first].dtype, np.number):
        df = df.rename(columns={first: "param"})
    if "param" not in df.columns:
        df = df.reset_index().rename(columns={"index": "param"})
    return df

def _extract_all_subject_params(df, central="mean"):
    by_param = {}
    pat = re.compile(r"^(?P<base>.+)_subj\.(?P<sid>\d+)$")
    for _, row in df.iterrows():
        m = pat.match(str(row["param"]))
        if not m:
            continue
        base = m.group("base"); sid = int(m.group("sid"))
        val  = float(row.get(central, row.get("mean")))
        by_param.setdefault(base, {})[sid] = val
    return by_param

def _p_text(p):
    if not np.isfinite(p): return "p=NA"
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"

def compute_es_accuracy_from_netcdf(nc_paths, *, subj_col="subj_idx", correct_col=None):
    """Return dict {subj_id: mean_accuracy} from ES observed_data."""
    idata = az.concat([az.from_netcdf(p) for p in nc_paths], dim="chain")
    df = idata.observed_data.to_dataframe().reset_index(drop=True)
    # find a correctness column if not given
    if correct_col is None:
        for c in ["correct", "accuracy", "corr"]:
            if c in df.columns:
                correct_col = c
                break
    if correct_col is None or subj_col not in df.columns:
        raise ValueError(f"Could not find subj_col='{subj_col}' and/or a correctness column in observed_data.")
    # coerce to 0/1
    corr = pd.to_numeric(df[correct_col], errors="coerce")
    acc_by_subj = pd.Series(corr, index=df[subj_col]).groupby(level=0).mean().dropna()
    # ensure ints for keys
    return {int(k): float(v) for k, v in acc_by_subj.to_dict().items()}

# ---- optional: add θ rows (same as you already have) ----
def add_theta_params_to_results(m35_in_csv, m35_out_csv, use_median=False):
    df = _read_results(m35_in_csv); central = "50q" if use_median else "mean"
    subj_maps = _extract_all_subject_params(df, central=central)
    need = {
        "num_chart":  "v_z_IAW_chart",
        "num_image":  "v_z_IAW_image",
        "den_low":    "v_z_AttentionW:C(OVcate)[low]",
        "den_medium": "v_z_AttentionW:C(OVcate)[medium]",
        "den_high":   "v_z_AttentionW:C(OVcate)[high]",
    }
    for p in need.values():
        if p not in subj_maps:
            raise ValueError(f"Missing '{p}_subj.<id>' in results.")
    combos = [
        ("theta_chart_low",   subj_maps[need["num_chart"]],  subj_maps[need["den_low"]]),
        ("theta_chart_medium",subj_maps[need["num_chart"]],  subj_maps[need["den_medium"]]),
        ("theta_chart_high",  subj_maps[need["num_chart"]],  subj_maps[need["den_high"]]),
        ("theta_image_low",   subj_maps[need["num_image"]],  subj_maps[need["den_low"]]),
        ("theta_image_medium",subj_maps[need["num_image"]],  subj_maps[need["den_medium"]]),
        ("theta_image_high",  subj_maps[need["num_image"]],  subj_maps[need["den_high"]]),
    ]
    new_rows, cols = [], list(df.columns)
    if "param" not in cols: cols = ["param"] + [c for c in cols if c != "param"]
    for base, num_map, den_map in combos:
        for sid in sorted(set(num_map).intersection(den_map)):
            den = den_map[sid]
            if den is None or np.isclose(den, 0.0): continue
            mean_val = num_map[sid] / den
            row = {c: np.nan for c in cols}
            row["param"] = f"{base}_subj.{sid}"
            row["mean"]  = mean_val
            if "50q" in cols: row["50q"] = mean_val
            new_rows.append(row)
    df_out = pd.concat([df, pd.DataFrame(new_rows, columns=cols)], ignore_index=True)
    df_out.to_csv(m35_out_csv, index=False)
    return m35_out_csv

# --------------- plotting ---------------
def plot_accuracy_correlations(es_accuracy, model35_results_csv, out_pdf, out_csv, use_median=False):
    m35 = _read_results(model35_results_csv)
    central = "50q" if use_median else "mean"
    params_by_name = _extract_all_subject_params(m35, central=central)

    panels, rows = [], []
    for pname, subj_map in sorted(params_by_name.items()):
        common = sorted(set(es_accuracy).intersection(subj_map))
        if len(common) < 5:  # skip tiny Ns
            continue
        x = np.array([es_accuracy[s] for s in common])    # ES accuracy
        y = np.array([subj_map[s]   for s in common])     # aDDM subj param

        r, p = stats.pearsonr(x, y)
        r2 = float(r**2)
        b1, b0 = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b1 * x_line + b0

        # 95% CI band for the regression line
        n = len(common)
        y_pred = b1 * x + b0
        resid  = y - y_pred
        s_err  = np.sqrt(np.sum(resid**2) / (n - 2))
        t_val  = stats.t.ppf(0.975, df=n - 2)
        ci = t_val * s_err * np.sqrt(1/n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        y_lo, y_hi = y_line - ci, y_line + ci

        panels.append(dict(name=pname, x=x, y=y, x_line=x_line, y_line=y_line, y_lo=y_lo, y_hi=y_hi, r2=r2, p=p))
        rows.append({"parameter": pname, "n": n, "pearson_r": r, "r2": r2, "p_value": float(p)})

    if not panels:
        raise ValueError("No overlapping subjects found to plot.")

    k = len(panels)
    ncols = 3 if k <= 9 else 4 if k <= 16 else 5
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    ACCENT = "darkmagenta"
    for ax, panel in zip(axes, panels):
        ax.scatter(panel["x"], panel["y"], s=30, color=ACCENT, alpha=0.85, edgecolors="none")
        ax.plot(panel["x_line"], panel["y_line"], color=ACCENT, lw=1.8)
        ax.fill_between(panel["x_line"], panel["y_lo"], panel["y_hi"], color=ACCENT, alpha=0.25, linewidth=0)
        ax.set_xlabel("ES accuracy")
        ax.set_ylabel(panel["name"])
        ax.set_title(panel["name"])
        ax.text(0.02, 0.98, f"R² = {panel['r2']:.3f}\n{_p_text(panel['p'])}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), fontsize=9)
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Correlations: ES accuracy vs aDDM subject-level parameters", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_csv}")

# ---------------- run ----------------
# 1) ES accuracy per subject
es_acc = compute_es_accuracy_from_netcdf(ES_NETCDFS, subj_col="subj_idx", correct_col=None)

# 2) Add θ rows (if not already created) and load aDDM results
m35_in  = (M35_DIAG / "results.csv")
m35_out = (M35_DIAG / "results_plus_theta.csv")
m35_aug = add_theta_params_to_results(m35_in, m35_out, use_median=False)

# 3) Correlate & plot
out_pdf = (OUT_DIR / "es_accuracy_vs_addm_params.pdf")
out_csv = (OUT_DIR / "es_accuracy_vs_addm_params_summary.csv")
plot_accuracy_correlations(es_acc, m35_aug, out_pdf.as_posix(), out_csv.as_posix(), use_median=False)

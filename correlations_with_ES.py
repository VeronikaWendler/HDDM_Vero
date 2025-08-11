from pathlib import Path
import os, re, math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---------------- paths ----------------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

M35_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "diagnostics"
OUT_DIR  = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# primary name; fallback to common misspelling if needed
ACC_CSV = OUT_DIR / "results_ES_accuracy.csv"
if not ACC_CSV.exists():
    alt = OUT_DIR / "reuslts_ES_accuracy.csv"
    if alt.exists():
        ACC_CSV = alt

# ---------------- helpers ----------------
def _read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.lower() in {"", "unnamed: 0"} or not np.issubdtype(df[first].dtype, np.number):
        df = df.rename(columns={first: "param"})
    if "param" not in df.columns:
        df = df.reset_index().rename(columns={"index": "param"})
    return df

def _extract_all_subject_params(df: pd.DataFrame, central="mean"):
    """Return {base_param: {sid: value, ...}, ...} for rows like '<base>_subj.<id>'."""
    by_param = {}
    pat = re.compile(r"^(?P<base>.+)_subj\.(?P<sid>\d+)$")
    for _, row in df.iterrows():
        m = pat.match(str(row["param"]))
        if not m:
            continue
        base = m.group("base")
        sid  = int(m.group("sid"))
        val  = float(row.get(central, row.get("mean")))
        by_param.setdefault(base, {})[sid] = val
    return by_param

def add_theta_params_to_results(m35_in_csv: Path, m35_out_csv: Path, use_median=False) -> Path:
    """Augment Model 35 results with theta_* subject-level params; returns output path."""
    df = _read_results(m35_in_csv)
    central = "50q" if use_median else "mean"
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
            raise ValueError(f"Missing subject-level parameter: {p}_subj.<id>")

    combos = [
        ("theta_chart_low",    subj_maps[need["num_chart"]], subj_maps[need["den_low"]]),
        ("theta_chart_medium", subj_maps[need["num_chart"]], subj_maps[need["den_medium"]]),
        ("theta_chart_high",   subj_maps[need["num_chart"]], subj_maps[need["den_high"]]),
        ("theta_image_low",    subj_maps[need["num_image"]], subj_maps[need["den_low"]]),
        ("theta_image_medium", subj_maps[need["num_image"]], subj_maps[need["den_medium"]]),
        ("theta_image_high",   subj_maps[need["num_image"]], subj_maps[need["den_high"]]),
    ]

    cols = list(df.columns)
    if "param" not in cols:
        cols = ["param"] + [c for c in cols if c != "param"]

    new_rows = []
    for base, num_map, den_map in combos:
        for sid in sorted(set(num_map).intersection(den_map)):
            den = den_map[sid]
            if den is None or np.isclose(den, 0.0):
                continue
            mean_val = num_map[sid] / den
            row = {c: np.nan for c in cols}
            row["param"] = f"{base}_subj.{sid}"
            row["mean"]  = mean_val
            if "50q" in cols:
                row["50q"] = mean_val
            new_rows.append(row)

    if not new_rows:
        raise ValueError("No theta rows were created.")
    df_out = pd.concat([df, pd.DataFrame(new_rows, columns=cols)], ignore_index=True)
    df_out.to_csv(m35_out_csv, index=False)
    print(f"Saved augmented results with θ params -> {m35_out_csv}")
    return m35_out_csv

def _p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"

# ---------------- new: accuracy vs aDDM params ----------------
def read_es_accuracy(path: Path) -> dict:
    """
    Reads ES-phase mean accuracy CSV with columns:
      sub_id (like 'subj.12'), mean_accuracy, std
    Returns {sid_int: mean_accuracy}.
    """
    acc = {}
    df = pd.read_csv(path)
    pat = re.compile(r"^subj\.(\d+)$")
    for _, row in df.iterrows():
        m = pat.match(str(row["sub_id"]))
        if not m:
            continue
        sid = int(m.group(1))
        acc[sid] = float(row["mean_accuracy"])
    if not acc:
        raise ValueError(f"No subject rows parsed from {path}")
    return acc

def plot_accuracy_correlations(
    accuracy_csv: Path,
    model35_results_csv: Path,
    out_pdf: Path,
    out_csv: Path,
    use_median: bool = False
):
    # 1) accuracy per subject
    acc_map = read_es_accuracy(accuracy_csv)  # {sid: accuracy}

    # 2) read all subject-level aDDM params (including theta if present)
    m35 = _read_results(model35_results_csv)
    central = "50q" if use_median else "mean"
    params_by_name = _extract_all_subject_params(m35, central=central)
    if not params_by_name:
        raise ValueError("No '*_subj.<id>' parameters found in model35_results_csv.")

    # 3) assemble panels & CSV stats
    panels, rows = [], []
    for base_name, subj_map in sorted(params_by_name.items()):
        common = sorted(set(acc_map).intersection(subj_map))
        if len(common) < 5:
            continue
        x = np.array([acc_map[s]   for s in common])   # ES accuracy
        y = np.array([subj_map[s]  for s in common])   # parameter value

        r, p = stats.pearsonr(x, y)
        r2   = float(r**2)
        n    = len(common)

        # OLS line + 95% CI
        b1, b0 = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b1 * x_line + b0
        y_pred = b1 * x + b0
        resid  = y - y_pred
        s_err  = np.sqrt(np.sum(resid**2) / max(n - 2, 1))
        t_val  = stats.t.ppf(0.975, df=max(n - 2, 1))
        ci     = t_val * s_err * np.sqrt(1/n + (x_line - x.mean())**2 / np.sum((x - x.mean())**2))
        y_lo, y_hi = y_line - ci, y_line + ci

        panels.append(dict(name=base_name, x=x, y=y,
                           x_line=x_line, y_line=y_line, y_lo=y_lo, y_hi=y_hi,
                           r2=r2, p=p))

        rows.append({"parameter": base_name, "n": n, "pearson_r": r, "r2": r2, "p_value": float(p)})

    if not panels:
        raise ValueError("Nothing to plot (no parameters with >=5 overlapping subjects).")

    # 4) single-page grid PDF
    k = len(panels)
    ncols = 3 if k <= 9 else 4 if k <= 16 else 5
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    ACCENT = "lightseagreen"
    for ax, panel in zip(axes, panels):
        ax.scatter(panel["x"], panel["y"], s=30, color=ACCENT, alpha=0.85, edgecolors="none")
        ax.plot(panel["x_line"], panel["y_line"], color=ACCENT, lw=1.8)
        ax.fill_between(panel["x_line"], panel["y_lo"], panel["y_hi"], color=ACCENT, alpha=0.25, linewidth=0)

        ax.set_xlabel("Mean accuracy (ES phase)")
        ax.set_ylabel(panel["name"])
        ax.set_title(panel["name"])

        # stats box top-left, no "N="
        txt = f"R² = {panel['r2']:.3f}\n{_p_text(panel['p'])}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), fontsize=9)

        # Optional: keep a bit of padding
        ax.margins(0.05)

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Correlations: ES accuracy vs subject-level aDDM parameters", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5) CSV summary
    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved CSV: {out_csv}")

# ---------------- run ----------------
# Augment Model 35 results with theta_* params (once)
m35_in      = M35_DIAG / "results.csv"
m35_plus    = M35_DIAG / "results_plus_theta.csv"
m35_aug_csv = add_theta_params_to_results(m35_in, m35_plus, use_median=False)

# Plot accuracy vs all aDDM params
out_pdf = OUT_DIR / "accuracy_vs_params_with_theta.pdf"
out_csv = OUT_DIR / "accuracy_vs_params_with_theta_summary.csv"

plot_accuracy_correlations(
    accuracy_csv=ACC_CSV,
    model35_results_csv=m35_aug_csv,
    out_pdf=out_pdf,
    out_csv=out_csv,
    use_median=False
)

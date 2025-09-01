from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import math

# Paths #
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

M35_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_VAL_36" / "diagnostics"
RL1_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_LE_RL_1" / "diagnostics"
OUT_DIR  = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_VAL_36" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def _read_results(path):
    df = pd.read_csv(path)
    # ensure parameter column is present
    first = df.columns[0]
    if first.lower() in {"", "unnamed: 0"} or not np.issubdtype(df[first].dtype, np.number):
        df = df.rename(columns={first: "param"})
    if "param" not in df.columns:
        df = df.reset_index().rename(columns={"index": "param"})
    return df

def _extract_all_subject_params(df, central="mean"):
    by_param = {}
    # Updated regex to capture things like a_subj(high).12
    pat = re.compile(r"^(?P<base>.+)_subj(?:\((?P<mod>.+?)\))?\.(?P<sid>\d+)$")
    for _, row in df.iterrows():
        m = pat.match(str(row["param"]))
        if not m:
            continue
        base = m.group("base")
        mod = m.group("mod")
        sid  = int(m.group("sid"))
        val  = float(row.get(central, row.get("mean")))

        if mod:
            full_name = f"{base}({mod})"
        else:
            full_name = base

        by_param.setdefault(full_name, {})[sid] = val
    return by_param

def inv_logit(x):
    return np.exp(x) / (1 + np.exp(x))

def add_theta_params_to_results(m35_in_csv, m35_out_csv, use_median=False):
    df = _read_results(m35_in_csv)
    central = "50q" if use_median else "mean"

    # All subj-level mappings
    subj_maps = _extract_all_subject_params(df, central=central)

    need = {
        "num_v_ES_InattentionW_S":  "v_ES_InattentionW_S",
        "num_v_ES_InattentionW_E":  "v_ES_InattentionW_E",
        "num_v_ES_AttentionW":      "v_ES_AttentionW"
    }
    for k, p in need.items():
        if p not in subj_maps:
            raise ValueError(f"Missing subject-level parameter in results: '{p}_subj.<id>'")

    num_InatWS  = subj_maps[need["num_v_ES_InattentionW_S"]]
    num_InatWE  = subj_maps[need["num_v_ES_InattentionW_E"]]
    att         = subj_maps[need["num_v_ES_AttentionW"]]

    combos = [
        ("theta_InatWS", num_InatWS, att),
        ("theta_InatWE", num_InatWE, att),
    ]

    new_rows = []
    cols = list(df.columns)
    if "param" not in cols:
        cols = ["param"] + [c for c in cols if c != "param"]

    for base, num_map, den_map in combos:
        common = sorted(set(num_map).intersection(den_map))
        for sid in common:
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

    df_theta = pd.DataFrame(new_rows, columns=cols)
    df_out = pd.concat([df, df_theta], ignore_index=True)
    df_out.to_csv(m35_out_csv, index=False)
    print(f"Saved augmented results with theta params in {m35_out_csv}")
    return m35_out_csv

def _p_text(p):
    if not np.isfinite(p):
        return "p=NA"
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"

# ---------------- plotting functions ----------------
def plot_alpha_correlations(
    rl_results_csv,
    model35_results_csv,
    out_pdf="alpha_param_correlations.pdf",
    out_csv="alpha_param_correlations_summary.csv",
    use_median=False,
    transform_alpha_if_needed=False
):
    rl = _read_results(rl_results_csv)
    m35 = _read_results(model35_results_csv)
    central = "50q" if use_median else "mean"

    # alpha_subj
    alpha_subj = {}
    pat_alpha = re.compile(r"^alpha_subj\.(\d+)$")
    for _, row in rl.iterrows():
        m = pat_alpha.match(str(row["param"]))
        if m:
            sid = int(m.group(1))
            val = float(row.get(central, row.get("mean")))
            alpha_subj[sid] = val
    if not alpha_subj:
        raise ValueError("No alpha_subj.* rows found in the RL results.")
    if transform_alpha_if_needed:
        alpha_subj = {sid: inv_logit(val) for sid, val in alpha_subj.items()}

    # subj-level params from the HDDM model
    params_by_name = _extract_all_subject_params(m35, central=central)
    
    rows, panels = [], []
    for base_name, subj_map in sorted(params_by_name.items()):
        common = sorted(set(alpha_subj).intersection(subj_map))
        if len(common) < 5:
            continue

        x = np.array([alpha_subj[s] for s in common])
        y = np.array([subj_map[s]   for s in common])

        r, p = stats.pearsonr(x, y)
        r2 = float(r**2)

        # regression fit + CI
        n = len(common)
        b1, b0 = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b1 * x_line + b0
        y_pred = b1 * x + b0
        resid  = y - y_pred
        s_err  = np.sqrt(np.sum(resid**2) / (n - 2))
        t_val  = stats.t.ppf(0.975, df=n - 2)
        ci = t_val * s_err * np.sqrt(1/n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        y_lo = y_line - ci
        y_hi = y_line + ci

        panels.append(dict(
            name=base_name, x=x, y=y, x_line=x_line, y_line=y_line,
            y_lo=y_lo, y_hi=y_hi, r2=r2, p=p, n=n
        ))

        rows.append({"parameter": base_name, "n": n, "pearson_r": r, "r2": r2, "p_value": float(p)})

    if not panels:
        raise ValueError("Nothing to plot (no parameters with >=5 overlapping subjects).")

    ncols = 3 if len(panels) <= 9 else 4 if len(panels) <= 16 else 5
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    ACCENT = "darksalmon"
    for ax, panel in zip(axes, panels):
        ax.scatter(panel["x"], panel["y"], s=30, color=ACCENT, alpha=0.85, edgecolors="none")
        ax.plot(panel["x_line"], panel["y_line"], color=ACCENT, lw=1.8)
        ax.fill_between(panel["x_line"], panel["y_lo"], panel["y_hi"], color=ACCENT, alpha=0.25, linewidth=0)

        ax.set_xlabel("α (learning rate)")
        ax.set_ylabel(panel["name"])
        ax.set_title(panel["name"])

        txt = f"R² = {panel['r2']:.3f}\n{_p_text(panel['p'])}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), fontsize=9)

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Correlations: α vs subject-level parameters", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved: {out_pdf}, {out_csv}")

def plot_alpha_correlations_with_split(
    rl_results_csv,
    model35_results_csv,
    out_pdf="alpha_param_correlations_split.pdf",
    out_csv="alpha_param_correlations_split_summary.csv",
    out_split_csv="alpha_median_split.csv",
    use_median=False,
    transform_alpha_if_needed=False
):
    rl = _read_results(rl_results_csv)
    m35 = _read_results(model35_results_csv)
    central = "50q" if use_median else "mean"

    # alpha_subj
    alpha_subj = {}
    pat_alpha = re.compile(r"^alpha_subj\.(\d+)$")
    for _, row in rl.iterrows():
        m = pat_alpha.match(str(row["param"]))
        if m:
            sid = int(m.group(1))
            val = float(row.get(central, row.get("mean")))
            alpha_subj[sid] = val
    if not alpha_subj:
        raise ValueError("No alpha_subj.* rows found in the RL results.")
    if transform_alpha_if_needed:
        alpha_subj = {sid: inv_logit(val) for sid, val in alpha_subj.items()}

    # median split
    median_val = np.median(list(alpha_subj.values()))
    group_map = {sid: ("poor" if val <= median_val else "good") for sid, val in alpha_subj.items()}
    pd.DataFrame([{"subj": sid, "alpha": val, "group": group_map[sid]} for sid, val in alpha_subj.items()])\
        .to_csv(out_split_csv, index=False)
    print(f"Saved alpha median split info: {out_split_csv}")

    # subj-level params from the HDDM model
    params_by_name = _extract_all_subject_params(m35, central=central)

    rows, panels = [], []
    for base_name, subj_map in sorted(params_by_name.items()):
        common = sorted(set(alpha_subj).intersection(subj_map))
        if len(common) < 5:
            continue

        x = np.array([alpha_subj[s] for s in common])
        y = np.array([subj_map[s]   for s in common])
        groups = [group_map[s] for s in common]

        r, p = stats.pearsonr(x, y)
        r2 = float(r**2)

        # regression fit + CI
        n = len(common)
        b1, b0 = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b1 * x_line + b0
        y_pred = b1 * x + b0
        resid  = y - y_pred
        s_err  = np.sqrt(np.sum(resid**2) / (n - 2))
        t_val  = stats.t.ppf(0.975, df=n - 2)
        ci = t_val * s_err * np.sqrt(1/n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        y_lo = y_line - ci
        y_hi = y_line + ci

        panels.append(dict(
            name=base_name, x=x, y=y, groups=groups,
            x_line=x_line, y_line=y_line, y_lo=y_lo, y_hi=y_hi,
            r2=r2, p=p, n=n
        ))

        rows.append({"parameter": base_name, "n": n, "pearson_r": r, "r2": r2, "p_value": float(p)})

    if not panels:
        raise ValueError("Nothing to plot (no parameters with >=5 overlapping subjects).")

    ncols = 3 if len(panels) <= 9 else 4 if len(panels) <= 16 else 5
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    colors = {"good": "steelblue", "poor": "darkorange"}
    for ax, panel in zip(axes, panels):
        for xi, yi, g in zip(panel["x"], panel["y"], panel["groups"]):
            ax.scatter(xi, yi, s=40, color=colors[g], alpha=0.85, edgecolors="none")
        ax.plot(panel["x_line"], panel["y_line"], color="black", lw=1.8)
        ax.fill_between(panel["x_line"], panel["y_lo"], panel["y_hi"], color="gray", alpha=0.25, linewidth=0)

        ax.set_xlabel("α (learning rate)")
        ax.set_ylabel(panel["name"])
        ax.set_title(panel["name"])

        txt = f"R² = {panel['r2']:.3f}\n{_p_text(panel['p'])}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), fontsize=9)

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Correlations: α vs subject-level parameters (median split)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved: {out_pdf}, {out_csv}, {out_split_csv}")

# ---------- run ----------
rl_results_csv = (RL1_DIAG / "results_alpha_transformed.csv").as_posix()
m35_in_csv     = (M35_DIAG / "results.csv").as_posix()
m35_plus_csv   = (M35_DIAG / "results_plus_theta.csv").as_posix()

m35_aug = add_theta_params_to_results(m35_in_csv, m35_plus_csv, use_median=False)

# Original correlation plots
plot_alpha_correlations(
    rl_results_csv=rl_results_csv,
    model35_results_csv=m35_aug,
    out_pdf=OUT_DIR / "alpha_param_correlations_with_theta.pdf",
    out_csv=OUT_DIR / "alpha_param_correlations_with_theta_summary.csv",
    use_median=False,
    transform_alpha_if_needed=False
)

# New version with median split
plot_alpha_correlations_with_split(
    rl_results_csv=rl_results_csv,
    model35_results_csv=m35_aug,
    out_pdf=OUT_DIR / "alpha_param_correlations_with_theta_split.pdf",
    out_csv=OUT_DIR / "alpha_param_correlations_with_theta_split_summary.csv",
    out_split_csv=OUT_DIR / "alpha_median_split.csv",
    use_median=False,
    transform_alpha_if_needed=False
)

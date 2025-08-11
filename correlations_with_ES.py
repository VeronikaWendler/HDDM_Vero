from pathlib import Path
import os, re, math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---------- base paths ----------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()
M35_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "diagnostics"
OUT_DIR  = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers already used in your pipeline ----------
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
        base = m.group("base")
        sid  = int(m.group("sid"))
        val  = float(row.get(central, row.get("mean")))
        by_param.setdefault(base, {})[sid] = val
    return by_param

def _p_text(p):
    if not np.isfinite(p):
        return "p=NA"
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"

# ---------- new: ES accuracy from the big CSV ----------
def _find_first(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _auto_phase_col(df):
    # Try likely names first
    candidates = ["phase", "task_phase", "experiment_phase", "stage", "block", "block_type"]
    col = _find_first(df, candidates)
    if col is not None:
        return col
    # Fallback: any string-like column that contains "ES" values
    for c in df.columns:
        s = df[c].astype(str).str.upper()
        if s.isna().all():
            continue
        if s.str.contains("ES").any():
            return c
    raise ValueError("Couldn't find a column that marks ES trials. Please set phase_col explicitly.")

def compute_es_accuracy(behav_csv, subj_col=None, phase_col=None, corr_col="corr"):
    beh = pd.read_csv(behav_csv)
    # subject column
    if subj_col is None:
        subj_col = _find_first(beh, ["subj_idx","subj","subject","participant","id","ppid","participant_id"])
    if subj_col is None:
        raise ValueError("Couldn't find a subject ID column (e.g., subj_idx/subject). Pass subj_col=...")

    # phase column
    if phase_col is None:
        phase_col = _auto_phase_col(beh)

    # corr column (0/1)
    corr_match = _find_first(beh, [corr_col]) or _find_first(beh, ["corr","correct"])
    if corr_match is None:
        raise ValueError("Couldn't find correctness column ('corr').")

    # ES mask (accept anything with 'ES' in the value)
    es_mask = beh[phase_col].astype(str).str.upper().str.contains("ES")

    df_es = beh.loc[es_mask, [subj_col, corr_match]].copy()
    df_es[corr_match] = pd.to_numeric(df_es[corr_match], errors="coerce")

    acc = df_es.groupby(subj_col)[corr_match].mean()        # mean accuracy
    ntr = df_es.groupby(subj_col)[corr_match].count()       # ES trial count

    # cast subject IDs to int where possible (to match results.csv)
    def _maybe_int(x):
        try: return int(x)
        except: return np.nan
    idx_int = acc.index.to_series().map(_maybe_int)
    keep = idx_int.notna()
    acc  = pd.Series(acc.values[keep.values], index=idx_int[keep].astype(int))
    ntr  = pd.Series(ntr.values[keep.values], index=idx_int[keep].astype(int))
    return acc.to_dict(), ntr.to_dict()

# ---------- new: plot ES accuracy vs aDDM params ----------
def plot_es_accuracy_vs_addm(
    behav_csv,
    model35_results_csv,
    out_pdf="accuracy_vs_addm_params.pdf",
    out_csv="accuracy_vs_addm_params_summary.csv",
    subj_col=None,
    phase_col=None,
    use_median=False
):
    # ES accuracy
    acc_by_sid, n_by_sid = compute_es_accuracy(behav_csv, subj_col=subj_col, phase_col=phase_col)

    # aDDM subject-level params
    m35 = _read_results(model35_results_csv)
    params_by_name = _extract_all_subject_params(m35, central=("50q" if use_median else "mean"))
    if not params_by_name:
        raise ValueError("No '*_subj.<id>' parameters found in model35_results_csv.")

    rows = []
    panels = []
    for base_name, subj_map in sorted(params_by_name.items()):
        common = sorted(set(acc_by_sid).intersection(subj_map))
        # keep only subjects that actually have ES trials
        common = [s for s in common if n_by_sid.get(s, 0) > 0]
        if len(common) < 5:
            continue

        x = np.array([acc_by_sid[s] for s in common])  # ES accuracy
        y = np.array([subj_map[s]   for s in common])

        r, p = stats.pearsonr(x, y)
        r2 = float(r**2)

        # OLS + CI
        n = len(common)
        b1, b0 = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b1*x_line + b0
        y_pred = b1*x + b0
        resid  = y - y_pred
        s_err  = np.sqrt(np.sum(resid**2) / max(n - 2, 1))
        t_val  = stats.t.ppf(0.975, df=max(n - 2, 1))
        ci     = t_val * s_err * np.sqrt(1/n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        y_lo, y_hi = y_line - ci, y_line + ci

        panels.append(dict(name=base_name, x=x, y=y, x_line=x_line, y_line=y_line,
                           y_lo=y_lo, y_hi=y_hi, r2=r2, p=p))
        rows.append({"parameter": base_name, "n": n, "pearson_r": float(r), "r2": r2, "p_value": float(p)})

    if not panels:
        raise ValueError("No parameters had >=5 overlapping subjects with ES accuracy.")

    # grid figure (one page)
    k = len(panels)
    ncols = 3 if k <= 9 else 4 if k <= 16 else 5
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    ACCENT = "darksalmon"
    for ax, p in zip(axes, panels):
        ax.scatter(p["x"], p["y"], s=30, color=ACCENT, alpha=0.85, edgecolors="none")
        ax.plot(p["x_line"], p["y_line"], color=ACCENT, lw=1.8)
        ax.fill_between(p["x_line"], p["y_lo"], p["y_hi"], color=ACCENT, alpha=0.25, linewidth=0)

        ax.set_xlabel("Accuracy (ES phase)")
        ax.set_ylabel(p["name"])
        ax.set_title(p["name"])
        ax.text(0.02, 0.98, f"R² = {p['r2']:.3f}\n{_p_text(p['p'])}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), fontsize=9)

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Correlations: ES accuracy vs subject-level aDDM parameters", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_csv}")

# ---------- run ----------
# Your raw behavioral CSV (Windows path; change if needed on your system)
BEHAV_CSV = pd.read_csv((PROJECT_DIR / "data_sets" / "data_sets_Garcia" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv").as_posix(), sep=",")       

MODEL35_RESULTS = (M35_DIAG / "results_plus_theta.csv")  # or M35_DIAG / "results.csv"

out_pdf = (OUT_DIR / "accuracy_vs_addm_params.pdf").as_posix()
out_csv = (OUT_DIR / "accuracy_vs_addm_params_summary.csv").as_posix()

plot_es_accuracy_vs_addm(
    behav_csv=BEHAV_CSV,
    model35_results_csv=MODEL35_RESULTS,
    out_pdf=out_pdf,
    out_csv=out_csv,
    subj_col=None,       # set to e.g. "subj_idx" if auto-detect fails
    phase_col=None,      # set to the exact ES phase column if auto-detect fails
    use_median=False
)

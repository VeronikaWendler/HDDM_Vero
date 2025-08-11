from pathlib import Path
import os
import re
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ---------------- paths ----------------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

M35_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "diagnostics"
OUT_DIR  = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# inputs
M35_RESULTS_CSV = (M35_DIAG / "results.csv").as_posix()
ES_ACC_CSV      = (OUT_DIR / "results_ES_accuracy.csv").as_posix()

# outputs
OUT_PDF = (OUT_DIR / "es_accuracy_vs_addm_params_with_theta.pdf").as_posix()
OUT_SUM = (OUT_DIR / "es_accuracy_vs_addm_params_with_theta_summary.csv").as_posix()
M35_PLUS = (M35_DIAG / "results_plus_theta.csv").as_posix()

# --------------- utilities ---------------
def _read_results(path):
    """Read HDDM/Kabuki results; ensure a 'param' column is present."""
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.lower() in {"", "unnamed: 0"} or not np.issubdtype(df[first].dtype, np.number):
        df = df.rename(columns={first: "param"})
    if "param" not in df.columns:
        df = df.reset_index().rename(columns={"index": "param"})
    return df

def _extract_all_subject_params(df, central="mean"):
    """Return {base_param: {sid: value}} for rows like 'base_subj.<id>'."""
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

def add_theta_params_to_results(m35_in_csv, m35_out_csv, use_median=False):
    """
    Add six θ parameters (chart/image over OVcate low/medium/high) at subject level.
    Saves augmented CSV and returns its path.
    """
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
            raise ValueError(f"Missing subject-level parameter in results: '{p}_subj.<id>'")

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

    if not new_rows:
        raise ValueError("No θ rows created—check denominators are present and non-zero.")

    df_out = pd.concat([df, pd.DataFrame(new_rows, columns=cols)], ignore_index=True)
    df_out.to_csv(m35_out_csv, index=False)
    print(f"Saved augmented results with θ params -> {m35_out_csv}")
    return m35_out_csv

def _p_text(p):
    if not np.isfinite(p):
        return "p=NA"
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"

def _load_es_accuracy(acc_csv):
    """
    Read ES accuracy CSV with columns like: sub_id, mean_accuracy, std
    sub_id can be 'subj.1' or '1'. Returns {sid:int -> mean_accuracy:float}.
    """
    acc = pd.read_csv(acc_csv)
    if "sub_id" not in acc.columns or "mean_accuracy" not in acc.columns:
        raise ValueError("ES accuracy file must have columns 'sub_id' and 'mean_accuracy'.")

    def _to_sid(x):
        s = str(x)
        m = re.search(r"(\d+)$", s)  # grabs trailing number (handles 'subj.1')
        if not m:
            raise ValueError(f"Could not parse subject id from '{s}'")
        return int(m.group(1))

    acc["sid"] = acc["sub_id"].apply(_to_sid)
    return dict(zip(acc["sid"].astype(int), acc["mean_accuracy"].astype(float)))

# --------------- main plotting ---------------
def plot_esacc_vs_addm_params(
    es_acc_csv,
    model35_results_csv,
    out_pdf,
    out_csv,
    use_median=False
):
    # load ES accuracy
    acc_map = _load_es_accuracy(es_acc_csv)

    # load HDDM params (with θ already added) and extract all subj-level params
    m35 = _read_results(model35_results_csv)
    central = "50q" if use_median else "mean"
    params_by_name = _extract_all_subject_params(m35, central=central)
    if not params_by_name:
        raise ValueError("No '*_subj.<id>' parameters found in model35_results_csv.")

    # build panels + summary rows
    panels, rows = [], []
    for base_name, subj_map in sorted(params_by_name.items()):
        common = sorted(set(acc_map).intersection(subj_map))
        if len(common) < 5:
            continue

        x = np.array([acc_map[s] for s in common])       # ES accuracy
        y = np.array([subj_map[s] for s in common])      # aDDM param

        r, p = stats.pearsonr(x, y)
        r2   = float(r**2)
        n    = len(common)

        # OLS line + 95% CI band
        b1, b0 = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b1 * x_line + b0
        y_pred = b1 * x + b0
        resid  = y - y_pred
        s_err  = np.sqrt(np.sum(resid**2) / max(n - 2, 1))
        t_val  = stats.t.ppf(0.975, df=max(n - 2, 1))
        ci     = t_val * s_err * np.sqrt(1/n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        y_lo   = y_line - ci
        y_hi   = y_line + ci

        panels.append(dict(
            name=base_name, x=x, y=y, x_line=x_line, y_line=y_line, y_lo=y_lo, y_hi=y_hi,
            r2=r2, p=p
        ))

        rows.append({
            "parameter": base_name,
            "n": n,
            "pearson_r": float(r),
            "r2": r2,
            "p_value": float(p),
        })

    if not panels:
        raise ValueError("Nothing to plot (no parameters with >= 5 overlapping subjects).")

    # grid size
    k = len(panels)
    ncols = 3 if k <= 9 else 4 if k <= 16 else 5
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    ACCENT

from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



# Base paths
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

M35_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "diagnostics"
RL1_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_LE_RL_1" / "diagnostics"  
OUT_DIR = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def _read_results(path):
    """Reads an HDDM/Kabuki results.csv that may have the param in index or an unnamed first col."""
    df = pd.read_csv(path)
    # try to find the parameter name column
    if df.columns[0].lower() in {"", "unnamed: 0"} or not np.issubdtype(df[df.columns[0]].dtype, np.number):
        df = df.rename(columns={df.columns[0]: "param"})
    if "param" not in df.columns:
        # parameter names are likely the index
        df = df.reset_index().rename(columns={"index": "param"})
    return df

def _extract_subj(df, prefix):
    """Return dict {subj_id: value} for rows like 'prefix_subj.<id>' using the 'mean' column by default."""
    out = {}
    pat = re.compile(rf"^{re.escape(prefix)}_subj\.(\d+)$")
    for _, row in df.iterrows():
        m = pat.match(str(row["param"]))
        if m:
            sid = int(m.group(1))
            out[sid] = float(row.get("mean", row.get("50q")))
    return out

# 1) allow central column selection
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


def inv_logit(x):
    return np.exp(x) / (1 + np.exp(x))



def plot_alpha_correlations(
    rl_results_csv,
    model35_results_csv,
    out_pdf="alpha_param_correlations.pdf",
    out_csv="alpha_param_correlations_summary.csv",
    alpha_name="alpha",                # row name for group alpha 
    use_median=False,                  # cn check median later
    transform_alpha_if_needed=False   
):
    # Load
    rl = _read_results(rl_results_csv)
    m35 = _read_results(model35_results_csv)

    # Choose central tendency column
    central = "50q" if use_median else "mean"

    # --- Get subject-level alpha ---
    alpha_subj = {}
    pat_alpha = re.compile(r"^alpha_subj\.(\d+)$")
    for _, row in rl.iterrows():
        m = pat_alpha.match(str(row["param"]))
        if m:
            sid = int(m.group(1))
            val = float(row.get(central, row.get("mean")))
            alpha_subj[sid] = val

    if not alpha_subj:
        raise ValueError("No alpha_subj.* rows found in the RL results. Check rl_results_csv.")

    # Optional inverse-logit 
    if transform_alpha_if_needed:
        alpha_subj = {sid: inv_logit(val) for sid, val in alpha_subj.items()}

    # --- Get ALL subject-level params from model 35 ---
    params_by_name = _extract_all_subject_params(m35)
    if not params_by_name:
        raise ValueError("No '*_subj.<id>' parameters found in model35_results_csv.")

    # Prepare output summary
    rows = []

    # PDF with one page per param
    with PdfPages(out_pdf) as pdf:
        for base_name, subj_map in sorted(params_by_name.items()):
            # Match subjects present in both alpha and this param
            common = sorted(set(alpha_subj).intersection(subj_map))
            if len(common) < 5:
                # Skip tiny Ns to avoid noisy stats
                continue

            x = np.array([alpha_subj[s] for s in common])
            y = np.array([subj_map[s]   for s in common])

            # Pearson correlation
            r, p = stats.pearsonr(x, y)
            r2 = r**2

            # Simple OLS line for plotting
            n = len(common)
            b1, b0 = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = b1 * x_line + b0
            
            # Calculate 95% CI for regression line
            y_pred = b1 * x + b0
            resid = y - y_pred
            s_err = np.sqrt(np.sum(resid**2) / (n - 2))
            t_val = stats.t.ppf(0.975, df=n - 2)
            
            ci = t_val * s_err * np.sqrt(
                1/n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2)
            )
            y_line_lower = y_line - ci
            y_line_upper = y_line + ci
            
            # --- Plot ---
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x, y)
            ax.plot(x_line, y_line, color="blue")
            ax.fill_between(x_line, y_line_lower, y_line_upper, color="blue", alpha=0.2)  # CI shading
            ax.set_xlabel("α (learning rate)")
            ax.set_ylabel(base_name)
            ax.set_title(base_name)
            ax.text(0.02, 0.98, f"R² = {r2:.3f}\np = {p:.3g}\nN = {n}",
                    transform=ax.transAxes, va="top", ha="left")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            
            rows.append({
                "parameter": base_name,
                "n": len(common),
                "pearson_r": r,
                "r2": r2,
                "p_value": p
            })

    summ = pd.DataFrame(rows).sort_values("r2", ascending=False)
    summ.to_csv(out_csv, index=False)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_csv}")

# Files
rl_results_csv = (RL1_DIAG / "results_alpha_transformed.csv").as_posix()
model35_results_csv = (M35_DIAG / "results.csv").as_posix()
out_pdf = (OUT_DIR / "alpha_param_correlations_trans.pdf").as_posix()
out_csv = (OUT_DIR / "alpha_param_correlations_summary_trans.csv").as_posix()

# Run the correlation plotting
plot_alpha_correlations(
    rl_results_csv=rl_results_csv,
    model35_results_csv=model35_results_csv,
    out_pdf=out_pdf,
    out_csv=out_csv,
    use_median=False,
    transform_alpha_if_needed=False  
)

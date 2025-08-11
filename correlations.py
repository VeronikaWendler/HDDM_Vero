from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

# Paths
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

M35_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "diagnostics"
RL1_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_LE_RL_1" / "diagnostics"
OUT_DIR  = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def add_theta_params_to_results(m35_in_csv, m35_out_csv, use_median=False):
    """
    Create six new subject-level params:
      theta_chart_low_subj.<id>    = v_z_IAW_chart_subj.<id>  / v_z_AttentionW:C(OVcate)[low]_subj.<id>
      theta_chart_medium_subj.<id> = v_z_IAW_chart_subj.<id>  / v_z_AttentionW:C(OVcate)[medium]_subj.<id>
      theta_chart_high_subj.<id>   = v_z_IAW_chart_subj.<id>  / v_z_AttentionW:C(OVcate)[high]_subj.<id>
      theta_image_low_subj.<id>    = v_z_IAW_image_subj.<id>  / v_z_AttentionW:C(OVcate)[low]_subj.<id>
      theta_image_medium_subj.<id> = v_z_IAW_image_subj.<id>  / v_z_AttentionW:C(OVcate)[medium]_subj.<id>
      theta_image_high_subj.<id>   = v_z_IAW_image_subj.<id>  / v_z_AttentionW:C(OVcate)[high]_subj.<id>
    Saves a new CSV with the extra rows added (other stats left NaN; mean is populated).
    """
    df = _read_results(m35_in_csv)
    central = "50q" if use_median else "mean"

    # All subj-level mappings
    subj_maps = _extract_all_subject_params(df, central=central)

    need = {
        "num_chart":  "v_z_IAW_chart",
        "num_image":  "v_z_IAW_image",
        "den_low":    "v_z_AttentionW:C(OVcate)[low]",
        "den_medium": "v_z_AttentionW:C(OVcate)[medium]",
        "den_high":   "v_z_AttentionW:C(OVcate)[high]",
    }
    for k, p in need.items():
        if p not in subj_maps:
            raise ValueError(f"Missing subject-level parameter in results: '{p}_subj.<id>'")

    num_chart  = subj_maps[need["num_chart"]]
    num_image  = subj_maps[need["num_image"]]
    den_low    = subj_maps[need["den_low"]]
    den_medium = subj_maps[need["den_medium"]]
    den_high   = subj_maps[need["den_high"]]

    combos = [
        ("theta_chart_low",   num_chart, den_low),
        ("theta_chart_medium",num_chart, den_medium),
        ("theta_chart_high",  num_chart, den_high),
        ("theta_image_low",   num_image, den_low),
        ("theta_image_medium",num_image, den_medium),
        ("theta_image_high",  num_image, den_high),
    ]

    # Prepare rows with same columns as df; fill non-mean stats with NaN
    new_rows = []
    cols = list(df.columns)
    if "param" not in cols:
        cols = ["param"] + [c for c in cols if c != "param"]

    for base, num_map, den_map in combos:
        common = sorted(set(num_map).intersection(den_map))
        for sid in common:
            den = den_map[sid]
            if den is None or np.isclose(den, 0.0):
                continue  # avoid div by zero
            mean_val = num_map[sid] / den
            row = {c: np.nan for c in cols}
            row["param"] = f"{base}_subj.{sid}"
            row["mean"]  = mean_val
            # optionally mirror into 50q for convenience if present
            if "50q" in cols:
                row["50q"] = mean_val
            new_rows.append(row)

    if not new_rows:
        raise ValueError("No theta rows were created — check that denominators exist and are not ~0.")

    df_theta = pd.DataFrame(new_rows, columns=cols)
    df_out = pd.concat([df, df_theta], ignore_index=True)
    df_out.to_csv(m35_out_csv, index=False)
    print(f"Saved augmented results with θ params -> {m35_out_csv}")
    return m35_out_csv


def _p_text(p):
    if not np.isfinite(p):
        return "p=NA"
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"

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

    # alpha_subj.<id>
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
    if not params_by_name:
        raise ValueError("No '*_subj.<id>' parameters found in model35_results_csv.")

    # Prepare data for panels + CSV
    rows = []
    panels = []  # list of dicts with x, y, name, fit, etc.

    for base_name, subj_map in sorted(params_by_name.items()):
        common = sorted(set(alpha_subj).intersection(subj_map))
        if len(common) < 5:
            continue

        x = np.array([alpha_subj[s] for s in common])
        y = np.array([subj_map[s]   for s in common])

        r, p = stats.pearsonr(x, y)
        r2 = float(r**2)

        # OLS line + 95% CI
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
            name=base_name, x=x, y=y, x_line=x_line, y_line=y_line, y_lo=y_lo, y_hi=y_hi,
            r2=r2, p=p, n=n
        ))

        rows.append({
            "parameter": base_name,
            "n": n,
            "pearson_r": r,
            "r2": r2,
            "p_value": float(p),
        })

    # ---- One-page PDF with a grid of subplots ----
    k = len(panels)
    if k == 0:
        raise ValueError("Nothing to plot (no parameters with >=5 overlapping subjects).")

    ncols = 3 if k <= 9 else 4 if k <= 16 else 5
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()
    
    ACCENT = "darksalmon"  # one place to change the theme color
    for ax, panel in zip(axes, panels):
        x = panel["x"]; y = panel["y"]
        # points
        ax.scatter(x, y, s=30, color=ACCENT, alpha=0.85, edgecolors="none")
        # regression line
        ax.plot(panel["x_line"], panel["y_line"], color=ACCENT, lw=1.8)
        # 95% CI band
        ax.fill_between(panel["x_line"], panel["y_lo"], panel["y_hi"],
                        color=ACCENT, alpha=0.25, linewidth=0)

        ax.set_xlabel("α (learning rate)")
        ax.set_ylabel(panel["name"])
        ax.set_title(panel["name"])

        # Stats box (top-left), NO "N = ..." in the text
        txt = f"R² = {panel['r2']:.3f}\n{_p_text(panel['p'])}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), fontsize=9)

    # Hide any unused axes
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Correlations: α vs subject-level parameters", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # CSV unchanged
    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved (single-page): {out_pdf}")
    print(f"Saved: {out_csv}")


# ---------- run ----------
rl_results_csv = (RL1_DIAG / "results_alpha_transformed.csv").as_posix()
m35_in_csv     = (M35_DIAG / "results.csv").as_posix()
m35_plus_csv   = (M35_DIAG / "results_plus_theta.csv").as_posix()

# 1) augment M35 with θ-parameters
m35_aug = add_theta_params_to_results(m35_in_csv, m35_plus_csv, use_median=False)

# 2) run correlations using augmented results
out_pdf = (OUT_DIR / "alpha_param_correlations_with_theta.pdf").as_posix()
out_csv = (OUT_DIR / "alpha_param_correlations_with_theta_summary.csv").as_posix()

plot_alpha_correlations(
    rl_results_csv=rl_results_csv,
    model35_results_csv=m35_aug,
    out_pdf=out_pdf,
    out_csv=out_csv,
    use_median=False,
    transform_alpha_if_needed=False
)



























# from pathlib import Path
# import os
# import re
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from scipy import stats
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages



# # Base paths
# PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

# M35_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "diagnostics"
# RL1_DIAG = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_LE_RL_1" / "diagnostics"  
# OUT_DIR = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_35" / "correlation"
# OUT_DIR.mkdir(parents=True, exist_ok=True)



# def _read_results(path):
#     """Reads an HDDM/Kabuki results.csv that may have the param in index or an unnamed first col."""
#     df = pd.read_csv(path)
#     # try to find the parameter name column
#     if df.columns[0].lower() in {"", "unnamed: 0"} or not np.issubdtype(df[df.columns[0]].dtype, np.number):
#         df = df.rename(columns={df.columns[0]: "param"})
#     if "param" not in df.columns:
#         # parameter names are likely the index
#         df = df.reset_index().rename(columns={"index": "param"})
#     return df

# def _extract_subj(df, prefix):
#     """Return dict {subj_id: value} for rows like 'prefix_subj.<id>' using the 'mean' column by default."""
#     out = {}
#     pat = re.compile(rf"^{re.escape(prefix)}_subj\.(\d+)$")
#     for _, row in df.iterrows():
#         m = pat.match(str(row["param"]))
#         if m:
#             sid = int(m.group(1))
#             out[sid] = float(row.get("mean", row.get("50q")))
#     return out

# # 1) allow central column selection
# def _extract_all_subject_params(df, central="mean"):
#     by_param = {}
#     pat = re.compile(r"^(?P<base>.+)_subj\.(?P<sid>\d+)$")
#     for _, row in df.iterrows():
#         m = pat.match(str(row["param"]))
#         if not m:
#             continue
#         base = m.group("base")
#         sid  = int(m.group("sid"))
#         val  = float(row.get(central, row.get("mean")))
#         by_param.setdefault(base, {})[sid] = val
#     return by_param


# def inv_logit(x):
#     return np.exp(x) / (1 + np.exp(x))



# def plot_alpha_correlations(
#     rl_results_csv,
#     model35_results_csv,
#     out_pdf="alpha_param_correlations.pdf",
#     out_csv="alpha_param_correlations_summary.csv",
#     alpha_name="alpha",                # row name for group alpha 
#     use_median=False,                  # cn check median later
#     transform_alpha_if_needed=False   
# ):
#     # Load
#     rl = _read_results(rl_results_csv)
#     m35 = _read_results(model35_results_csv)

#     # Choose central tendency column
#     central = "50q" if use_median else "mean"

#     # --- Get subject-level alpha ---
#     alpha_subj = {}
#     pat_alpha = re.compile(r"^alpha_subj\.(\d+)$")
#     for _, row in rl.iterrows():
#         m = pat_alpha.match(str(row["param"]))
#         if m:
#             sid = int(m.group(1))
#             val = float(row.get(central, row.get("mean")))
#             alpha_subj[sid] = val

#     if not alpha_subj:
#         raise ValueError("No alpha_subj.* rows found in the RL results. Check rl_results_csv.")

#     # Optional inverse-logit 
#     if transform_alpha_if_needed:
#         alpha_subj = {sid: inv_logit(val) for sid, val in alpha_subj.items()}

#     # --- Get ALL subject-level params from model 35 ---
#     params_by_name = _extract_all_subject_params(m35)
#     if not params_by_name:
#         raise ValueError("No '*_subj.<id>' parameters found in model35_results_csv.")

#     # Prepare output summary
#     rows = []

#     # PDF with one page per param
#     with PdfPages(out_pdf) as pdf:
#         for base_name, subj_map in sorted(params_by_name.items()):
#             # Match subjects present in both alpha and this param
#             common = sorted(set(alpha_subj).intersection(subj_map))
#             if len(common) < 5:
#                 # Skip tiny Ns to avoid noisy stats
#                 continue

#             x = np.array([alpha_subj[s] for s in common])
#             y = np.array([subj_map[s]   for s in common])

#             # Pearson correlation
#             r, p = stats.pearsonr(x, y)
#             r2 = r**2

#             # Simple OLS line for plotting
#             n = len(common)
#             b1, b0 = np.polyfit(x, y, 1)
#             x_line = np.linspace(x.min(), x.max(), 100)
#             y_line = b1 * x_line + b0
            
#             # Calculate 95% CI for regression line
#             y_pred = b1 * x + b0
#             resid = y - y_pred
#             s_err = np.sqrt(np.sum(resid**2) / (n - 2))
#             t_val = stats.t.ppf(0.975, df=n - 2)
            
#             ci = t_val * s_err * np.sqrt(
#                 1/n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2)
#             )
#             y_line_lower = y_line - ci
#             y_line_upper = y_line + ci
            
#             # --- Plot ---
#             fig, ax = plt.subplots(figsize=(5, 4))
#             ax.scatter(x, y)
#             ax.plot(x_line, y_line, color="blue")
#             ax.fill_between(x_line, y_line_lower, y_line_upper, color="blue", alpha=0.2)  # CI shading
#             ax.set_xlabel("α (learning rate)")
#             ax.set_ylabel(base_name)
#             ax.set_title(base_name)
#             ax.text(0.02, 0.98, f"R² = {r2:.3f}\np = {p:.3g}\nN = {n}",
#                     transform=ax.transAxes, va="top", ha="left")
#             pdf.savefig(fig, bbox_inches="tight")
#             plt.close(fig)
            
#             rows.append({
#                 "parameter": base_name,
#                 "n": len(common),
#                 "pearson_r": r,
#                 "r2": r2,
#                 "p_value": p
#             })

#     summ = pd.DataFrame(rows).sort_values("r2", ascending=False)
#     summ.to_csv(out_csv, index=False)

#     print(f"Saved: {out_pdf}")
#     print(f"Saved: {out_csv}")

# # Files
# rl_results_csv = (RL1_DIAG / "results_alpha_transformed.csv").as_posix()
# model35_results_csv = (M35_DIAG / "results.csv").as_posix()
# out_pdf = (OUT_DIR / "alpha_param_correlations_trans.pdf").as_posix()
# out_csv = (OUT_DIR / "alpha_param_correlations_summary_trans.csv").as_posix()

# # Run the correlation plotting
# plot_alpha_correlations(
#     rl_results_csv=rl_results_csv,
#     model35_results_csv=model35_results_csv,
#     out_pdf=out_pdf,
#     out_csv=out_csv,
#     use_median=False,
#     transform_alpha_if_needed=False  
# )

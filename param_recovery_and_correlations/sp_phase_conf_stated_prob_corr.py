#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlate subject-level aDDM parameters with SP-phase ratings.

This script builds SIX correlation panels (each with a CSV summary):
  1) Stated probability ratings (SP) for E options vs aDDM params
  2) Stated probability ratings (SP) for S options vs aDDM params
  3) Stated probability ratings (SP) for ALL options vs aDDM params
  4) Confidence ratings (SP) for E options vs aDDM params
  5) Confidence ratings (SP) for S options vs aDDM params
  6) Confidence ratings (SP) for ALL options vs aDDM params

Output: one PDF grid per target and one CSV summary per target in OUT_DIR.

Paths are controlled via environment variables when available.
- PROJECT_DIR: base project directory (default: /workspace)
- SP_DATA_CSV: path to combined CSV with SP trials (should include: phase, sub_id, op1, cho, and ideally confidence)
- CONF_GLOB  : optional glob for per-subject files with confidence (if not present in SP_DATA_CSV),
               default: {PROJECT_DIR}/data/sub-*/beh/EXP4_Garcia_participant_*.csv
- EXCLUDE_SUBJECTS: comma-separated subject IDs to exclude (default: 6,99)

Assumptions about columns:
- aDDM results at: {PROJECT_DIR}/figures_dir_garcia/garcia_replication_ES_VAL_36/diagnostics/results.csv
- SP combined CSV has: 'phase' (expects 'SP'), 'sub_id', 'op1' (E/S), 'cho' (stated probability rating).
  If confidence is present, it may be named one of: ['confidence','conf','confidenceLevelsArrayEXP','confidence_level','confidenceLevels']
- If confidence isn't in SP combined CSV, we fall back to CONF_GLOB files, where we expect columns:
  'SubID', 'confidenceLevelsArrayEXP', and 'selectedImageNamesArrayEXP' used to infer option type:
  contains 'Pie' => S, otherwise E.

The script also augments aDDM results with theta parameters:
  theta_InatWS = v_ES_InattentionW_S / v_ES_AttentionW
  theta_InatWE = v_ES_InattentionW_E / v_ES_AttentionW

Author: ChatGPT (for Veronika)
Date: 2025-08-31
"""

from __future__ import annotations
from pathlib import Path
import os
import re
import math
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ------------------------- config & paths -------------------------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", r"C:/Cluster_Github/HDDM_Vero")).resolve()
M35_DIAG    = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_VAL_36" / "diagnostics"
OUT_DIR     = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_ES_VAL_36" / "correlation" / "sp_phase"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input data for SP trials
SP_DATA_CSV = os.getenv("SP_DATA_CSV")  # recommended to set
if SP_DATA_CSV is None:
    # try a few common locations
    CANDIDATES = [
        PROJECT_DIR / "data" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv",
        PROJECT_DIR / "data_sets" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv",
        Path(r"D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/data_sets/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv"),
        Path(r"C:/Cluster_Github/HDDM_Vero/data_sets/data_sets_Garcia/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv"),
    ]
    SP_PATH = next((str(p) for p in CANDIDATES if Path(p).exists()), None)
else:
    SP_PATH = SP_DATA_CSV

# Optional glob for per-subject confidence files (fallback)
CONF_GLOB = os.getenv("CONF_GLOB") or r"D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/sub-*/beh/EXP4_Garcia_participant_*.csv"

# Exclusion list
_excl = os.getenv("EXCLUDE_SUBJECTS", "1,4,5,6,14,99").strip()
EXCLUDE_SUBJECTS = [] if _excl == "" else [int(x) for x in _excl.split(",") if x.strip().isdigit()]

# ------------------------- helpers -------------------------
def _read_results(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    first = df.columns[0]
    if isinstance(first, str) and (first.lower() in {"", "unnamed: 0"} or not np.issubdtype(df[first].dtype, np.number)):
        df = df.rename(columns={first: "param"})
    if "param" not in df.columns:
        df = df.reset_index().rename(columns={"index": "param"})
    return df


def _extract_all_subject_params(df: pd.DataFrame, central: str = "mean") -> dict[str, dict[int, float]]:
    """Return mapping {param_name: {sid: value}}.
    Supports names like:
      a_subj.12 -> 'a'
      a_subj(high).12 -> 'a(high)'
      v_ES_AttentionW_subj.12 -> 'v_ES_AttentionW'
    """
    by_param: dict[str, dict[int, float]] = {}
    pat = re.compile(r"^(?P<base>.+)_subj(?:\((?P<mod>.+?)\))?\.(?P<sid>\d+)$")
    for _, row in df.iterrows():
        m = pat.match(str(row["param"]))
        if not m:
            continue
        base = m.group("base")
        mod = m.group("mod")
        sid = int(m.group("sid"))
        val = float(row.get(central, row.get("mean")))
        full_name = f"{base}({mod})" if mod else base
        by_param.setdefault(full_name, {})[sid] = val
    return by_param


def add_theta_params_to_results(m35_in_csv: Path | str, m35_out_csv: Path | str, use_median: bool = False) -> str:
    df = _read_results(m35_in_csv)
    central = "50q" if use_median else "mean"

    subj_maps = _extract_all_subject_params(df, central=central)
    need = {
        "num_v_ES_InattentionW_S": "v_ES_InattentionW_S",
        "num_v_ES_InattentionW_E": "v_ES_InattentionW_E",
        "num_v_ES_AttentionW": "v_ES_AttentionW",
    }
    for k, p in need.items():
        if p not in subj_maps:
            raise ValueError(f"Missing subject-level parameter in results: '{p}_subj.<id>'")

    num_inat_s = subj_maps[need["num_v_ES_InattentionW_S"]]
    num_inat_e = subj_maps[need["num_v_ES_InattentionW_E"]]
    att        = subj_maps[need["num_v_ES_AttentionW"]]

    combos = [
        ("theta_InatWS", num_inat_s, att),
        ("theta_InatWE", num_inat_e, att),
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

    if new_rows:
        df_theta = pd.DataFrame(new_rows, columns=cols)
        df_out = pd.concat([df, df_theta], ignore_index=True)
    else:
        df_out = df

    Path(m35_out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(m35_out_csv, index=False)
    print(f"Saved augmented results with theta params in {m35_out_csv}")
    return str(m35_out_csv)


def _p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    return "p<.001" if p < 1e-3 else f"p={p:.3f}"


# ------------------------- SP data loaders -------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Correlate aDDM params with SP-phase ratings & confidence.")
    p.add_argument("--results", type=str, default=os.getenv("M35_RESULTS_CSV"),
                   help="Path to aDDM results.csv (overrides auto-detection).")
    p.add_argument("--sp-csv", type=str, default=os.getenv("SP_DATA_CSV"),
                   help="Path to combined SP CSV (phase=subset 'SP', with sub_id, op1, cho, and optionally confidence).")
    p.add_argument("--conf-glob", type=str, default=os.getenv("CONF_GLOB"),
                   help="Glob for per-subject SP files if confidence isn't in the combined CSV.")
    p.add_argument("--exclude", type=str, default=os.getenv("EXCLUDE_SUBJECTS", "1,4,5,6,14,99"),
                   help="Comma-separated subject IDs to exclude.")
    args, _ = p.parse_known_args(argv)
    return args

# ------------------------- SP data loaders -------------------------
SP_CONF_CANDIDATE_COLS = [
    "confidence", "conf", "confidenceLevelsArrayEXP", "confidence_level", "confidenceLevels"
]


def _standardize_op1(op):
    if pd.isna(op):
        return np.nan
    s = str(op).strip().upper()
    if s in {"E", "S"}:
        return s
    # try to infer from strings like 'Pie...' present in some files
    return "S" if "PIE" in s else "E"


def read_sp_from_combined(csv_path: str | Path) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float] | None, dict[int, float] | None, dict[int, float] | None]:
    """Return six maps: (prob_E, prob_S, prob_all, conf_E, conf_S, conf_all).
    conf_* may be None if confidence not available in this file.
    """
    df = pd.read_csv(csv_path)
    if "phase" not in df.columns:
        raise ValueError("Combined SP CSV must have a 'phase' column.")

    # filter SP
    df = df[df["phase"].astype(str).str.upper() == "SP"].copy()
    if df.empty:
        raise ValueError("No SP-phase rows found in the combined CSV.")

    # exclusions
    if "sub_id" not in df.columns:
        raise ValueError("Combined SP CSV must have 'sub_id'.")
    df = df[~df["sub_id"].isin(EXCLUDE_SUBJECTS)].copy()
    df["sub_id"] = pd.to_numeric(df["sub_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["sub_id"])  # keep valid SIDs
    df["sub_id"] = df["sub_id"].astype(int)

    # op1 and stated probability
    if "op1" not in df.columns:
        raise ValueError("Combined SP CSV must have 'op1' column for option type (E/S)")
    df["op1_std"] = df["op1"].map(_standardize_op1)
    df = df.dropna(subset=["op1_std"]).copy()

    if "cho" not in df.columns:
        raise ValueError("Combined SP CSV must include 'cho' as stated probability rating.")
    df["cho"] = pd.to_numeric(df["cho"], errors="coerce")
    df = df.dropna(subset=["cho"])  # ratings present

    # mean stated probability per subject by option
    prob_E   = df[df["op1_std"] == "E"].groupby("sub_id")["cho"].mean().to_dict()
    prob_S   = df[df["op1_std"] == "S"].groupby("sub_id")["cho"].mean().to_dict()
    prob_all = df.groupby("sub_id")["cho"].mean().to_dict()

    # confidence if present
    conf_col = next((c for c in SP_CONF_CANDIDATE_COLS if c in df.columns), None)
    if conf_col is None:
        return prob_E, prob_S, prob_all, None, None, None

    df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce")
    df_conf = df.dropna(subset=[conf_col]).copy()
    if df_conf.empty:
        return prob_E, prob_S, prob_all, None, None, None

    conf_E   = df_conf[df_conf["op1_std"] == "E"].groupby("sub_id")[conf_col].mean().to_dict()
    conf_S   = df_conf[df_conf["op1_std"] == "S"].groupby("sub_id")[conf_col].mean().to_dict()
    conf_all = df_conf.groupby("sub_id")[conf_col].mean().to_dict()
    return prob_E, prob_S, prob_all, conf_E, conf_S, conf_all


def read_confidence_from_glob(glob_pattern: str | Path) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Fallback loader for confidence from per-subject files.
    Requires columns: SubID, confidenceLevelsArrayEXP, selectedImageNamesArrayEXP.
    Infers op1: contains 'Pie' -> 'S', else 'E'. Returns (conf_E, conf_S, conf_all).
    """
    files = sorted(glob.glob(str(glob_pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched CONF_GLOB pattern: {glob_pattern}")

    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            dfs.append(tmp)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not dfs:
        raise ValueError("No readable CSVs for confidence fallback.")

    df = pd.concat(dfs, ignore_index=True)
    needed = {"SubID", "confidenceLevelsArrayEXP", "selectedImageNamesArrayEXP"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Confidence fallback files missing columns: {missing}")

    df = df.dropna(subset=["SubID", "confidenceLevelsArrayEXP"]).copy()
    df["SubID"] = pd.to_numeric(df["SubID"], errors="coerce")
    df = df.dropna(subset=["SubID"]).copy()
    df["SubID"] = df["SubID"].astype(int)

    # infer op1
    img = df["selectedImageNamesArrayEXP"].astype(str)
    op1 = np.where(img.str.contains("Pie", case=False, na=False), "S", "E")
    df = df.assign(op1_std=op1)

    conf = pd.to_numeric(df["confidenceLevelsArrayEXP"], errors="coerce")
    df = df.assign(conf=conf).dropna(subset=["conf"]).copy()

    # exclusions
    df = df[~df["SubID"].isin(EXCLUDE_SUBJECTS)].copy()

    conf_E   = df[df["op1_std"] == "E"].groupby("SubID")["conf"].mean().to_dict()
    conf_S   = df[df["op1_std"] == "S"].groupby("SubID")["conf"].mean().to_dict()
    conf_all = df.groupby("SubID")["conf"].mean().to_dict()
    return conf_E, conf_S, conf_all


# ------------------------- correlation plotting -------------------------
def correlate_and_plot(target_map: dict[int, float],
                       params_by_name: dict[str, dict[int, float]],
                       x_label: str,
                       title_prefix: str,
                       accent: str,
                       out_pdf: Path | str,
                       out_csv: Path | str,
                       min_overlap: int = 5) -> None:
    """Make grid of scatter+fit+CI and a CSV summary."""
    panels = []
    rows = []

    for base_name, subj_map in sorted(params_by_name.items()):
        common = sorted(set(target_map).intersection(subj_map))
        if len(common) < min_overlap:
            continue

        x = np.array([target_map[s] for s in common], dtype=float)
        y = np.array([subj_map[s]   for s in common], dtype=float)

        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            # no variability -> skip correlation
            continue

        r, p = stats.pearsonr(x, y)
        r2 = float(r**2)
        n = len(common)

        # regression & 95% CI
        b1, b0 = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b1 * x_line + b0
        y_pred = b1 * x + b0
        resid  = y - y_pred
        s_err  = np.sqrt(np.sum(resid**2) / max(n - 2, 1))
        t_val  = stats.t.ppf(0.975, df=max(n - 2, 1))
        denom  = np.sum((x - x.mean())**2)
        ci     = t_val * s_err * np.sqrt(1/n + (x_line - x.mean())**2 / max(denom, 1e-12))
        y_lo   = y_line - ci
        y_hi   = y_line + ci

        panels.append(dict(
            name=base_name, x=x, y=y,
            x_line=x_line, y_line=y_line, y_lo=y_lo, y_hi=y_hi,
            r2=r2, p=p, n=n
        ))

        rows.append({
            "parameter": base_name,
            "n": n,
            "pearson_r": r,
            "r2": r2,
            "p_value": float(p),
        })

    if not panels:
        print(f"[WARN] Nothing to plot for {title_prefix} (no parameters with >= {min_overlap} overlapping subjects).")
        return

    k = len(panels)
    ncols = 3 if k <= 9 else 4 if k <= 16 else 5
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, panel in zip(axes, panels):
        ax.scatter(panel["x"], panel["y"], s=30, color=accent, alpha=0.85, edgecolors="none")
        ax.plot(panel["x_line"], panel["y_line"], color=accent, lw=1.8)
        ax.fill_between(panel["x_line"], panel["y_lo"], panel["y_hi"], color=accent, alpha=0.25, linewidth=0)

        ax.set_xlabel(x_label)
        ax.set_ylabel(panel["name"])
        ax.set_title(panel["name"])

        txt = f"R² = {panel['r2']:.3f}\n{_p_text(panel['p'])}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), fontsize=9)
        ax.margins(0.05)

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title_prefix, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved CSV: {out_csv}")


# ------------------------- run pipeline -------------------------
if __name__ == "__main__":
    args = _parse_args()

    # ---- exclusions & cross-folder data locations ----
    global EXCLUDE_SUBJECTS, CONF_GLOB, SP_PATH

    # exclusions
    _excl = (args.exclude or "").strip()
    EXCLUDE_SUBJECTS = [] if _excl == "" else [int(x) for x in _excl.split(",") if x.strip().isdigit()]

    # confidence files glob
    CONF_GLOB = args.conf_glob or os.getenv("CONF_GLOB") or \
        r"D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/sub-*/beh/EXP4_Garcia_participant_*.csv"

    # SP combined CSV (stated probabilities & maybe confidence)
    if args.sp_csv:
        SP_PATH = args.sp_csv
    elif os.getenv("SP_DATA_CSV"):
        SP_PATH = os.getenv("SP_DATA_CSV")
    else:
        sp_candidates = [
            Path(r"D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/data_sets/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv"),
            PROJECT_DIR / "data" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv",
            PROJECT_DIR / "data_sets" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv",
        ]
        SP_PATH = next((str(p) for p in sp_candidates if Path(p).exists()), None)

    # ---- aDDM parameters (with theta) ----
    # locate aDDM results (results.csv) robustly
    if args.results:
        candidates = [Path(args.results)]
        m35_in = candidates[0]
    else:
        candidates = [
            M35_DIAG / "results.csv",
            PROJECT_DIR / "figures_dir_garcia" / "macleod_cluster_out" / "garcia_replication_ES_VAL_36" / "diagnostics" / "results.csv",
            Path(r"C:/Cluster_Github/HDDM_Vero/figures_dir_garcia/garcia_replication_ES_VAL_36/diagnostics/results.csv"),
            Path(r"C:/Cluster_Github/HDDM_Vero/figures_dir_garcia/macleod_cluster_out/garcia_replication_ES_VAL_36/diagnostics/results.csv"),
            Path(r"D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/figures_dir_garcia/garcia_replication_ES_VAL_36/diagnostics/results.csv"),
            Path(r"D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/figures_dir_garcia/macleod_cluster_out/garcia_replication_ES_VAL_36/diagnostics/results.csv"),
        ]
        m35_in = next((p for p in candidates if p.exists()), None)

    if m35_in is None or not Path(m35_in).exists():
        tried = "".join(map(str, candidates))
        raise FileNotFoundError("Couldn't find aDDM results.csv. Use --results or set PROJECT_DIR/M35_RESULTS_CSV. Tried:" + tried
        )

    # where to save the theta-augmented CSV
    m35_plus = Path(m35_in).with_name("results_plus_theta.csv")

    print(f"[INFO] Using aDDM results: {m35_in}")
    print(f"[INFO] Using SP combined CSV: {SP_PATH}")
    print(f"[INFO] Using CONF_GLOB: {CONF_GLOB}")
    print(f"[INFO] Excluding subjects: {EXCLUDE_SUBJECTS}")

    m35_aug_path = add_theta_params_to_results(m35_in, m35_plus, use_median=False)
    m35_df = _read_results(m35_aug_path)
    params_by_name = _extract_all_subject_params(m35_df, central="mean")
    if not params_by_name:
        raise ValueError("No '*_subj.<id>' parameters found in aDDM results.")

    # ---- stated probabilities & confidence from combined SP CSV ----
    if SP_PATH is None:
        raise FileNotFoundError(
            "SP_DATA_CSV not set and no default combined CSV found. Provide --sp-csv or set SP_DATA_CSV."
        )

    (prob_E, prob_S, prob_all,
     conf_E_from_comb, conf_S_from_comb, conf_all_from_comb) = read_sp_from_combined(SP_PATH)

    # ---- confidence fallback (if needed) ----
    conf_E = conf_E_from_comb
    conf_S = conf_S_from_comb
    conf_all = conf_all_from_comb

    if conf_E is None or conf_S is None or conf_all is None:
        print("[INFO] Confidence ratings not found in combined SP CSV. Trying CONF_GLOB fallback...")
        try:
            conf_E, conf_S, conf_all = read_confidence_from_glob(CONF_GLOB)
            print(f"[INFO] Loaded confidence from per-subject files: pattern {CONF_GLOB}")
        except Exception as e:
            print(f"[WARN] Could not load confidence ratings from fallback: {e}")
            conf_E = conf_E or {}
            conf_S = conf_S or {}
            conf_all = conf_all or {}

    # ---------------- make 6 outputs ----------------
    # Colors per target
    COL_E   = "#5B95B5"  # blue-ish
    COL_S   = "#AA4E73"  # magenta-ish
    COL_ALL = "teal"

    # 1–3: stated probability ratings
    correlate_and_plot(
        prob_E, params_by_name,
        x_label="Mean stated probability (SP, E)",
        title_prefix="Correlations: SP stated probability (E) vs aDDM parameters",
        accent=COL_E,
        out_pdf=OUT_DIR / "addm_vs_sp_ratings_E.pdf",
        out_csv=OUT_DIR / "addm_vs_sp_ratings_E_summary.csv",
    )

    correlate_and_plot(
        prob_S, params_by_name,
        x_label="Mean stated probability (SP, S)",
        title_prefix="Correlations: SP stated probability (S) vs aDDM parameters",
        accent=COL_S,
        out_pdf=OUT_DIR / "addm_vs_sp_ratings_S.pdf",
        out_csv=OUT_DIR / "addm_vs_sp_ratings_S_summary.csv",
    )

    correlate_and_plot(
        prob_all, params_by_name,
        x_label="Mean stated probability (SP, all)",
        title_prefix="Correlations: SP stated probability (ALL) vs aDDM parameters",
        accent=COL_ALL,
        out_pdf=OUT_DIR / "addm_vs_sp_ratings_ALL.pdf",
        out_csv=OUT_DIR / "addm_vs_sp_ratings_ALL_summary.csv",
    )

    # 4–6: confidence ratings (only if we have any)
    if len(conf_E) + len(conf_S) + len(conf_all) == 0:
        print("[WARN] No confidence data found. Skipping confidence correlations (4–6).")
    else:
        if conf_E:
            correlate_and_plot(
                conf_E, params_by_name,
                x_label="Mean confidence (SP, E)",
                title_prefix="Correlations: SP confidence (E) vs aDDM parameters",
                accent=COL_E,
                out_pdf=OUT_DIR / "addm_vs_sp_confidence_E.pdf",
                out_csv=OUT_DIR / "addm_vs_sp_confidence_E_summary.csv",
            )
        else:
            print("[WARN] No E-option confidence data; skipping (4).")

        if conf_S:
            correlate_and_plot(
                conf_S, params_by_name,
                x_label="Mean confidence (SP, S)",
                title_prefix="Correlations: SP confidence (S) vs aDDM parameters",
                accent=COL_S,
                out_pdf=OUT_DIR / "addm_vs_sp_confidence_S.pdf",
                out_csv=OUT_DIR / "addm_vs_sp_confidence_S_summary.csv",
            )
        else:
            print("[WARN] No S-option confidence data; skipping (5).")

        if conf_all:
            correlate_and_plot(
                conf_all, params_by_name,
                x_label="Mean confidence (SP, all)",
                title_prefix="Correlations: SP confidence (ALL) vs aDDM parameters",
                accent=COL_ALL,
                out_pdf=OUT_DIR / "addm_vs_sp_confidence_ALL.pdf",
                out_csv=OUT_DIR / "addm_vs_sp_confidence_ALL_summary.csv",
            )
        else:
            print("[WARN] No ALL confidence data; skipping (6).")

    print("Done. Outputs are in:", OUT_DIR)

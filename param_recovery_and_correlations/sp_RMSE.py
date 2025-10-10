
from __future__ import annotations
import argparse
import os
import re
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


PROJECT_DIR = Path(os.getenv("PROJECT_DIR", r"C:/Cluster_Github/HDDM_Vero")).resolve()
OUT_DIR     = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_For_paper_7" / "correlation" / "sp_phase_rmse"
DEFAULT_EXCLUDE = [1,4,5,6,14,99]   # subs for EXP2:6, 14, 20, 26, 2, 9, 18        For EXP1:  1,4,5,6,14,99

M35_DEFAULT = PROJECT_DIR / "figures_dir_garcia" / "macleod_cluster_out" / "garcia_replication_For_paper_7" / "diagnostics" / "results.csv"

SP_CSV_FALLBACKS = [
    Path(r"D:/Aberdeen_Uni_June24/cap/THESIS/Garcia_Analysis/data/data_sets/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv"),
    PROJECT_DIR / "data" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv",
    PROJECT_DIR / "data_sets" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv",
    Path(r"C:/Cluster_Github/HDDM_Vero/data_sets/data_sets_garcia/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv"),
]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="aDDM vs SP RMSE (strict same-subjects)")
    p.add_argument("--results",    type=str, default=os.getenv("M35_RESULTS_CSV"),
                   help="Path to aDDM results.csv")
    p.add_argument("--sp-csv",     type=str, default=os.getenv("SP_DATA_CSV"),
                   help="Combined SP CSV with: phase, sub_id, op1, p1, cho")
    p.add_argument("--exclude",    type=str, default=os.getenv("EXCLUDE_SUBJECTS", ",".join(map(str, DEFAULT_EXCLUDE))),
                   help="Comma-separated subject IDs to exclude")
    p.add_argument("--project-dir", type=str, default=os.getenv("PROJECT_DIR"),
                   help="Override PROJECT_DIR for outputs")
    args, _ = p.parse_known_args(argv)
    return args


# ------------------- HELPERS -------------------
def _read_results(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    first = df.columns[0]
    if isinstance(first, str) and (first.lower() in {"", "unnamed: 0"} or not np.issubdtype(df[first].dtype, np.number)):
        df = df.rename(columns={first: "param"})
    if "param" not in df.columns:
        df = df.reset_index().rename(columns={"index": "param"})
    return df


def _extract_all_subject_params(df: pd.DataFrame, central: str = "mean") -> dict[str, dict[int, float]]:
    """Return mapping {param_name: {sid: value}}. Supports a_subj.12 and a_subj(high).12"""
    by_param: dict[str, dict[int, float]] = {}
    pat = re.compile(r"^(?P<base>.+)_subj(?:\((?P<mod>.+?)\))?\.(?P<sid>\d+)$")
    for _, row in df.iterrows():
        m = pat.match(str(row["param"]))
        if not m:
            continue
        base = m.group("base"); mod = m.group("mod"); sid = int(m.group("sid"))
        val  = float(row.get(central, row.get("mean")))
        name = f"{base}({mod})" if mod else base
        by_param.setdefault(name, {})[sid] = val
    return by_param


def _maybe_add_theta(results_df: pd.DataFrame) -> pd.DataFrame:
    """Append theta rows if Attention/ Inattention weights are present."""
    by = _extract_all_subject_params(results_df)
    need = ["v_ES_InattentionW_S", "v_ES_InattentionW_E", "v_ES_AttentionW"]
    if not all(k in by for k in need):
        return results_df
    att = by["v_ES_AttentionW"]
    rows, cols = [], list(results_df.columns)
    if "param" not in cols:
        cols = ["param"] + [c for c in cols if c != "param"]
    for nm, num in [("theta_InatWS", by["v_ES_InattentionW_S"]),
                    ("theta_InatWE", by["v_ES_InattentionW_E"])]:
        for sid in sorted(set(num).intersection(att)):
            den = att[sid]
            if den is None or np.isclose(den, 0.0):
                continue
            v = num[sid] / den
            row = {c: np.nan for c in cols}
            row["param"], row["mean"] = f"{nm}_subj.{sid}", v
            rows.append(row)
    if rows:
        results_df = pd.concat([results_df, pd.DataFrame(rows, columns=cols)], ignore_index=True)
    return results_df


def _normalize_prob_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return x/100.0 if x.max(skipna=True) > 1.5 else x


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((true - pred)**2)))


def rmse_by_subject(sp_df: pd.DataFrame, which: str | None, subjects: list[int]) -> pd.Series:
    d = sp_df if which is None else sp_df[sp_df["op1_std"] == which]
    out = {}
    for sid in subjects:
        g = d.loc[d["sub_id"] == sid, ["p1", "cho"]].dropna()
        out[sid] = np.nan if g.empty else rmse(g["p1"].to_numpy(), g["cho"].to_numpy())
    return pd.Series(out).sort_index()


def correlate_grid(target: pd.Series,
                   params_by_name: dict[str, dict[int, float]],
                   subjects: list[int],
                   out_pdf: Path, out_csv: Path,
                   title: str, accent: str) -> None:
    panels, rows = [], []
    x_all = target.reindex(subjects)

    for pname, pmap in sorted(params_by_name.items()):
        y_all = pd.Series(pmap).reindex(subjects)
        mask = ~(x_all.isna() | y_all.isna())
        x = x_all[mask].to_numpy(); y = y_all[mask].to_numpy()
        n = len(x)
        if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            continue
        r, p = stats.pearsonr(x, y); r2 = float(r**2)
        b1, b0 = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = b1 * x_line + b0
        y_pred = b1 * x + b0
        resid  = y - y_pred
        s_err  = np.sqrt(np.sum(resid**2) / max(n - 2, 1))
        t_val  = stats.t.ppf(0.975, df=max(n - 2, 1))
        denom  = np.sum((x - x.mean())**2)
        ci     = t_val * s_err * np.sqrt(1/n + (x_line - x.mean())**2 / max(denom, 1e-12))
        panels.append(dict(name=pname, x=x, y=y, x_line=x_line, y_line=y_line,
                           y_lo=y_line-ci, y_hi=y_line+ci, r2=r2, p=p))
        rows.append({"parameter": pname, "n": n, "pearson_r": r, "r2": r2, "p_value": float(p)})

    if not panels:
        print(f"[WARN] No panels for: {title}")
        return

    k = len(panels)
    ncols = 3 if k <= 9 else 4 if k <= 16 else 5
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, panel in zip(axes, panels):
        # scatter + regression
        ax.scatter(panel["x"], panel["y"], s=60, color=accent, alpha=0.85, edgecolors="none")
        ax.plot(panel["x_line"], panel["y_line"], color=accent, lw=2.5)
        ax.fill_between(panel["x_line"], panel["y_lo"], panel["y_hi"],
                        color=accent, alpha=0.25, linewidth=0)

        ax.set_xlabel("RMSE", fontsize=16, labelpad=8)
        ax.set_ylabel(panel["name"], fontsize=16, labelpad=8)
        ax.set_title(panel["name"], fontsize=18, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=14, width=1.5)

        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["bottom", "left"]:
            ax.spines[side].set_linewidth(1.5)

        txt = f"R² = {panel['r2']:.3f}\n" + ("p<.001" if panel['p'] < 1e-3 else f"p={panel['p']:.3f}")
        ax.text(
            0.03, 0.98, txt, transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.9),
            fontsize=14,
        )

        ax.margins(0.05)


    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=0.995, fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).sort_values("r2", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved CSV: {out_csv}")


# ------------------- MAIN -------------------
if __name__ == "__main__":
    args = _parse_args()

    # Project dir override (optional)
    if args.project_dir:
        PROJECT_DIR = Path(args.project_dir).resolve()
        OUT_DIR     = PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_For_paper_7" / "correlation" / "sp_phase_rmse"

    # Exclusions
    exclude_ids = [int(s) for s in (args.exclude or "").split(",") if s.strip().isdigit()] or DEFAULT_EXCLUDE

    # ----- aDDM results (auto) -----
    if args.results:
        results_csv = Path(args.results)
    else:
        candidates = [
            M35_DEFAULT,
            PROJECT_DIR / "figures_dir_garcia" / "garcia_replication_For_paper_7" / "diagnostics" / "results.csv",
            PROJECT_DIR / "figures_dir_garcia" / "macleod_cluster_out" / "garcia_replication_For_paper_7" / "diagnostics" / "results.csv",
            Path(r"C:/Cluster_Github/HDDM_Vero/figures_dir_garcia/macleod_cluster_out/garcia_replication_For_paper_7/diagnostics/results.csv"),
        ]
        results_csv = next((p for p in candidates if p.exists()), M35_DEFAULT)
    print(f"[INFO] Using aDDM results: {results_csv}")
    if not results_csv.exists():
        raise FileNotFoundError(f"aDDM results not found: {results_csv}")

    results_df  = _read_results(results_csv)
    results_df  = _maybe_add_theta(results_df)
    params_by   = _extract_all_subject_params(results_df, central="mean")
    if not params_by:
        raise ValueError("No '*_subj.<id>' parameters found in results.csv.")

    # ----- SP CSV (auto) -----
    if args.sp_csv:
        sp_csv = Path(args.sp_csv)
    else:
        sp_csv = next((p for p in SP_CSV_FALLBACKS if p.exists()), None)
    if sp_csv is None or not sp_csv.exists():
        tried = "\n  - " + "\n  - ".join(str(p) for p in SP_CSV_FALLBACKS)
        raise FileNotFoundError("SP CSV not found. Tried:" + tried + "\n"
                                "Fix: put the file in one of those places or pass --sp-csv <path>.")
    print(f"[INFO] Using SP CSV: {sp_csv}")

    sp = pd.read_csv(sp_csv)
    need = {"phase", "sub_id", "op1", "p1", "cho"}
    miss = need - set(sp.columns)
    if miss:
        raise ValueError(f"SP CSV missing columns: {miss}")

    # filter & clean (strict same-subjects after exclusions)
    sp = sp[sp["phase"].astype(str).str.upper() == "SP"].copy()
    sp["sub_id"] = pd.to_numeric(sp["sub_id"], errors="coerce").astype("Int64")
    sp = sp.dropna(subset=["sub_id"]).copy()
    sp["sub_id"] = sp["sub_id"].astype(int)
    sp = sp[~sp["sub_id"].isin(exclude_ids)].copy()

    # exact E/S (you said op1 is already clean E/S)
    sp["op1_std"] = sp["op1"].astype(str).str.strip().str.upper()

    # normalize scales (e.g., 90 -> 0.90)
    sp["p1"]  = _normalize_prob_series(sp["p1"])
    sp["cho"] = _normalize_prob_series(sp["cho"])

    # subject list used everywhere
    subjects = sorted(sp["sub_id"].unique().tolist())
    print(f"[INFO] N subjects (after exclusion): {len(subjects)}")

    # RMSE targets
    rmse_E   = rmse_by_subject(sp, "E", subjects)
    rmse_S   = rmse_by_subject(sp, "S", subjects)
    rmse_ALL = rmse_by_subject(sp, None, subjects)

    # correlate & plot (three grid PDFs + CSVs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    correlate_grid(rmse_E,   params_by, subjects,
                   out_pdf=OUT_DIR / "addm_vs_sp_rmse_E.pdf",
                   out_csv=OUT_DIR / "addm_vs_sp_rmse_E_summary.csv",
                   title="Correlations: SP RMSE (E) vs aDDM parameters",
                   accent="#5B95B5")

    correlate_grid(rmse_S,   params_by, subjects,
                   out_pdf=OUT_DIR / "addm_vs_sp_rmse_S.pdf",
                   out_csv=OUT_DIR / "addm_vs_sp_rmse_S_summary.csv",
                   title="Correlations: SP RMSE (S) vs aDDM parameters",
                   accent="#AA4E73")

    correlate_grid(rmse_ALL, params_by, subjects,
                   out_pdf=OUT_DIR / "addm_vs_sp_rmse_ALL.pdf",
                   out_csv=OUT_DIR / "addm_vs_sp_rmse_ALL_summary.csv",
                   title="Correlations: SP RMSE (ALL) vs aDDM parameters",
                   accent="teal")

    print(f"Done. Outputs are in: {OUT_DIR}")

import numpy as np
import pandas as pd
from pathlib import Path
import os

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/workspace")).resolve()

# -----------------------------
# helpers
# -----------------------------
def normalize_prob_series(x):
    x = pd.to_numeric(x, errors="coerce")
    return x / 100.0 if x.max(skipna=True) > 1.5 else x

def rmse(true, pred):
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((true - pred) ** 2)))

# -----------------------------
# file path
# -----------------------------
main_path = Path(os.getenv(
    "DATA_FILE",
    (PROJECT_DIR / "data_sets" / "data_sets_Garcia" / "GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv").as_posix()
)).resolve()

main_df = pd.read_csv(main_path, sep=",")

# -----------------------------
# optional exclusions
# -----------------------------
exclude_ids = [1, 4, 5, 6, 14, 99]

# -----------------------------
# prepare SP data
# -----------------------------
sp = main_df.copy()
sp = sp[sp["phase"].astype(str).str.upper() == "SP"].copy()

sp["sub_id"] = pd.to_numeric(sp["sub_id"], errors="coerce")
sp = sp.dropna(subset=["sub_id"]).copy()
sp["sub_id"] = sp["sub_id"].astype(int)

sp = sp[~sp["sub_id"].isin(exclude_ids)].copy()

sp["p1"] = normalize_prob_series(sp["p1"])
sp["cho"] = normalize_prob_series(sp["cho"])
sp = sp.dropna(subset=["p1", "cho"]).copy()

# -----------------------------
# RMSE per participant
# -----------------------------
rmse_df = (
    sp.groupby("sub_id")
      .apply(lambda g: rmse(g["p1"], g["cho"]))
      .reset_index(name="rmse_sp")
)

rmse_df["memory_precision"] = -rmse_df["rmse_sp"]

rmse_df["memory_precision_z"] = (
    rmse_df["memory_precision"] - rmse_df["memory_precision"].mean()
) / rmse_df["memory_precision"].std(ddof=1)

rmse_df["rmse_sp_z"] = (
    rmse_df["rmse_sp"] - rmse_df["rmse_sp"].mean()
) / rmse_df["rmse_sp"].std(ddof=1)

print(rmse_df.head())

# -----------------------------
# merge into main dataframe
# -----------------------------
main_df["subj_idx"] = pd.to_numeric(main_df["subj_idx"], errors="coerce")

# remove old versions if script is run again
cols_to_add = ["rmse_sp", "rmse_sp_z", "memory_precision", "memory_precision_z"]
existing = [c for c in cols_to_add if c in main_df.columns]
if existing:
    main_df = main_df.drop(columns=existing)

main_df = main_df.merge(
    rmse_df[["sub_id", "rmse_sp", "rmse_sp_z", "memory_precision", "memory_precision_z"]],
    left_on="subj_idx",
    right_on="sub_id",
    how="left"
)

# only drop the merged helper column
if "sub_id" in main_df.columns:
    main_df = main_df.drop(columns=["sub_id"])

print(main_df[["subj_idx", "rmse_sp", "memory_precision_z"]].drop_duplicates().head())

# -----------------------------
# save back to same file
# -----------------------------
main_df.to_csv(main_path, index=False)

print(f"Saved merged file to:\n{main_path}")
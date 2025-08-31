import os
import re
import pandas as pd

# ---- paths ----
in_csv = r"C:/Cluster_Github/HDDM_Vero/data_sets/data_sets_Garcia/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv"
out_dir = r"C:/Cluster_Github/HDDM_Vero/figures_dir_garcia/macleod_cluster_out/garcia_replication_ES_VAL_36/correlation"
out_csv = os.path.join(out_dir, "reuslts_ES_accuracy.csv")  # keeping your exact filename

# ---- load ----
df = pd.read_csv(in_csv)

# ---- sanity checks ----
needed_cols = {"sub_id", "corr", "phase"}
missing = needed_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ---- filter to ES phase ----
df_es = df[df["phase"].astype(str).str.upper() == "ES"].copy()
if df_es.empty:
    raise ValueError("No ES-phase rows found. Check the 'phase' column values.")

# ---- ensure corr is numeric 0/1; treat NaN/other as 0 (incorrect) ----
corr_num = pd.to_numeric(df_es["corr"], errors="coerce").fillna(0)
# If corr might be non-binary but positive for correct, threshold:
corr_num = (corr_num == 1).astype(int)
df_es["corr_num"] = corr_num

# ---- per-subject aggregates ----
g = df_es.groupby("sub_id")
n_trials = g["corr_num"].size()
sum_ones = g["corr_num"].sum()

mean_acc = sum_ones / n_trials
# std across 0/1 trials (population std, ddof=0): sqrt(p(1-p))
std = g["corr_num"].std(ddof=0)

res = pd.DataFrame({
    "mean_accuracy": mean_acc,
    "std": std,
})

# ---- rename index from sub_id -> subj.<number> (extract digits if needed) ----
def subj_label(x):
    m = re.search(r"\d+", str(x))
    num = m.group(0) if m else str(x)
    return f"subj.{num}"

res.index = res.index.map(subj_label)

# ---- save ----
os.makedirs(out_dir, exist_ok=True)
# subject labels as index, first col = mean, second = std (no explicit subject column)
res.to_csv(out_csv, index=True, header=["mean_accuracy", "std"])

print(f"Saved {len(res)} subjects to:\n{out_csv}")

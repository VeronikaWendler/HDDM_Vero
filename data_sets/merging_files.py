import pandas as pd
from pathlib import Path

# --- load both files ------------------------------------------------
base = Path("C:/Cluster_Github/HDDM_Vero/data_sets")

df_garcia = pd.read_csv(base / "data_sets_Garcia/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv")
df_ov = pd.read_csv(base / "data_sets_OV/OVParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv")

df_garcia['sub_id'] = df_garcia['sub_id'].astype(int)
df_ov['sub_id']     = df_ov['sub_id'].astype(int)

common_cols = df_garcia.columns.intersection(df_ov.columns).tolist()

df_garcia_common = df_garcia[common_cols]
df_ov_common     = df_ov[common_cols]

df_merged = pd.concat(
    [df_garcia_common, df_ov_common],
    ignore_index=True,              # 0‒n index
    sort=False                      # ooriginal column order
)
print(f"Rows in Garcia: {len(df_garcia)}, OV: {len(df_ov)}")
print(f"Rows in merged: {len(df_merged)} (should be the sum above)")
print("Any duplicate sub_id values?",
      df_merged['sub_id'].duplicated().any())   # should be False now

df_merged.to_csv(base / "combined_Garcia_OV.csv",index=False)

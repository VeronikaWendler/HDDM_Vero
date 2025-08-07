import os
import re
from pathlib import Path

BASE_DIR = Path(r"C:/Cluster_Github/HDDM_Vero/figures_dir_garcia/macleod_cluster_out")

# ---- 2.  Collect DIC values
models = {}  # { "ES_1": 123.4, ... }

for i in range(1, 42):          # ES_1 .. ES_11
    dic_file = BASE_DIR / f"garcia_replication_ES_{i}" / "diagnostics" / "DIC.txt"
    try:
        with dic_file.open() as f:
            first_line = f.readline()
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", first_line)
            if not m:
                raise ValueError("No number found in first line")
            models[f"ES_{i}"] = float(m.group())
    except FileNotFoundError:
        print(f"{dic_file} not found — skipping.")
    except ValueError as err:
        print(f"{dic_file}: {err} — skipping.")

# sort and report
print("\n Model comparison with DIC")
for model, dic in sorted(models.items(), key=lambda x: x[1]):
    print(f"{model:<5}  {dic:,.2f}")

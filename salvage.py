#!/usr/bin/env python
"""
Re‑create .pkl and .nc files from already‑finished HDDM chains.
"""

import sys, types, glob, pathlib
import hddm, dill, arviz as az      # make sure the same conda env is loaded

# ------------------------------------------------------------------
# 🔧 EDIT ME
MODEL_DIR = pathlib.Path("/workspace/models_dir_garcia")   # absolute or ~/
MODEL     = "garcia_replication_ES_13"                     # common prefix
# ------------------------------------------------------------------

# 1) provide the dummy _gdbm module so dill never crashes
sys.modules['_gdbm'] = types.ModuleType('_gdbm')

# 2) iterate over every chain already on disk
pattern = str(MODEL_DIR / f"{MODEL}_*.hddm")
for chain_file in sorted(glob.glob(pattern)):
    idx = chain_file.rsplit("_", 1)[-1].split(".")[0]      # "0", "1", …

    print(f"⟳  chain {idx}: loading {chain_file}")
    model = hddm.load(chain_file)

    # -----------------------------------------------------------------
    # write .pkl  (skip if it already exists)
    pkl_file = MODEL_DIR / f"{MODEL}_{idx}.pkl"
    if not pkl_file.exists():
        with open(pkl_file, "wb") as f:
            dill.dump(model, f)
        print(f"   ✓ wrote {pkl_file.name}")
    else:
        print(f"   • {pkl_file.name} already present")

    # -----------------------------------------------------------------
    # write .nc  (InferenceData)   – comment out if you don’t need them
    nc_file = MODEL_DIR / f"{MODEL}_{idx}.nc"
    if not nc_file.exists():
        # If you had saved infdata earlier you could load it; here we rebuild
        infdata = model.get_traces()        # this is lightweight
        az.to_netcdf(infdata, nc_file)
        print(f"   ✓ wrote {nc_file.name}")
    else:
        print(f"   • {nc_file.name} already present")

print("All done ✔")

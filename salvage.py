#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re‑create missing .pkl / .nc files from finished HDDM chains
"""

import sys, types, glob, pathlib, dill, arviz as az, hddm

MODEL_DIR  = pathlib.Path("/workspace/models_dir_garcia")
MODEL_BASE = "garcia_replication_ES_14"      # ← adjust

# stub the missing C‑extension so dill won’t choke
sys.modules['_gdbm'] = types.ModuleType('_gdbm')

for hddm_path in sorted(MODEL_DIR.glob(f"{MODEL_BASE}_*.hddm")):
    idx = hddm_path.stem.split("_")[-1]                     # “0”, “1”, …
    print(f"⟳  chain {idx}: loading {hddm_path.name}")
    mdl = hddm.load(hddm_path)

    # .pkl -----------------------------------------------------------------
    pkl_path = MODEL_DIR / f"{MODEL_BASE}_{idx}.pkl"
    if not pkl_path.exists():
        with pkl_path.open("wb") as f:
            dill.dump(mdl, f)
        print("   ✓ wrote", pkl_path.name)

    # .nc  -----------------------------------------------------------------
    nc_path = MODEL_DIR / f"{MODEL_BASE}_{idx}.nc"
    if not nc_path.exists():
        try:                                          # HDDM ≥ 0.9.8
            infdata = mdl.to_inference_data()
        except AttributeError:                       # ★ fallback
            infdata = az.from_pymc3(trace=mdl.get_traces())
        az.to_netcdf(infdata, nc_path)
        print("   ✓ wrote", nc_path.name)

print("All done ✔")


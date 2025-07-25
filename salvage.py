#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-create missing .pkl / .nc files from finished HDDM chains
"""

import sys, types, glob, pathlib
import dill, hddm, arviz as az
import pandas as pd, numpy as np            # needed for the ArviZ fallback

MODEL_DIR  = pathlib.Path("/workspace/models_dir_garcia")
MODEL_BASE = "garcia_replication_ES_14"     # ← adjust to your prefix

# --------------------------------------------------------------------------
# stub the missing gdbm C-extension so dill can pickle safely
sys.modules['_gdbm'] = types.ModuleType('_gdbm')

def traces_to_inference_data(trace_df):
    """
    Minimal, version-agnostic conversion of an HDDM trace (pandas.DataFrame)
    to ArviZ InferenceData.  Keeps only the posterior samples.
    """
    # ArviZ wants shape (chain, draw, *shape_of_param)
    # Our traces are single-chain dataframes → chain = 1
    posterior = {}
    for col in trace_df.columns:
        vals = trace_df[col].values[None, :]        # add chain axis
        posterior[col] = vals

    return az.from_dict(posterior=posterior)

# --------------------------------------------------------------------------
for hddm_path in sorted(MODEL_DIR.glob(f"{MODEL_BASE}_*.hddm")):
    idx = hddm_path.stem.split("_")[-1]
    print(f"⟳  chain {idx}: loading {hddm_path.name}")
    mdl = hddm.load(hddm_path)

    # ..........................................................  .pkl
    pkl_path = MODEL_DIR / f"{MODEL_BASE}_{idx}.pkl"
    if not pkl_path.exists():
        with pkl_path.open("wb") as f:
            dill.dump(mdl, f)
        print("   ✓ wrote", pkl_path.name)
    else:
        print("   •", pkl_path.name, "already present")

    # ..........................................................  .nc
    nc_path = MODEL_DIR / f"{MODEL_BASE}_{idx}.nc"
    if not nc_path.exists():
        # robust conversion independent of HDDM/ArviZ versions
        infdata = traces_to_inference_data(mdl.get_traces())
        az.to_netcdf(infdata, nc_path)
        print("   ✓ wrote", nc_path.name)
    else:
        print("   •", nc_path.name, "already present")

print("All done ✔")

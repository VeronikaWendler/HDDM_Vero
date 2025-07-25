#!/usr/bin/env python
import types, sys
# Stub out _gdbm so dill.save never blows up
sys.modules["_gdbm"] = types.ModuleType("_gdbm")

import os
from pathlib import Path

import hddm
import dill
import pymc
import arviz as az

# ─── EDIT THESE ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent
PREFIX    = "garcia_replication_ES_14"
TEMPLATE  = MODEL_DIR / f"{PREFIX}_0.hddm"
# ────────────────────────────────────────────────────────────────────────────

if not TEMPLATE.exists():
    raise FileNotFoundError(f"Could not find template {TEMPLATE}")

for chain in [1, 2]:
    db_path = MODEL_DIR / f"{PREFIX}_db{chain}"
    out_stem = MODEL_DIR / f"{PREFIX}_{chain}"
    print(f"\n⟳  Salvaging chain {chain} from {db_path} …")

    if not db_path.exists():
        print(f"   ⚠️  {db_path} missing, skipping.")
        continue

    # reload base model spec
    model = hddm.load(str(TEMPLATE))

    # attach that chain’s backend
    if db_path.is_dir():
        # pickle‐backend
        model.mc.db = pymc.database.pickle.load(str(db_path))
    else:
        # sqlite‐backend
        model.mc.db = pymc.database.sqlite.load(str(db_path))

    # write out the .hddm + .pkl
    model.save(str(out_stem))

    # rebuild InferenceData and write .nc
    idata = az.from_pymc3(trace=model.get_traces(), model=model.mc)
    idata.to_netcdf(str(MODEL_DIR / f"{PREFIX}_{chain}.nc"))

    print(f"   ✓  Wrote {out_stem}.hddm  {out_stem}.pkl  and .nc")

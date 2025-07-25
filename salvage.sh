#!/usr/bin/env python
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o logs/salvage_%j.out
#SBATCH -e logs/salvage_%j.err
#SBATCH --mem=200G                          # total memory for the job
#SBATCH --mail-type=ALL                     # email when job ends or fails
#SBATCH --mail-user=u04vw21@abdn.ac.uk      # university email

import types, sys, glob, pathlib, dill, hddm, arviz as az

# stub the missing C‑extension so dill won’t choke
sys.modules['_gdbm'] = types.ModuleType('_gdbm')

MODEL_DIR = pathlib.Path("/workspace/models_dir_garcia")
MODEL_BASE = "garcia_replication_ES_14"     

for hddm_path in sorted(MODEL_DIR.glob(f"{MODEL_BASE}_*.hddm")):
    idx = hddm_path.stem.split("_")[-1]
    print(f"⟳  chain {idx}: loading {hddm_path.name}")

    mdl = hddm.load(hddm_path)

    # (1) .pkl  – skip if it already exists
    pkl_path = MODEL_DIR / f"{MODEL_BASE}_{idx}.pkl"
    if pkl_path.exists():
        print("   pkl already present")
    else:
        with open(pkl_path, "wb") as f:
            dill.dump(mdl, f)
        print("   wrote", pkl_path.name)

    # (2) .nc   – re‑create only if missing
    nc_path = MODEL_DIR / f"{MODEL_BASE}_{idx}.nc"
    if nc_path.exists():
        print("   nc already present")
    else:
        try:
            infdata = mdl.to_inference_data()        # HDDM ≥ 0.9.8
        except AttributeError:
            # very old HDDM – fall back to an approximate conversion
            infdata = az.from_pymc3(trace=mdl.get_traces())

        az.to_netcdf(infdata, nc_path)
        print("   wrote", nc_path.name)

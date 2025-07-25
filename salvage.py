#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
salvage_hddm.py – re‑create *.hddm, *.pkl and *.nc files from raw HDDM
sampling databases (…_db0/1/2) when a salvage job was interrupted or only
chain‑0 was processed.

"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Third‑party
import hddm  # type: ignore

try:
    import arviz as az  # type: ignore
except ImportError:  # graceful degradation if ArviZ is unavailable
    az = None  # type: ignore
    
from pathlib import Path
import os, sys, types             #  ← already imported, keep only once

# ----------------------------------------------------------------------
# Allow an *absolute* or *relative* prefix.
# If the first CLI argument looks like a path, turn it into:
#   • working directory = <parent folder>
#   • prefix            = <file stem without path>
# ----------------------------------------------------------------------
if len(sys.argv) > 1 and "/" in sys.argv[1]:
    abs_prefix = Path(sys.argv[1]).expanduser().resolve()
    os.chdir(abs_prefix.parent)       # change into the model directory
    sys.argv[1] = abs_prefix.name     # now just "garcia_replication_ES_14"
# ----------------------------------------------------------------------

# !!! keep the rest of your script unchanged !!!


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def chains_to_salvage(prefix: str, chains: list[int] | None) -> list[int]:
    """Return the list of chain indices that still *need* salvaging.

    If *chains* is provided explicitly, just return it. If `--auto` was used,
    scan the directory for "<prefix>_db*" files that do not yet have a
    corresponding "<prefix>_* .hddm" file.
    """

    if chains is not None:
        return chains

    # auto‑detect mode
    db_files = sorted(Path().glob(f"{prefix}_db[0-9]*"))
    detected: list[int] = []
    for db in db_files:
        # extract the chain index from the suffix after "db"
        try:
            idx = int(db.name.split("_db")[-1])
        except ValueError:
            continue
        # skip if we already have an .hddm file for this chain
        if Path(f"{prefix}_{idx}.hddm").exists():
            continue
        detected.append(idx)
    return detected


# ---------------------------------------------------------------------------
# main logic
# ---------------------------------------------------------------------------

def salvage_chain(prefix: str, chain: int, template_path: Path) -> None:
    """Salvage a single chain, writing .hddm / .pkl / .nc."""

    db_name = f"{prefix}_db{chain}"
    stem = f"{prefix}_{chain}"

    if not Path(db_name).exists():
        print(f"[WARN] {db_name} not found – skipping chain {chain}.")
        return

    if Path(stem + ".hddm").exists():
        print(f"[INFO] {stem}.hddm already exists – nothing to do for chain {chain}.")
        return

    print(f"⟳  chain {chain}: salvaging from {db_name} …", flush=True)

    try:
        # 1. Load the *model specification* from chain‑0 .hddm
        model = hddm.load(str(template_path))

        # 2. Attach this chain’s trace
        model.load(db_name)

        # 3. Save in HDDM native formats
        model.save(stem)  # -> writes .hddm and .pkl

        # 4. Export NetCDF (optional)
        if az is not None:
            idata = az.from_pymc3(trace=model.get_traces(), model=model.mc)
            nc_path = Path(stem + ".nc")
            idata.to_netcdf(nc_path)
        else:
            print("[WARN] ArviZ not installed – skipping .nc export.")

        print(f"✓  wrote {stem}.hddm / .pkl" + (" / .nc" if az is not None else ""))
    except Exception:
        print(f"[ERROR] salvage for chain {chain} failed:")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Salvage HDDM databases into hddm/pkl/nc files.",
        epilog="Example: python salvage_hddm.py garcia_replication_ES_14 1 2",
    )
    parser.add_argument(
        "prefix",
        help="Common filename prefix, e.g. 'garcia_replication_ES_14'",
    )
    parser.add_argument(
        "chains",
        nargs="*",
        type=int,
        help="Chain indices to salvage (omit when using --auto).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically salvage every chain that has a db file but no .hddm.",
    )

    args = parser.parse_args()
    if args.auto and args.chains:
        parser.error("Provide either explicit chain numbers *or* --auto, not both.")

    # ---------------------------------------------------------------------
    # sanity checks
    # ---------------------------------------------------------------------
    template_path = Path(f"{args.prefix}_0.hddm")
    if not template_path.exists():
        parser.error(f"Template model {template_path} missing – cannot proceed.")

    # Figure out which chains to process
    chains = chains_to_salvage(args.prefix, args.chains if not args.auto else None)
    if not chains:
        print("[INFO] Nothing to salvage – all requested chains are already complete.")
        return

    # Process each chain separately (re‑loading template each time keeps memory low)
    for ch in chains:
        salvage_chain(args.prefix, ch, template_path)


if __name__ == "__main__":
    main()

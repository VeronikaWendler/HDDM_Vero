#!/usr/bin/env python
"""
list_hddm_params.py  FILE_PATTERN  [--filter REGEX]  [--posterior]

Examples
--------
# Show *all* node names in a single model
./list_hddm_params.py  garcia_replication_ES_12_0.hddm

# Wildcard over the five chains of model 12
./list_hddm_params.py  'garcia_replication_ES_12_*.hddm'

# Only the a‑parameters
./list_hddm_params.py  garcia_replication_ES_12_0.hddm  --filter '^a'

# Look into the ArviZ InferenceData instead of .hddm
./list_hddm_params.py  garcia_replication_ES_12_0.nc  --posterior
"""
import argparse, glob, os, re, sys, textwrap

def list_from_hddm(path, regex=None):
    import hddm
    m = hddm.load(path)
    names = m.nodes_db.index
    return [n for n in names if not regex or re.search(regex, n)]

def list_from_nc(path, regex=None):
    import arviz as az
    idata = az.from_netcdf(path)
    names = idata.posterior.data_vars.keys()
    return [n for n in names if not regex or re.search(regex, n)]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent(__doc__))
parser.add_argument('file_pattern', help='*.hddm or *.nc (wildcards OK)')
parser.add_argument('--filter',   help='regex to keep only matching names')
parser.add_argument('--posterior', action='store_true',
                    help='Force .nc mode (otherwise .hddm)')
args = parser.parse_args()

files = sorted(glob.glob(args.file_pattern))
if not files:
    sys.exit(f'No files match {args.file_pattern!r}')

for f in files:
    try:
        if args.posterior or f.endswith('.nc'):
            names = list_from_nc(f, args.filter)
        else:
            names = list_from_hddm(f, args.filter)
    except Exception as e:
        print(f'⚠️  Could not load {f}: {e}', file=sys.stderr)
        continue

    print(f'\n### {os.path.basename(f)}  ({len(names)} params)')
    for n in names:
        print('  ', n)

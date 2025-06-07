#!/bin/bash
#SBATCH --job-name=hddm_fit            # a short name
#SBATCH --partition=standard           # GPU partition
#SBATCH --cpus-per-task=8               # number of CPU cores for your chains
#SBATCH --mem=128G                      # total memory for the job
#SBATCH --time=23:30:00                # walltime hh:mm:ss
#SBATCH -o logs/slurm.%j.out           # STDOUT goes to this file
#SBATCH -e logs/slurm.%j.err           # STDERR goes to this file
#SBATCH --mail-type=END,FAIL            # email when job ends or fails
#SBATCH --mail-user=u04vw21@abdn.ac.uk  # your university email

# ── 1) Load the Singularity module ────────────────────────────────────────────
module load singularity/3.8.5

# ── 2) Define variables ───────────────────────────────────────────────────────
IMAGE=$HOME/containers/hddm_latest.sif
PROJECT=$HOME/sharedscratch/HDDM_Vero

export PROJECT_DIR=/workspace
export MPLBACKEND=Agg

# ── 3) Run inside the container, with GPU support ─────────────────────────────
singularity exec \
    --bind ${PROJECT}:/workspace \
    ${IMAGE} \
    python /workspace/aDDM_Garcia_run_all_mod.py

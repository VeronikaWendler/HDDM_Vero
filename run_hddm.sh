#!/bin/bash
#SBATCH --job-name=hddm_garcia
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/hddm_%j.out
#SBATCH --error=logs/hddm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=VAW508@student.bham.ac.uk

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module purge
module load bb-singularity-conf/live

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplcache"
mkdir -p "$MPLCONFIGDIR"

IMAGE="$HOME/containers/hddm_latest.sif"
CODE_DIR="$HOME/projects/HDDM_Vero"
DATA_DIR_HOST="/rds/homes/v/vaw508/projects/HDDM_Vero/data_sets/data_sets_Garcia"
OUT_DIR_HOST="/rds/projects/z/zhanglp-vwendler-core/HDDM_Vero/derivatives/hddm"

mkdir -p "${OUT_DIR_HOST}"/{models,figures,logs}

apptainer exec --cleanenv \
  --bind "${CODE_DIR}:/workspace" \
  --bind "${DATA_DIR_HOST}:/data" \
  --bind "${OUT_DIR_HOST}:/out" \
  --env PROJECT_DIR="/workspace" \
  --env DATA_FILE="/data/GarciaParticipants_Eye_Response_Feed_Allfix_addm_OV_Abs_CCT.csv" \
  --env MODEL_DIR="/out/models" \
  --env FIG_DIR="/out/figures" \
  --env LOG_DIR="/out/logs" \
  --env N_JOBS="${SLURM_CPUS_PER_TASK}" \
  "${IMAGE}" \
  python /workspace/aDDM_OV_run_all_mod.py

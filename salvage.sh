#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -o logs/salvage_%j.out
#SBATCH -e logs/salvage_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=u04vw21@abdn.ac.uk

module load singularity/3.8.5

IMAGE=$HOME/containers/hddm_latest.sif
PROJECT=$HOME/sharedscratch/HDDM_Vero
PREFIX=garcia_replication_ES_14

singularity exec \
  --bind $PROJECT:/workspace \
  $IMAGE \
  bash -lc "\
    cd /workspace/models_dir_garcia && \
    python /workspace/salvage.py --auto $PREFIX \
  "

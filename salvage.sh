#!/bin/bash
###############################################################
#  Slurm wrapper that re‑creates the missing *.pkl / *.nc files
#  for an HDDM model that has already produced *.hddm chains.
###############################################################

#SBATCH --job-name=salvage_ES_13          #  anything you like
#SBATCH --partition=compute               #  same partition you used
#SBATCH --cpus-per-task=1                 #  one core is enough
#SBATCH --mem=4G                          #  very small footprint
#SBATCH --time=00:15:00                   #  plenty of time
#SBATCH -o logs/salvage_%j.out            #  STDOUT
#SBATCH -e logs/salvage_%j.err            #  STDERR
#SBATCH --mail-type=END,FAIL              #  optional
#SBATCH --mail-user=u04vw21@abdn.ac.uk    #  optional

module load singularity/3.8.5             #  same module as before
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

IMAGE=$HOME/containers/hddm_latest.sif
PROJECT=$HOME/sharedscratch/HDDM_Vero     # folder that holds models_dir_garcia
# -------------------------------------------------------------

mkdir -p logs

# run the helper inside the very same container
singularity exec \
  --bind ${PROJECT}:/workspace \
  ${IMAGE} \
  python /workspace/salvage.py

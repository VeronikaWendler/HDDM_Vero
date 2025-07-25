#!/bin/bash
#SBATCH --partition=compute                 # CPU partition (on MacLeod, not sure about Maxwell)
#SBATCH --cpus-per-task=8                   # number of CPU cores for chains
#SBATCH --mem=200G                          # total memory for the job
#SBATCH -o logs/salvage_%j.out
#SBATCH -e logs/salvage_%j.err
#SBATCH --mem=200G                          # total memory for the job
#SBATCH --mail-type=ALL                     # email when job ends or fails
#SBATCH --mail-user=u04vw21@abdn.ac.uk      # university email

#SBATCH --mail-type=ALL
#SBATCH --mail-user=u04vw21@abdn.ac.uk

module load singularity/3.8.5

IMAGE=$HOME/containers/hddm_latest.sif       # make sure it is readable
PROJECT=$HOME/sharedscratch/HDDM_Vero        # contains models_dir_garcia
PREFIX=garcia_replication_ES_14              # *basename only*

singularity exec \
  --bind "${PROJECT}:/workspace" \
  --pwd  /workspace/models_dir_garcia \
  "${IMAGE}" \
  python /workspace/salvage.py --auto "${PREFIX}"
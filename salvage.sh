#!/bin/bash
#SBATCH --partition=compute                 # CPU partition (on MacLeod, not sure about Maxwell)
#SBATCH --cpus-per-task=8                   # number of CPU cores for chains
#SBATCH --mem=200G                          # total memory for the job
#SBATCH -o logs/salvage_%j.out
#SBATCH -e logs/salvage_%j.err
#SBATCH --mem=200G                          # total memory for the job
#SBATCH --mail-type=ALL                     # email when job ends or fails
#SBATCH --mail-user=u04vw21@abdn.ac.uk      # university email


module load singularity/3.8.5

IMAGE=$HOME/containers/hddm_latest.sif          # ⟨container image⟩
PROJECT=$HOME/sharedscratch/HDDM_Vero           # ⟨folder to bind⟩
PREFIX=garcia_replication_ES_14                 # ⟨model prefix⟩

singularity exec \
  --bind ${PROJECT}:/workspace \
  ${IMAGE} \
  python /workspace/salvage_hddm.py --auto ${PREFIX}
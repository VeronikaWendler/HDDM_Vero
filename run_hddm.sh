#!/bin/bash
#SBATCH --partition=compute                 # CPU partition (on MacLeod, not sure about Maxwell)
#SBATCH --cpus-per-task=8                   # number of CPU cores for chains
#SBATCH --mem=200G                          # total memory for the job
#SBATCH -o logs/slurm.%j.out                # STDOUT goes to this file
#SBATCH -e logs/slurm.%j.err                # STDERR goes to this file
#SBATCH --mail-type=ALL                     # email when job ends or fails
#SBATCH --mail-user=u04vw21@abdn.ac.uk      # university email

#Singularity module 
module load singularity/3.8.5

# Define variables 
IMAGE=$HOME/containers/hddm_latest.sif
PROJECT=$HOME/sharedscratch/HDDM_Vero

export PROJECT_DIR=/workspace
export MPLBACKEND=Agg

# Run inside the container
singularity exec \
    --bind ${PROJECT}:/workspace \
    ${IMAGE} \
    python /workspace/aDDM_OV_run_all_mod.py

# How to get this repo running on your university's cluster 

# 1. confirm that you can log into your cluster (at your university in Hamburg I assume) via ssh ... etc. or whatever access system you use
# 2. Assuming you can log in, create a folder (e.g. projects) and create another folder (outisde of the first one) called 'containers' or 'images' or something similar
# 3. Get my repository onto your cluster into the project folder using this command: git clone https://github.com/VeronikaWendler/HDDM_Vero.git (let me know if this sept does not work; it actually should because my repo is public)
# 4. cd HDDM_Vero and check if everything is in there
# 5. check if you use apptainer or singularity using: module avail apptainer singularity
# 6. use: module load apptainer or module load singularity
# 7. use: (singularity) or apptainer pull ~/containers/hddm_latest.sif docker://hcp4715/hddm:latest     (if this doesn't work it might be an internet issue, just let me know)
# 8. Verify if it's there using: ls -lah ~/containers/hddm_latest.sif
# 9. use: (singularity) or apptainer exec ~/containers/hddm_latest.sif python -V
# 10. Create a Hamburg Slurm script (or modify my run_hddm.sh script) like so, change apptainer to singularity if you are also using singularity:

#!/bin/bash
#SBATCH --job-name=hddm_fit
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=ALL                     # email when job ends or fails (if it also works like that on the Hamburg cluster)
#SBATCH --mail-user=u04vw21@abdn.ac.uk      # university email (your email form Hamburg, I assume you do not need to modify it, in Birmingham we need to write the email in lowercase )



set -euo pipefail
mkdir -p logs

module purge
module load apptainer

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mplcache

IMAGE=$HOME/containers/hddm_latest.sif
PROJECT=$HOME/projects/HDDM_Vero

apptainer exec \
  --bind ${PROJECT}:/workspace \
  ${IMAGE} \
  python /workspace/aDDM_Garcia_run_all_mod.py
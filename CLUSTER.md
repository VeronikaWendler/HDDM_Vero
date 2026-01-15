# Running Veronika’s HDDM_Vero repository on the University of Hamburg cluster (I believe you guys use Hummel-2)

Steps involved:

1. Repository: https://github.com/VeronikaWendler/HDDM_Vero (public, no special access needed from Veronika)

If you want to change code, you can do it locally in your clone without ever pushing. Forking is optional and only needed if you want your own GitHub copy.

2. Make sure you have a UHH HPC account. You need an account that is enabled for the Hamburg HPC (Hummel-2 I believe). If you can’t log in, none of the later steps matter

3.You need network access (VPN may be required) - I use BigIP Edge Client (f5). Depending on where you are connecting from, you may need the University VPN to reach the login gateways.

4. Create an SSH key on your laptop (recommended if not already set up). On your laptop (Mac/Linux/Windows PowerShell with OpenSSH) do this:
`ssh-keygen -t ed25519 -C "your_email@uni-hamburg.de" (or whatever exact email you guys use)`

5. Press Enter to accept defaults. This creates:
- private key: ~/.ssh/id_ed25519
- public key: ~/.ssh/id_ed25519.pub

6. Show the public key using this command:

`cat ~/.ssh/id_ed25519.pub`

7. Add that key to the method UHH requires for SSH public key authentication (I think UHH documents public key authentication for their HPC systems: https://www.rrz.uni-hamburg.de/en/services/hpc/basics/ssh/pubkeys.html)

8. Log in (multi hop login is expected): I believe Hummel-2 uses login gateway nodes and then front-end nodes for actual work (prepare jobs, submit jobs)From your laptop do: `ssh "UHH name"@hummel3.rrz.uni-hamburg.de` or something like that.

9. After you are on the gateway, go to a front end node: `ssh front1` (or `ssh front2`). You should be on a front-end node where you can load modules, pull containers, and submit Slurm jobs.

10. Now, create a folder layout on the cluster. On the front-end node do for example:

```bash
mkdir -p $HOME/projects
mkdir -p $HOME/containers
mkdir -p $HOME/projects/HDDM_Vero/logs
```
Explanation:

projects holds code + outputs

containers holds .sif images

logs holds Slurm stdout/stderr

Get my repository onto the cluster (no permissions needed): Go to your projects directory and clone:

bash
Code kopieren
cd $HOME/projects
git clone https://github.com/VeronikaWendler/HDDM_Vero.git
cd HDDM_Vero
Check the repo contents:

ls -lah

If git is not available:

module avail git
module load git

Note on permissions:

Cloning a public repo does not require any GitHub access from me.

Only pushing back to my repo would require her to grant write permissions (not needed for running). Hence, what you could do alternatively to step 11. (alternatively to this: git clone https://github.com/VeronikaWendler/HDDM_Vero.git you could simply fork the repostory first so you would have your own version on your GitHub and could do something like:

git remote -v
git remote add myfork https://github.com/"Your name on git"/HDDM_Vero.git
git push -u myfork hamburg_run

(You can try both ways, I believe forking first and then running my stuff is probably better)

Set up the container runtime (Apptainer or Singularity)

Check what is available on your cluster by using these commands:

module avail apptainer singularity

Load one (apptainer or singularity):

module load apptainer

If Apptainer is not available, use Singularity using the command:

module load singularity

Pull the HDDM container image (one-time only; developed by Lei Zhang's student: Hu Chuan-Peng: https://github.com/hcp4715):

On the front-end node:

If using Apptainer do this:

apptainer pull $HOME/containers/hddm_latest.sif docker://hcp4715/hddm:latest

If using Singularity do:

singularity pull $HOME/containers/hddm_latest.sif docker://hcp4715/hddm:latest

Verify the image exists (in the container folder that you previously created):

ls -lah $HOME/containers/hddm_latest.sif

Test that Python runs inside the container:

Apptainer (use this command):

apptainer exec $HOME/containers/hddm_latest.sif python -V

Singularity (use this command):

singularity exec $HOME/containers/hddm_latest.sif python -V

If the pull fails (if this step above fails please let me know, I would be interested as it should not fail):
This is usually because compute/login nodes may have restricted internet. The workaround is to pull the .sif on a machine that can reach Docker Hub and then copy the file to $HOME/containers using scp.

I think your Hummel documentation highlights these typical job characteristics for slurm:

the smallest job size is 8 CPU cores

they recommend using --export=NONE and source /sw/batch/init.sh.

--mail-user could be ignored; and the routing may be handled via the Hummel-2 mailing list configuration

So: do not worry if email settings behave differently than in my Aberdeen script

Create the reproducible Slurm script (Hamburg version)

Go into my repo on the cluster and create a job script using (create a logs file to monitor where the code mail fail or succeed):

bash
Code kopieren
cd $HOME/projects/HDDM_Vero
mkdir -p logs
Create a sulrm script (either use a modified version of my 'run_hddm.sh' or create a differently named one) like this (choose Apptainer or Singularity depending on what you loaded earlier; below is the start of the file):

bash
Code kopieren
#!/bin/bash
#SBATCH --job-name=hddm_fit
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                   # you can also increase this here, e.g. a 100 or something
#SBATCH --output=logs/%x.%j.out     # these folders monitor output and error
#SBATCH --error=logs/%x.%j.err
#SBATCH --export=NONE                
#SBATCH --mail-type=ALL                  # get mails for everything (start,end, failure etc..)
#SBATCH --mail-user=u04vw21@abdn.ac.uk      # university email (here it would be your Hamburg email)


set -euo pipefail
mkdir -p logs

source /sw/batch/init.sh

module purge
module load apptainer             # again, use singularity if you do not have apptainer

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mplcache

IMAGE=$HOME/containers/hddm_latest.sif
PROJECT=$HOME/projects/HDDM_Vero

apptainer exec \
  --bind ${PROJECT}:/workspace \
  ${IMAGE} \
  python /workspace/aDDM_Garcia_run_all_mod.py               # this file would run the modelling for the first study (workspace is basically pwd)


(this is the end of the .sh script you must have in the repo on the cluster)
If you must use Singularity, change only two things:

module load apptainer → module load singularity

apptainer exec → singularity exec

Submit the job:

sbatch run_hddm_hummel2.slurm

Check whether it is running:

squeue -u $USER

Inspect logs:

bash
Code kopieren
ls -lah logs
tail -n 50 logs/hddm_fit.*.err
tail -n 50 logs/hddm_fit.*.out
or just cd logs and do cat 'name of the job.err' or 'name of the job.out'

My most common failures (data / paths) and the fix
Why this happens

On Aberdeen I bind-mounted:

$HOME/sharedscratch/HDDM_Vero to /workspace

On Hamburg you would probably bind-mount:

$HOME/projects/HDDM_Vero to /workspace

This part is nice but Python scripts could still assume that the data files exist in certain subfolders (and I forgot if I created them on the cluster or uploaded them)

So, before running a full job I would check if the script expects folders that do not exist, and create them if they are absent (I think I am gitignoring them so just create them to be sure:

bash
Code kopieren
mkdir -p models_dir_garcia
mkdir -p models_dir_OV

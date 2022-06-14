import os
import shutil
import subprocess
import textwrap
import time
import datetime

#############################################################################
# These variables are user-dependent and assume Frontera is the machine
#############################################################################
ncpus = 56
ACCOUNT = '[ADD YOUR FRONTERA ACCOUNT HERE]'

HOME = '[ADD YOUR FRONTERA HOME DIRECTORY HERE]'
code_dir = f'{HOME}/spectralles'
notebook = f'{HOME}/LAB_NOTEBOOK/spectralles'

WORK = '[ADD YOUR FRONTERA SCRATCH/WORK DRIVE DIRECTORY HERE]'
workdir = f'{WORK}/DNS'
os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)

this_script = os.path.basename(__file__)
today = time.strftime('%Y_%m_%d')

jobname = "DNS"
slurm_script = f"slurm.{jobname}"

ntasks = X
walltime = str(datetime.timedelta(hours=Y))
nstripes = Z   # Lustre striping, follow Frontera recommendations

queue = 'development'

#############################################################################
# Shouldn't need to touch anything below here
#############################################################################
subprocess.run(['lfs', 'setstripe', '-c', str(nstripes), workdir])

script = f"""\
#!/bin/bash
# ---------------------------------------------------------------------
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={jobname}
#SBATCH --output=%j.{jobname}.out      # Name of stdout output file
#SBATCH --error=%j.{jobname}.out       # Name of stderr error file
#SBATCH --qos={queue}

#SBATCH --nodes={nnodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time={walltime}       # Desired walltime (hh:mm:ss)

#SBATCH --mail-type=all
#SBATCH --mail-user=[ADD YOUR EMAIL ADDRESS HERE]
# ---------------------------------------------------------------------

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

JOBDIR={notebook}/{today}/${{SLURM_JOB_ID}}.{jobname}

cp ${{JOBDIR}}/spectralles.py ./
cp ${{JOBDIR}}/single_dns_run.py ./

ibrun python single_dns_run.py
"""

with open(slurm_script, 'wt') as fh:
    fh.write(textwrap.dedent(script))

print(f"submitting {slurm_script}...")
output = subprocess.run(['sbatch', slurm_script],
                        capture_output=True, text=True).stdout
print(output)

jobid = output.split()[-1]
jobdir = f"{notebook}/{today}/{jobid}.{jobname}"
os.makedirs(jobdir, exist_ok=True)

shutil.copy(slurm_script, jobdir)
shutil.copy(f"{code_dir}/{this_script}", jobdir)
shutil.copy(f"{code_dir}/single_dns_run.py", jobdir)
shutil.copy(f"{code_dir}/spectralles.py", jobdir)

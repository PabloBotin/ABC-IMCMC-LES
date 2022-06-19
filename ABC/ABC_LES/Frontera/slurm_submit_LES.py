from math import ceil
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
workdir = f'{WORK}/LES_ABC'
os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)

this_script = os.path.basename(__file__)
today = time.strftime('%Y_%m_%d')

jobname = "LES_ABC_runs"
slurm_script = f"slurm.{jobname}"

#############################################################################
# These variables are about the LES and how long it takes to run, etc.
#############################################################################
N_les = 32  # spectral-space mesh size of LES run (LES uses padded FFTs)
N_samples_per_param = 7

ntasks_per_run = 4       # try using 4 tasks if N=32, 32 if N=64
nruns_in_parallel = 14   # this sets the total number of tasks/nodes needed
wt_per_run = 4           # walltime in minutes: use ~4 for N=32, ~20 for N=64

nstripes = 1           # Lustre striping, generally use 1 for LES
queue = 'development'

#############################################################################
# Shouldn't need to touch anything below here
#############################################################################
N_runs = N_samples_per_param ** 4
N_serial_runs = N_runs // nruns_in_parallel

ntasks = ntasks_per_run * nruns_in_parallel
ntasks_per_node = ntasks_per_run * (ncpus // ntasks_per_run)
nnodes = ceil(ntasks / ntasks_per_node)

wt_mins = wt_per_run * N_serial_runs
walltime = str(datetime.timedelta(minutes=wt_mins))

script = f"""\
#!/bin/bash
# ---------------------------------------------------------------------
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={jobname}
#SBATCH --output=%j.{jobname}.out      # Name of stdout output file
#SBATCH --error=%j.{jobname}.out       # Name of stderr error file
#SBATCH --qos={queue}

#SBATCH --nodes={nnodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --time={walltime}       # Desired walltime (hh:mm:ss)

#SBATCH --begin=now+1           # delays eligible start of job by 1 second
#SBATCH --mail-type=all
#SBATCH --mail-user=[ADD YOUR EMAIL ADDRESS HERE]
# ---------------------------------------------------------------------

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

JOBDIR={notebook}/{today}/${{SLURM_JOB_ID}}.{jobname}

cp ${{JOBDIR}}/spectralles.py ./
cp ${{JOBDIR}}/mpmd_les_abc.py ./

ibrun python mpmd_les_abc.py {N_les} {N_samples_per_param} {ntasks_per_run}
"""

with open(slurm_script, 'wt') as fh:
    fh.write(textwrap.dedent(script))

print(f"submitting {slurm_script}...")
output = subprocess.run(['sbatch','--parsable', slurm_script],
                        capture_output=True, text=True).stdout
print(output)

jobid = output.split()[-1]
jobdir = f"{notebook}/{today}/{jobid}.{jobname}"
print(f'making job directory {jobdir}')
os.makedirs(jobdir, exist_ok=True)

shutil.copy(slurm_script, jobdir)
shutil.copy(f"{code_dir}/{this_script}", jobdir)
shutil.copy(f"{code_dir}/mpmd_les_abc.py", jobdir)
shutil.copy(f"{code_dir}/spectralles.py", jobdir)

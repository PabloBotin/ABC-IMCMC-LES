"""Uniform-grid sampling for classical ABC parameter estimation for LES.

"""
from mpi4py import MPI

import os
import sys
from math import pi

import numpy as np

from spectralles import Config, SpectralLES

wcomm = MPI.COMM_WORLD

# Step 0. Define the LES and ABC parameters
# -----------------------------------------------------------------------------
K = 512     # presumed bandwidth of future DNS run
kfLow = 3   # need to force at higher modes to get better stationarity
kfHigh = 5  # this is inclusive
eps = 1.0   # JHU used 0.103, for whatever reason, but 1.0 is much nicer
eta = 2 / K  # expected Kolmogorov scale of DNS
nu = (eps * eta**4)**(1/3)  # viscosity required by eps and eta
nu = round(nu, 9)           # rounded to 9 decimal places
ell = 2 * pi / 4  # integral scale will be around k = 4 if forcing from 3 to 5
tau = (ell**2/eps)**(1/3)   # coarse approximation

N_dealiased = int(sys.argv[1])  # get N from slurm script via command line
N_samples_per_param = int(sys.argv[2])  # get this via command line as well

# redirect all spectralLES file outputs somewhere else
odir = f'{os.getcwd()}/data'

# Olga's coefficients based on stress alone
C1, C2, C3, C4 = -0.043, 0.018, 0.036, 0.036

ntasks = wcomm.size
N_tasks_per_run = int(sys.argv[3])
N_parallel_runs = ntasks // N_tasks_per_run

# Split the default MPI communicator into several comms that will run jobs
# in parallel, simultaneously
sub_comm_id = wcomm.rank // N_tasks_per_run
sub_comm_rank = wcomm.rank % N_tasks_per_run
subcomm = wcomm.Split(sub_comm_id, sub_comm_rank)
init_file = f'dyn_smag_{sub_comm_id}.checkpoint.h5'

# Step 1. Generate reference data D
# -----------------------------------------------------------------------------
smag_cfg = Config(pid=f'dyn_smag_{sub_comm_id}',
                  model='dyn_smag',
                  N=N_dealiased,
                  epsilon=eps,
                  nu=nu,
                  kfLow=kfLow,
                  kfHigh=kfHigh,
                  KE_init=round(2*tau*eps, 3),   # assumes tau ~ tke/eps
                  kExp_init=4.0,
                  init_cond='random',
                  tlimit=round(3*tau, 3),
                  odir=odir,
                  idir=odir,
                  )

# All subcomms run their own case of Dyn Smag
if wcomm.rank == 0:
    print('running initial Dynamic Smag interval', flush=True)
SpectralLES(smag_cfg, comm=subcomm).run_quiet()

config = Config(pid='baseline',
                model='4term',
                N=N_dealiased,
                epsilon=eps,
                nu=nu,
                kfLow=kfLow,
                kfHigh=kfHigh,
                C1=C1,
                C2=C2,
                C3=C3,
                C4=C4,
                init_cond='file',
                init_file=init_file,
                tlimit=round(2*tau, 3),
                odir=odir,
                idir=odir,
                # limit is about 3-4x the cycles needed for baseline
                cycle_limit=800*(N_dealiased//32),
                )

# Only the first subcomm runs the baseline 4term case
if wcomm.rank == 0:
    print('running baseline LES', flush=True)
if sub_comm_id == 0:
    SpectralLES(config, comm=subcomm).run_quiet()

# Steps 2 & 3. Uniform gridded sampling
# -----------------------------------------------------------------------------
C_limits = [(0.5*C, 1.5*C) for C in (C1, C2, C3, C4)]

C = np.empty([4, N_samples_per_param])
for i in range(4):
    C[i] = np.linspace(C_limits[i][0], C_limits[i][1], N_samples_per_param)

C_grids = np.meshgrid(*C, indexing='ij')

# Step 4. Run LES for all sets of coefficients
# -----------------------------------------------------------------------------
N_runs = C_grids[0].size  # should be equal to N_samples**4
Cs = np.empty(4)
success = np.zeros(N_runs, dtype=bool)

smag_cfg.init_cond = 'file'
smag_cfg.init_file = init_file
smag_cfg.tlimit = round(tau, 3)

rstart = sub_comm_id
rend = N_runs
rstride = N_parallel_runs

for r in range(rstart, rend, rstride):
    if sub_comm_rank == 0:
        print(f'running test {r}', flush=True)

    for p in range(4):
        Cs[p] = C_grids[p].flat[r]

    config.pid = f'abc_run_{r}'
    config.C1 = C_grids[0].flat[r]
    config.C2 = C_grids[1].flat[r]
    config.C3 = C_grids[2].flat[r]
    config.C4 = C_grids[3].flat[r]

    SpectralLES(smag_cfg, comm=subcomm).run_quiet()
    success[r] = SpectralLES(config, comm=subcomm).run_quiet()

wcomm.Barrier()

# Perform a "Logical Or" reduction operation across all MPI tasks to get
# a single record of which runs were successful
if wcomm.rank == 0:
    wcomm.Reduce(MPI.IN_PLACE, success, op=MPI.LOR)

    # save the success array to file in numpy save format
    np.save(f'{odir}/success_array.npy', success)

else:
    wcomm.Reduce(success, None, op=MPI.LOR)

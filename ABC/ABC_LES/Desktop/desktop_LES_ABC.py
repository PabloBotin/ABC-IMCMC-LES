"""Uniform-grid sampling for classical ABC parameter estimation for LES.

"""
from mpi4py import MPI

import os
import time
from math import pi

import numpy as np
import h5py

# from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from spectralles import Config, SpectralLES

comm = MPI.COMM_WORLD

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

N_dealiased = 16  # spectral-space mesh size of LES run (LES uses padded FFTs)
N_params = 4      # number of LES coefficients
N_samples_per_param = 3  # mesh size of uniform coefficient samples

# redirect all spectralLES file outputs somewhere else
odir = f'{os.getcwd()}/data'

# Olga's coefficients based on stress alone
C1, C2, C3, C4 = -0.043, 0.018, 0.036, 0.036

if comm.rank == 0:
    print(f'Correct solution: C1={C1:0.3f}, C2={C2:0.3f},  '
          f'C3={C3:0.3f},  C4={C4:0.3f}')

# Step 1. Generate reference data D
# -----------------------------------------------------------------------------
smag_cfg = Config(pid='dyn_smag',
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
                init_file='dyn_smag.checkpoint.h5',
                tlimit=round(2*tau, 3),
                odir=odir,
                idir=odir,
                )

SpectralLES(smag_cfg).run_quiet()
SpectralLES(config).run_quiet()

# Steps 2 & 3. Uniform gridded sampling
# -----------------------------------------------------------------------------
C_limits = [(0.5*C, 1.5*C) for C in (C1, C2, C3, C4)]

C = np.empty([N_params, N_samples_per_param])
for i in range(N_params):
    C[i] = np.linspace(C_limits[i][0], C_limits[i][1], N_samples_per_param)

C_grids = np.meshgrid(*C, indexing='ij')

# Step 4. Run LES for all sets of coefficients
# -----------------------------------------------------------------------------
N_runs = C_grids[0].size  # should be equal to N_samples**N_params
Cs = np.empty(N_params)
success = np.zeros(N_runs, dtype=bool)

smag_cfg.init_cond = 'file'
smag_cfg.init_file = 'dyn_smag.checkpoint.h5'
smag_cfg.tlimit = round(tau, 3)

tstart = time.time()

for r in range(N_runs):
    for p in range(N_params):
        Cs[p] = C_grids[p].flat[r]

    config.pid = f'abc_run_{r}'
    config.C1 = C_grids[0].flat[r]
    config.C2 = C_grids[1].flat[r]
    config.C3 = C_grids[2].flat[r]
    config.C4 = C_grids[3].flat[r]

    # 1st) run Dyn Smag for another tau to get a new initial condition
    SpectralLES(smag_cfg).run_quiet()

    # 2nd) run the 4-term model from this DS solution for 2 tau
    # Note: run_quiet() and run_verbose() now return True if the simulation
    # finished and False if it failed for any reason
    success[r] = SpectralLES(config).run_quiet()

tend = time.time()

###############################################################################
# FROM HERE ON EVERYTHING SHOULD BE SERIAL
###############################################################################
if comm.rank == 0:
    print(f'Loop took {tend-tstart:0.2f} s')

    filename = f'{odir}/postprocessing.h5'
    out_fh = h5py.File(filename, 'w')
    out_fh['success'] = success

    # Step 5. Postprocess distance
    # -------------------------------------------------------------------------
    filename = f'{odir}/baseline.statistics.h5'
    fh = h5py.File(filename)

    # there are only outputs '000' and '001' corresponding to the random
    # initial condition and the final solution at t_sim = tlimit
    logEk_base = np.log(fh['001/Ek'][:-1])
    base_pdfs = []
    for i, S in enumerate(['Pi', 'sigma_11', 'sigma_12', 'sigma_13']):
        hist = fh[f'001/{S}/hist'][:]
        edges = fh[f'001/{S}/edges'][:]
        x = 0.5 * (edges[:-1] + edges[1:])
        density = gaussian_kde(x, weights=hist)
        base_pdfs.append((x, np.log(density(x))))

    fh.close()

    dist = np.empty(N_runs)
    for r in range(N_runs):
        if success[r]:
            filename = f'{odir}/abc_run_{r}.statistics.h5'
            fh = h5py.File(filename)

            Ek = fh['001/Ek'][:-1]

            # compute L2 norm of Ek
            dist[r] = np.sum((np.log(Ek) - logEk_base)**2)

            for i, S in enumerate(['Pi', 'sigma_11', 'sigma_12', 'sigma_13']):
                hist = fh[f'001/{S}/hist'][:]
                edges = fh[f'001/{S}/edges'][:]
                x = 0.5 * (edges[:-1] + edges[1:])
                density = gaussian_kde(x, weights=hist)
                x_b, y_b = base_pdfs[i]
                y = np.log(density(x_b))  # evaluate PDF at baseline x values

                # add L2 norm of each PDF to L2 norm of Ek for total distance
                dist[r] += np.sum((y - y_b)**2)

            fh.close()

        else:
            # run not successful, set distance to infinity
            dist[r] = np.inf

    out_fh['distance'] = dist

    # Step 6. Accept X% of all runs
    # -------------------------------------------------------------------------
    N_accept = int(0.5 * N_runs)
    R_sort = np.argsort(dist)   # this is an array of indices that would sort dist
    epsilon = dist[R_sort[N_accept-1]]  # dist <= epsilon for all accepted runs

    # copy accepted C values from uniform grid
    C_accept = np.empty((N_params, N_accept))
    for p in range(N_params):
        C_accept[p] = C_grids[p].flat[R_sort[:N_accept]]

    # Step 7. Get Kernel Density Estimation (KDE) for accepted data
    # -------------------------------------------------------------------------
    density = gaussian_kde(C_accept)
    x = np.vstack([C.ravel() for C in C_grids])
    jpdf = np.reshape(density(x), C_grids[0].shape)

    out_fh['posterior'] = jpdf

    # Step 8. Get the Maximum A Posteriori (MAP) set of coefficients from the KDE
    # -------------------------------------------------------------------------
    map0, map1, map2, map3 = np.where(jpdf == jpdf.max())
    # np.where always outputs arrays, even for single values or empty results
    map0 = map0[0]
    map1 = map1[0]
    map2 = map2[0]
    map3 = map3[0]

    c_map = np.empty(N_params)
    c_map[0] = C[0, map0]
    c_map[1] = C[1, map1]
    c_map[2] = C[2, map2]
    c_map[3] = C[3, map3]

    print(f'MAP Coefficients: C1={c_map[0]:0.3f}, C2={c_map[1]:0.3f},  '
          f'C3={c_map[2]:0.3f},  C4={c_map[3]:0.3f}')

    out_fh.close()

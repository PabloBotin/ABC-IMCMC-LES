from mpi4py import MPI

import os
from math import pi

from spectralles import Config, SpectralLES

K = 512     # presumed "bandwidth" of future DNS run
kfLow = 3   # need to force at higher modes to get better stationarity
kfHigh = 5  # this is inclusive
eps = 1.0   # JHU used 0.103, for whatever reason, but 1.0 is much nicer
eta = 2 / K  # expected Kolmogorov scale of DNS
nu = (eps * eta**4)**(1/3)  # viscosity required by eps and eta
ell = 2 * pi / 4  # integral scale will be around k = 4 if forcing from 3 to 5
tau = (ell**2/eps)**(1/3)   # coarse approximation

# redirect all spectralLES file outputs somewhere else
odir = f'{os.getcwd()}/data'

N_dealiased = K  # if K=512, yields 768^3 physical-space mesh and kmax*eta=1

config = Config(pid='K{K}_DNS',
                model='dns',
                N=N_dealiased,
                epsilon=eps,
                nu=nu,
                kfLow=kfLow,
                kfHigh=kfHigh,
                KE_init=round(2*tau*eps, 3),   # assumes tau ~ tke/eps
                kExp_init=4.0,
                init_cond='random',
                tlimit=round(5*tau, 3),  # need to run DNS out a bit longer
                odir=odir,
                idir=odir,
                dt_rst=round(0.5*tau, 3),  # give yourself some restarts
                dt_stat=round(tau/16, 3),  # give yourself spectra outputs at high rate
                )

SpectralLES(config).run_verbose()

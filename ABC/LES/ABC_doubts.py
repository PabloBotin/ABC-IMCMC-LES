
# This is a classic ABC destinated to calibrate LES parameters.  
# In order to test it, I will use 
# reference data obtained from a spectralLES run using trivial parameters.

# Import modules 
from mpi4py import MPI  # must always be imported first

import h5py
from .spectralles import Config, SpectralLES
comm = MPI.COMM_WORLD
import numpy as np
import itertools
import os

#### Step 1. Generate reference data D  ####
# For test purposes, I will generate the ref data with a spectralLES run
# However, it will be computed with a DNS for paper purposes
# 64^3. spectralLES was initially set at 32, running it at 64^3 is too heavy, so I will stick to 32 for test purposes
# Ref parameters taken from 4 GEV version of mixed P & sigma training: 
C0=-0.032
C1=-0.014
C2=-0.2
C3=-0.2
# Data consists on the 3D velocities and must be obtained from a snapshot after statistic
# steady state is reached. We consider steady state from 0.8s, from plot observation.
# Also, the cycle _limit has be set in 5 (instead of 5000) just to verify if it runs. 
# However, we can't leave at 5000 cycles because ut takes so long, so I will have to find ...
# the minimum number of cycles which allows running it for 0.8s. Does this make sense? 
# From run.py...
def main_abc_program():

	# Start by running a Dynamic Smagorinsky case with random initial condition
    # in order to get the "baseline" LES comparison and a better solution
    # field from which to restart all of the ABC runs.
	config = Config(pid='dyn_smag', model='dyn_smag', test_filter='gaussian',
                    tlimit=4.0, cycle_limit=5)

    sim = SpectralLES(config)  # get new LES instance
    sim.run_quiet()  # ignore the results

    # Now, the real run:
    config = Config(pid='abc_run1', model='4term',
                    C0=-0.032, C1=-0.014, C2=-0.2, C3=-0.2,
                    # C0=0.01, C1=0.01, C2=0.01, C3=0.01,
                    init_cond='file', init_file='dyn_smag.checkpoint.h5',
                    tlimit=0.8, cycle_limit=5)

    sim = SpectralLES(config)  # get new LES instance
    results = sim.run_verbose()

    # Process the results into an ABC distance metric
    # At this moment, the results are the summary statistics of Pi.
    # I have to modify spectralLES in order to output the 3D velocities to 'fh' 
    fh = h5py.File(results) 

    # Load the 3D snapshot into memory, it takes the spectral velocities (u_hat)
    # from a h5py file and Fourier transforms it into u (physical velocity) 
    # so we have in memory the 3D velocities, from there we can compute statistics 
    sim.initialize_from_file(fh)

    # Compute the summary statistics from 3D velocities
    # I need to modify update_4term_viscosity() in order to compute:
    #   log of pdf of Pi
    #   log of pdf of sigmas (x3) 
    sim.update_4term_viscosity()    
    # Right now, does it take the velocities stored in 'u' when running initialize_from_file() 
    # or should I related them some how? 

    # Once calculated the summary statistics, I can save them into a h5py file, 
    # Which we ll use later for computing the distances
    # For that, create a h5py file with the keys: Pi, Sigma11, Sigma12 & Sigma13
    # And store the relative values in a h5py file. How can I do this? 
    return log_pdf_Pi, log_pdf_sigma11, log_pdf_Sigma12, log_pdf_Sigma13 

# Run the defined method and store summary statistics into a h5py file
ref = h5py.File(main_abc_program())



#### Step 2 & 3. Define prior P(c) #### 
# It must be a uniform distribution of ci with width Ci. Bounds are Ci-Ci/2 and Ci+Ci/2 
C_limits = (((C0-C0/2),(C0+C0/2)),((C1-C1/2),(C1+C1/2)),((C2-C2/2),(C2+C2/2)),((C3-C3/2),(C3+C3/2)))

# 7 intervals in each dimension. 7^4=2401 sets of Ci.
N=7

# Definition of the sampling method .
def sample_uniform_grid(N, C_limits):
    # Creates an array of N evenly spaced values for each parameter inside of their defined range. 
    # Returns; N_params arrays of N values arrays (N_params * N matrix)
    N_params = len(C_limits)    # Returns the num of params, corresponding to the num of ranges. 
    C = np.empty([N_params, N]) # Initializes array of arrays 
    for i in range(N_params):   # Creates an array of N evenly spaced values for each parameter 
        C[i, :] = np.linspace(C_limits[i][0], C_limits[i][1], N)
    permutation = itertools.product(*C) 
    C_matrix = list(map(list, permutation)) # Creates N^N_params samples with all possible combination of param values
    return C_matrix

# Running the method 
Ci_matrix = sample_uniform_grid(N, C_limits) # Array of N sampled parameter sets



#### Step 4. Run LES for all the parameter sets #### 
# Instead of LES< we'll use the RANS ODE for test purposes 
# The loop must follow this logic:
for i in range (len(Ci_matrix)):
    main_abc_program(C_matrix[i][0], C_matrix[i][1], C_matrix[i][2], C_matrix[i][3])
# INTRODUCE LOOP OF RANS RUNS HERE !!!

# We must output the final snapshot of 3D velocities to a new h5py file. 
# Should we do that or just directly compute the sigmas and Pis and directly store them in a file? 



# If we dont compute then... We'll have to:
#### Step 5. Calculate D' from F(ci) LES model  ####
# Postprocessing loop; Compute sigmas (x3) and Pi for all snapshots and store them into a h5py file with similar
# structure to the ref data one. 



#### Step 6. Compute distances ####




#### Step 7. Calculate statistical distance d(S',S) ####
def distance (S_LES, S_DNS):
    d= np.sqrt(np.absolute(np.log(S_DNS)-np.log(S_LES))**2)
# I must loop it for all different runs of LES 



#### Step 8. Accept / Reject Sample ####
# If d < e. We accept the set of parameters by storing it in the array of accepted samples 
# If d > e. Reject (Just not store it).
# How can I store it? Create another h5py file with the keys: c0, c1, c2,c3


#### Step 9. Approximate posterior joint pdf ####
# Using the list of accepted sets of parameters, we build a pdf.
# the three- or four-dimensional posterior pdfs, respectively, are calculated using kernel density estimation (KDE)
# with a Gaussian kernel and bandwidth defined by Scott’s Rule. Check pyabc -> kde.py & test_kde.py



#### Post Processing ####
# Select the parameters to be used. 
# Which technique should we use for this? MAP! It is the mode of the posterior distibution (value which appears the most)
# Run LES with those parameters (MAP sample)
# Compare it with classic Smagorinsky and LES runs using other parameter calibration techniques to validate
# this calibration method is more accurate. 
# Use my postprocessing doc for that. 




# How to update GitHub:
#   git add <file> (or .)
#   git commit -m 'message'
#   git push origin 

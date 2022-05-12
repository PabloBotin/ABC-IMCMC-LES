from mpi4py import MPI  # must always be imported first

import h5py
from .spectralles import Config, SpectralLES
comm = MPI.COMM_WORLD
import numpy as np
import itertools
import os


#########################################################################################
################################# RUN, BABE, RUN  #######################################
#########################################################################################

# Test sampling_uniform_grid()
# N = 2
# C_limits = ((0.5,5), (0,2), (0,3), (1,2.5))
# print (sample_uniform_grid(N, C_limits))
# print (len(sampling_uniform_grid(N, C_limits)))

############# Algorithm 1. ABC rejection sampling algorithm ###########################


#### Step 1. Calculate reference summary statistics (S) from ref data (D) ####
# A. Ek spectra 
# Load and store the whole spectrum of the ref kinetic energy 
jhu_data = './JHU_DNS.spectra'
x_ref = np.loadtxt(jhu_data, skiprows=2)[:, 0]
Ek_ref = np.loadtxt(jhu_data, skiprows=2)[:, 1]
# However, I should filter this info in accordance to the LES filter (small scales are not computed in LES)

# B & C
# Files:    
#           - Sum_stats_true (4, 100) = values of the log pdfs (3 first rows correspond to sigma, 4th row to P).
#           - Sum_stats_bins (2,100) = values of bins / x coordinates, (1st row -> sigma / 2nd row -> P).
# Load DNS arrays 
folder = '/Users/pablo/Documents/ABCIMCMC/Algorithm1'
y_dns = np.loadtxt(os.path.join(folder, 'sum_stat_true'))
x_dns = np.loadtxt(os.path.join(folder, 'sum_stat_bins'))

# B. Log of pdf of deviatoric stress tensor (Sigma) 
dns_log_pdf_sigma_11 = y_dns[0]
dns_log_pdf_sigma_12 = y_dns[1]
dns_log_pdf_sigma_13 = y_dns[2]
x_sigma = x_dns [0]

# C. Log of pdf of production rate (P) 
dns_log_pdf_P = y_dns[3]
dns_x_P = x_dns[1]




#### Step 3. Sample N parameters Ci from prior distribution ####
# We don't know prior distribution, so we use a uniform distribution: sample of all possible
# combinations of N uniformly assigned values for each parameter.
# N and C_limits is inspired on Olga's RANS forward run paper. 

# First, we define the sampling method .
def sample_uniform_grid(N, C_limits):
    # Creates an array of N evenly spaced values for each parameter inside of their defined range. 
    # In our case, N=60 & C_limits = ([0.5,5], [0,2], [0,3], [1,2.5])
    # Returns; N_params arrays of N values arrays (N_params * N matrix)

    N_params = len(C_limits)    # Returns the num of params, corresponding to the num of ranges. 
    C = np.empty([N_params, N]) # Initializes array of arrays 
    for i in range(N_params):   # Creates an array of N evenly spaced values for each parameter 
        C[i, :] = np.linspace(C_limits[i][0], C_limits[i][1], N)
    permutation = itertools.product(*C) 
    C_matrix = list(map(list, permutation)) # Creates N^N_params samples with all possible combination of param values
    return C_matrix

# C_limits is taken from Olga's dissertation - Chapter 4.5.1.
# Ranges depend of whether we are training in production or in sigma. 
N = 60
C_limits = ((-0.3,0), (-0.5,0.5), (-0.2,0.2), (-0.2,0.2))  # Training in sigma 
# C_limits = ((-0.3,0), (-0.5,0.5), (-0.5,0.2), (-0.2,0.5))  # Training in production 
# C_limits = ((-0.3,0), (-0.5,0.5), (-0.5,0.2), (-0.5,0.5))  # Training on sigma & P combined 
C_matrix = sample_uniform_grid(N, C_limits) # Array of N sampled parameter sets
# print (len(C_matrix))

#### Step 5. Calculate D' from F(ci) LES model  ####
# A. In order to test the algorithm, I will use a RANS system of ODE's for homogeneous turbulence. 
# Because LES takes around 2 min to run each simulation meanwhile the RANS ODE takes less than 1 sec. 
# So.. which ODE should I use? Colin asked Olga to provide hers. 
# Also, which is the output of RANS? Similar statistics? 

# B. spectralLES
# Colin provided a new spectralLES code which is way simpler. The problem is that the output statistic is only 
# the energy spectrum. I asked for sigmas and P so let's see if he computes them, otherwise I ll compute em myself.
# The implementation of the LES run will be set taking the code 'run.py' as a reference, modifying the method 
# In order to adjust the parameters for each run. 
def main_abc_program():

    # Start by running a Dynamic Smagorinsky case with random initial condition
    # in order to get the "baseline" LES comparison and a better solution
    # field from which to restart all of the ABC runs.
    config = Config(pid='dyn_smag', model='dyn_smag', test_filter='gaussian',
                    tlimit=4.0, dt_stat=1.0, dt_init=1)

    sim = SpectralLES(config)  # get new LES instance
    sim.run_quiet()  # ignore the results

    # Run a GEV test case for debugging.
    # NOTE to Pablo: Replace this part with your ABC algorithm.
    config = Config(pid='abc_run1', model='4term',
                    C0=-0.069, C1=0.07, C2=0.0056, C3=0,
                    init_cond='file', init_file='dyn_smag.checkpoint.h5',
                    tlimit=4.0, dt_stat=1.0)
    sim = SpectralLES(config)  # get new LES instance
    results = sim.run_verbose()

    # Process the results into an ABC distance metric
    fh = h5py.File(results)

    Ek = np.zeros(sim.num_wavemodes)
    Pi = np.zeros(fh['000/Pi/hist'].shape)

    for step in fh:  # this loops over group keys not the groups themselves!
        Ek += fh[f'{step}/Ek']
        Pi += fh[f'{step}/Pi/hist']

        # you can also access...
        # fh[f'{step}/Pi/edges'] -> histogram bin edges
        # fh[f'{step}/Pi/moments'] -> 1st 4 raw moments
        # fh[f'{step}/Pi/range'] -> 1st and last bin edge
        # fh[f'{step}/Pi/log'] -> whether the histogram was computed using the
        #                         log10 of the data because data was strictly
        #                         positive. (If so, edges will be for log10 of
        #                         data, but range and moments are always for
        #                         the raw data).

        # I can also just remove the log(data) functionality.

    # ...

    return Summary statistics (sigma & Production rate)
    
# Run the LES model with all the sampled parameter sets. 
for i in range (len(C_matrix)):
    main_abc_program(C_matrix[i][0], C_matrix[i][1], C_matrix[i][2], C_matrix[i][3])

# Once I have implemented the summary statistic computation inside of the LES run.. I have to compute the distance.  


#### Step 6. Calculate model summary statistic S' from D' ####
# This would be included in Step 5 if Colin does it. He is working on that. 
# Even if he does not, I would need to include it by myself when building the LES run.
# But, let's go back to the origins... Which summary statistic should I use? 
### A. Energy Spectrum 
### B. pdf of Sigma 
### C. Production rate 
### D. Combination of sigma and P. See Olga's dissertation -> Chapter 4.5.1.3
#  Olga uses B, C, a combination of both (how?), and compares them.
# Let's see how is the output provided by Colin. 
# From that, I'll have to figure out how to compute spectra, log pdf of sigma and log pdf of P. 
# This can be based on ABC_static_test.py and my postprocessing code.
# Elaborar c'odigo que lo calcule desde los h5py files. 
# Para ello, runnear un static test del nuevo spectralLES 

#### Step 7. Calculate statistical distance d(S',S) ####
# A. Summatory of the normalized difference of each wavenumber (only those corresponding to the LES resolved scales).
# B. MSE of log of pdf of S' and S.
# C. Kullbacl-Lieber (KL) divergence of S' and S.
# Ask Peter which one should I use. 
# Meanwhile, prepare the methods for computing it in the 3 proposed ways. 
# B. Mean Square Error (MSE). Production's distance is always computed with this technique.
# B. Applied to Sigma 
# Is the log necessary? I mean, it is part of the MSE or is it assuming that S is the pdf straight? 
# MSE should be divided between number of observations (N), but we do not compute it here since we compute 
# the results 1 by 1. 
def d_MSE_sigma (S_LES, S_DNS):
    d = 0
    for i in range (3):     # For Sigma11, Sigma12 and Sigma13
        d =+ (S_LES - S_DNS)**2
    return d


# B. Applied to Production 
def d_MSE_P (S_LES, S_DNS):
    d = (S_LES - S_DNS)**2

# C. Kullbalc-Lieber (KL) divergence 
# I do not understand what is Sf, I ll take it as the ref summary statistic btm. 
# Also, I would have to compute the inverse of np.log(). Ask Peter.
def d_KL_sigma (S_LES, S_DNS):
    d = 0
    for i in range (3):     # For Sigma11, Sigma12 and Sigma13
        d =+ np.log.inv(S_LES) * np.absolute(S_LES - S_DNS)
    return d

# I need to wait to see how are the results outputted in order to implement the distance calculation

#### Step 8. Accept / Reject Sample ####
# If d < e. We accept the set of parameters by storing it in the array of accepted samples 
# If d > e. Just not store it. 
# How can I store it? I should definetely use a 

#### Step 11. Approximate posterior joint pdf ####
# Using the list of accepted sets of parameters, we build a pdf.
# the three- or four-dimensional posterior pdfs, respectively, are calculated using kernel density estimation (KDE)
# with a Gaussian kernel and bandwidth defined by Scott’s Rule. Check pyabc -> kde.py & test_kde.py



#### Post Processing ####
# Select the parameters to be used. Which technique should we use for this? MAP
# Run LES with those parameters (MAP sample)
# Compare it with classic Smagorinsky and LES runs using other parameter calibration techniques to validate it is more similar to DNS.
# Use my postprocessing doc for that. 






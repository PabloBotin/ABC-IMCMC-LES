
# Version A. dist computed from spectra (Ek) and Peter's formula 

import numpy as np
import h5py
from scipy.stats import gaussian_kde

# Shut up warnings
# import warnings
# warnings.filterwarnings("ignore")

# Define the LES and ABC parameters
# -----------------------------------------------------------------------------
N_dealiased = 32            # What are these commands for? 
N_samples_per_param = 3     # Match with the run

# Olga's coefficients based on stress alone
C1, C2, C3, C4 = -0.043, 0.018, 0.036, 0.036    # Ref. Why manually set? 

C_limits = [(0.5*C, 1.5*C) for C in (C1, C2, C3, C4)] # Set ranges of possible values for each coef.  

C = np.empty([4, N_samples_per_param])      # Generate Nsamples_per_param values for each coef. 
for i in range(4):
    C[i] = np.linspace(C_limits[i][0], C_limits[i][1], N_samples_per_param) 

C_grids = np.meshgrid(*C, indexing='ij')    # Generate all possible combination of values 
N_runs = C_grids[0].size  # should be equal to N_samples**4. Number of coef sets 

data_dir = '/Users/zeketopanga/Documents/ABC-IMCMC-LES/ABC/Frontera_output/ABC_LES_test/data'
success = np.load(f'{data_dir}/success_array.npy') # Returns an array with the number of N_runs 
# print ('succes: ', success)
# specifying TRUE (succesful) or FALSE (unsuccesful) for each file. Numpy ndarray.   

# Postprocess distance
# -------------------------------------------------------------------------
filename = f'{data_dir}/baseline.statistics.h5'     # filename of ref statistics ? 
fh = h5py.File(filename)    # load baseline statistics 

# there are only outputs '000' and '001' corresponding to the random
# initial condition and the final solution at t_sim = tlimit
logEk_base = np.log(fh['001/Ek'][:-1])  # Log of baseline energy spectrum 
logEk_base[26]=0.000000001
# print ('logEk_base', logEk_base)
# base_pdfs = []  
# for i, S in enumerate(['Pi', 'sigma_11', 'sigma_12', 'sigma_13']):  # i=0,1,2,3 S= values in Pi, Sigma..
#                                     # So it goes through all datasets one by one. 
#     hist = fh[f'001/{S}/hist'][:]       # It creates a pdf for each Pi, S11, S12, S13
#     edges = fh[f'001/{S}/edges'][:]
#     x = 0.5 * (edges[:-1] + edges[1:])
#     density = gaussian_kde(x, weights=hist)
#     # print ('density: ', density(x))
#     base_pdfs.append((x, np.log(density(x)))) # It adds the pdf of Ek.
#     # print ('np.log(density(x):  ', np.log(density(x)))
#     # base_pdfs contais all pdf from Pi, S11, S12, S13 and Ek. 

fh.close()

dist = np.zeros(N_runs)
for r in range(N_runs): # Computes the distance for all runs
    if success[r]:                                 # Error, dice que el 81 está 'out of bounds' 
        filename = f'{data_dir}/abc_run_{r}.statistics.h5' 
        fh = h5py.File(filename)    # Loads the datafile corresponding to the run. 

        Ek = fh['001/Ek'][:-1]      # Loads energy spectrum. 
        Ek[26]=0.0000000001               # Produces matrix of -infs. 
        # print (Ek.shape)
        # print ('np.log(Ek)', np.log(Ek))
        # compute L2 norm of Ek
        # Computes distance of Ek between current run&ref
        dist[r] = np.sum(np.sqrt((np.log(Ek) - logEk_base)**2))  # Computes distance of Ek between current run&ref
            # Verify if it matches Peter's provided distance. 
        # print ('dist:', dist)
        # Creates the pdf's of Pi, S11, S12, S13 and computes the distance with ref's pdf (also computed here)
        # for i, S in enumerate(['Pi', 'sigma_11', 'sigma_12', 'sigma_13']):
        #     hist = fh[f'001/{S}/hist'][:]
        #     edges = fh[f'001/{S}/edges'][:]
        #     x = 0.5 * (edges[:-1] + edges[1:])
        #     density = gaussian_kde(x, weights=hist)
        #     x_b, y_b = base_pdfs[i]
        #     # print ('desity2:', density(x_b))
        #     # print ('base_pdfs', base_pdfs[i])
        #     y = np.log(density(x_b))  # evaluate PDF at baseline x values
        #     # print (y)
        #     # add L2 norm of each PDF to L2 norm of Ek for total distance
        #     dist[r] += np.sum((y - y_b)**2)

        fh.close()

    else:
        # run not successful, set distance to infinity
        dist[r] = np.inf
# print ('Ek:',Ek)
# print ('np.log(Ek)', np.log(Ek))
# print ('dist:', dist)
# Accept X% of all runs
# -------------------------------------------------------------------------
r= 0.30     # Acceptance rate 
N_accept = int(r * N_runs)
R_sort = np.argsort(dist)   # this is an array of indices that would sort dist
epsilon = dist[R_sort[N_accept-1]]  # dist <= epsilon for all accepted runs

# copy accepted C values from uniform grid
C_accept = np.empty((4, N_accept))
for p in range(4):
    C_accept[p] = C_grids[p].flat[R_sort[:N_accept]]
# print ('C_accept:', C_accept)         # For validation purposes    
# Get Kernel Density Estimation (KDE) for accepted data
# -------------------------------------------------------------------------
density = gaussian_kde(C_accept)
x = np.vstack([C.ravel() for C in C_grids])
jpdf = np.reshape(density(x), C_grids[0].shape)

# Get the Maximum A Posteriori (MAP) set of coefficients from the KDE
# -------------------------------------------------------------------------
map0, map1, map2, map3 = np.where(jpdf == jpdf.max())
# np.where always outputs arrays, even for single values or empty results
map0 = map0[0]
map1 = map1[0]
map2 = map2[0]
map3 = map3[0]

c_map = np.empty(4)
c_map[0] = C[0, map0]
c_map[1] = C[1, map1]
c_map[2] = C[2, map2]
c_map[3] = C[3, map3]

print(f'Baseline Coefficients: C1=-0.043, C2=0.018, C3=0.036,  C4=0.036')

print(f'MAP Coefficients: C1={c_map[0]:0.3f}, C2={c_map[1]:0.3f},  '
      f'C3={c_map[2]:0.3f},  C4={c_map[3]:0.3f}')

# print ('dist:', dist)
# print ('jpdf:', jpdf)
np.savez(f'{data_dir}/abc_posterior.npz', dist=dist, jpdf=jpdf)

# you can reload this file with:
# fh = np.load(f'{data_dir}/abc_posterior.npz')
# dist = fh['dist']
# jpdf = fh['jpdf']

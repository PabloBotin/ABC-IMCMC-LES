import numpy as np
import h5py
from scipy.stats import gaussian_kde

# Define the LES and ABC parameters
# -----------------------------------------------------------------------------
N_dealiased = 32
N_samples_per_param = 7

# Olga's coefficients based on stress alone
C1, C2, C3, C4 = -0.043, 0.018, 0.036, 0.036

C_limits = [(0.5*C, 1.5*C) for C in (C1, C2, C3, C4)]

C = np.empty([4, N_samples_per_param])
for i in range(4):
    C[i] = np.linspace(C_limits[i][0], C_limits[i][1], N_samples_per_param)

C_grids = np.meshgrid(*C, indexing='ij')
N_runs = C_grids[0].size  # should be equal to N_samples**4

data_dir = 'Wherever you downloaded data on your personal computer'
success = np.load(f'{data_dir}/success_array.npy')

# Postprocess distance
# -------------------------------------------------------------------------
filename = f'{data_dir}/baseline.statistics.h5'
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
        filename = f'{data_dir}/abc_run_{r}.statistics.h5'
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

# Accept X% of all runs
# -------------------------------------------------------------------------
N_accept = int(0.05 * N_runs)
R_sort = np.argsort(dist)   # this is an array of indices that would sort dist
epsilon = dist[R_sort[N_accept-1]]  # dist <= epsilon for all accepted runs

# copy accepted C values from uniform grid
C_accept = np.empty((4, N_accept))
for p in range(4):
    C_accept[p] = C_grids[p].flat[R_sort[:N_accept]]

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

print(f'MAP Coefficients: C1={c_map[0]:0.3f}, C2={c_map[1]:0.3f},  '
      f'C3={c_map[2]:0.3f},  C4={c_map[3]:0.3f}')

np.savez(f'{data_dir}/abc_posterior.npz', dist=dist, jpdf=jpdf)

# you can reload this file with:
# fh = np.load(f'{data_dir}/abc_posterior.npz')
# dist = fh['dist']
# jpdf = fh['jpdf']

"""
This is a test of ABC using a system of ordinary differential equations (ODEs)
rather than LES, so that it is much faster and can be run on a laptop.

Also, Olga performed these same tests and published results here:
https://arc.aiaa.org/doi/pdf/10.2514/1.J060308


Numba is a "Just-in-Time" compiler that converts Python math functions into
fast machine code (as fast as optimized C). The `@njit` decorator is a
shorthand way of asking Numba to JIT compile the defined function the first
time it is called in the code (which is inside `solve_ivp`). Using Numba is
good practice when a simple function like `rans_ode` will be called thousands
of times in rapid succession (the `solve_ivp` time-integration loop)

"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


# Step 0. Define the system of equations
# -----------------------------------------------------------------------------
@njit(nogil=True, cache=True)
def rans_ode(t, y, Ca1, Ca2, Ce1, Ce2, Smax, f):
    """The Hamlington and Dahm Non-Equilibrium RANS model.

    """
    k = y[0]
    eps = y[1]
    a = y[2:]
    P = 0.0
    dy_dt = np.empty_like(y)
    S = 0.5 * Smax * np.sin(Smax * f * t)  # S_01 = S_10 = S[1] if 6-vector
    P = -2 * k * a[1] * S

    dy_dt[0] = P - eps
    dy_dt[1] = (eps/k) * (Ce1*P - Ce2*eps)
    dy_dt[2:] = -(P/eps - 1 + Ca1) * (eps/k) * a
    dy_dt[3] += (Ca2 - 4/3) * S  # only "S[1]" is non-zero

    return dy_dt


t_span = [0, 50/3.3]
t_eval = np.linspace(0, 50/3.3, 100)
y0 = [1., 1., 0., 0., 0., 0., 0., 0.]

# Step 1. Generate reference data D
# -----------------------------------------------------------------------------
# For verification purposes, generate the ref data with baseline coefficients:
# args: Ca1, Ca2, Ce1,  Ce2,  Smax, f
args = (1.5, 0.8, 1.44, 1.83, 3.3, 0.5)
sol = solve_ivp(rans_ode, t_span, y0, t_eval=t_eval, args=args)
k_ref = sol.y[0]

# Step 2 Define a uniform prior with uniform sampling
# -----------------------------------------------------------------------------
# Coefficient bounds imposed by Olga in her paper
C_limits = ((0.5, 5), (0, 2), (0, 3), (1, 2.5))
N_params = len(C_limits)
N = 15
C = np.empty([N_params, N])
for i in range(N_params):
    C[i] = np.linspace(C_limits[i][0], C_limits[i][1], N)

C_grids = np.meshgrid(*C, indexing='ij')

# Step 3. Loop over sample set of coefficients
# -----------------------------------------------------------------------------
N_runs = C_grids[0].size  # should be equal to N^N_params = 60**4
dist = np.empty(N_runs)

tstart = time.time()

for r in range(N_runs):
    Cs = np.empty(N_params)
    for p in range(N_params):
        Cs[p] = C_grids[p].flat[r]

    args = (*Cs, 3.3, 0.5)

    # Step 3a. Run RANS model for sampled set of coefficients
    # ------------------------------------------------------------------------
    sol = solve_ivp(rans_ode, t_span, y0, t_eval=t_eval, args=args)

    # Step 3b. Compute the distance metric for the run
    # ------------------------------------------------------------------------
    if sol.success is True:
        dist[r] = np.sum((sol.y[0] - k_ref)**2)**0.5  # L2 norm of k
    else:
        dist[r] = np.inf

tend = time.time()
print(f'Loop took {tend-tstart:0.2f} s')

# Step 4. Since its a uniform gridded sampling, can just find global minimum
# distance as an optimal estimator for coefficients
# -----------------------------------------------------------------------------
r_min = np.argmin(dist)
c_min = np.empty(N_params)
for p in range(N_params):
    c_min[p] = C_grids[p].flat[r_min]

# c_min is a single set of coefficients
print(f'Optimal coefficients based on minimum distance: Ca1={c_min[0]:0.2f}, '
      f'Ca2={c_min[1]:0.2f},  Ce1={c_min[2]:0.2f},  Ce2={c_min[3]:0.2f}')

# Step 5. Accept 0.05% of all runs
# -----------------------------------------------------------------------------
N_accept = int(0.0005 * N_runs)  # 0.05% of N_runs
R_sort = np.argsort(dist)   # this is an array of indices that would sort dist
epsilon = dist[R_sort[N_accept-1]]  # dist <= epsilon for all accepted runs

# copy accepted C values from uniform grid
C_accept = np.empty((N_params, N_accept))
for p in range(N_params):
    C_accept[p] = C_grids[p].flat[R_sort[:N_accept]]

# Step 6. Perform Kernel Density Estimation (KDE) on accepted data
# -----------------------------------------------------------------------------

# Step 7. Get the Maximum A Posteriori (MAP) set of coefficients from the KDE
# -----------------------------------------------------------------------------

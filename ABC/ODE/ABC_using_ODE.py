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
from scipy.integrate import odeint
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
# sol = solve_ivp(rans_ode, t_span, y0, t_eval=t_eval, args=args) # solve_ivp() will be replaced for a faster one
sol = odeint(rans_ode, y0, t_eval, args=args, tfirst=True)
# k_ref = sol.y[0] # Use with solve_ivp()
k_ref = sol[0]  # Use with odeint()
print (k_ref)

# Step 2 Define a uniform prior with uniform sampling
# -----------------------------------------------------------------------------
# Coefficient bounds imposed by Olga in her paper
C_limits = ((0.5, 5), (0, 2), (0, 3), (1, 2.5))
N_params = len(C_limits)
N = 5
C = np.empty([N_params, N])
for i in range(N_params):
    C[i] = np.linspace(C_limits[i][0], C_limits[i][1], N)

C_grids = np.meshgrid(*C, indexing='ij') 
# Creates a matrix of [i,j] where i= N_params (lines) and j= N^4 (columns)

# Step 3. Loop over sample set of coefficients
# -----------------------------------------------------------------------------
N_runs = C_grids[0].size  # C_grids es el array de la primera fila.
dist = np.empty(N_runs) # Initialization of distances array. 

tstart = time.time()

for r in range(N_runs):
    Cs = np.empty(N_params) # initialization of coefficients array. This array will change in each run.
    for p in range(N_params):
        Cs[p] = C_grids[p].flat[r] # .flat[r] returns the r value of the 1D version array.
        # It loops over all coefs in all sets, creating a set of coefficeints for each run  

    args = (*Cs, 3.3, 0.5)  # What is this? Input for the solver (args: Ca1, Ca2, Ce1,  Ce2,  Smax, f)

    # Step 3a. Run RANS model for sampled set of coefficients
    # ------------------------------------------------------------------------
    # sol = solve_ivp(rans_ode, t_span, y0, t_eval=t_eval, args=args) 
    sol = odeint(rans_ode, y0, t_eval, args=args, tfirst=True) # tfirst=True because it comes from f(t,y)
    # what about t_eval? How should it be set that way? What about t_span? Not needed to set cuz its in t_eval, no?  
    # Try to replace solve_ivp() by ODEINT function from scipy.integrate and see how the 
    # loop time changes

    # Step 3b. Compute the distance metric for the run
    # ------------------------------------------------------------------------
    # With solve_ivp()
    # if sol.success is True:
    #     dist[r] = np.sum((sol.y[0] - k_ref)**2)**0.5  # L2 norm of k. Distance of only the first coef? 
    #     # No, distance of just one estimator. We don´t compute the d of the coefs but the d of results. 
    #     # We take y[0] as the result.
    # else:
    #     dist[r] = np.inf    # We reject coefficients by assigning a big distance. 


    # # With odeint()
    # if sol.size == t_eval.size: # Not correct 
    #     dist[r] = np.sum((sol[0] - k_ref)**2)**0.5  # L2 norm of k. Distance of only the first coef? 
    #     # No, distance of just one estimator. We don´t compute the d of the coefs but the d of results. 
    #     # We take y[0] as the result.
    # else:
    #     dist[r] = np.inf    # We reject coefficients by assigning a big distance. 

    dist[r] = np.sum((sol[0] - k_ref)**2)**0.5

print (dist)
tend = time.time()
print(f'Loop took {tend-tstart:0.2f} s')    # :0.2f is the format type: float with 2 decimals. 
# This measures the total time taken by the loop (solve and compute distance of all sets of coefficients)

# Step 4. Since it is a uniform gridded sampling, can just find global minimum
# distance as an optimal ¿estimator? for coefficients.
# -----------------------------------------------------------------------------
r_min = np.argmin(dist) # Returns the index of the set of coefficients (num of run) with the min distance. 
c_min = np.empty(N_params) # Initialization of array of best coefficients (min disance)
for p in range(N_params):
    c_min[p] = C_grids[p].flat[r_min] # Fills c_min array with the coefficients of the min distance run.  

# c_min is a single set of coefficients
print(f'Optimal coefficients based on minimum distance: Ca1={c_min[0]:0.2f}, '
      f'Ca2={c_min[1]:0.2f},  Ce1={c_min[2]:0.2f},  Ce2={c_min[3]:0.2f}')

# Step 5. Accept 0.05% of all runs
# -----------------------------------------------------------------------------
N_accept = int(0.0005 * N_runs)  # 0.05% sets of N_runs must be accepted 
R_sort = np.argsort(dist)   # this is an array of indices that would sort dist
epsilon = dist[R_sort[N_accept-1]]  # dist <= epsilon for all accepted runs.
# Epsilon is the maximum distance to be accepted 

# Copy accepted C values from uniform grid
C_accept = np.empty((N_params, N_accept))
for p in range(N_params):
    C_accept[p] = C_grids[p].flat[R_sort[:N_accept]] 
    # Fills the matrix of accepted sets by order of distance magnitude. 


# Step 6. Perform Kernel Density Estimation (KDE) on accepted data
# Colin is going to do this. 


# Step 7. Get the Maximum A Posteriori (MAP) set of coefficients from the KDE
# Colin is going to do this.

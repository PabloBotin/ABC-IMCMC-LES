"""
This is a test of ABC-IMCMC using a system of ordinary differential equations (ODEs)
rather than LES, so that it is much faster and can be run on a laptop.

Numba is a "Just-in-Time" compiler that converts Python math functions into
fast machine code (as fast as optimized C). The `@njit` decorator is a
shorthand way of asking Numba to JIT compile the defined function the first
time it is called in the code (which is inside `solve_ivp`). Using Numba is
good practice when a simple function like `rans_ode` will be called thousands
of times in rapid succession (the `solve_ivp` time-integration loop)

"""
import time
import numpy as np
import statistics 
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
sol = solve_ivp(rans_ode, t_span, y0, t_eval=t_eval, args=args) # solve_ivp() will be replaced for a faster one
# sol = odeint(rans_ode, y0, t_eval, args=args, tfirst=True)
k_ref = sol.y[0] # Use with solve_ivp()
# k_ref = sol[0]  # Use with odeint()

# Step 2. Calibration Step (Nc, r) 
# In the LES version, I'll have to perform several calibration steps as described in Chapter 4.C.
# 	Step 2.A. Sample Nc parameters ci from prior distribution π(c)
C_limits = ((0.5, 5), (0, 2), (0, 3), (1, 2.5))
N_params = len(C_limits)
Nc = 5	# It will be smaller than 60 for sure, but this is just a random small value for test purposes
C = np.empty([N_params, Nc])
for i in range(N_params):
    C[i] = np.linspace(C_limits[i][0], C_limits[i][1], Nc)
C_grids = np.meshgrid(*C, indexing='ij') 
# Creates a matrix of [i,j] where i= N_params (lines) and j= N^4 (columns)

# 	Step 2.B. Solve the model with all sampled sets of coefficients
N_runs = C_grids[0].size  # C_grids es el array de la primera fila.
dist = np.empty(N_runs) # Initialization of distances array. 

tstart = time.time()

for r in range(N_runs):
    Cs = np.empty(N_params) # initialization of coefficients array. This array will change in each run.
    for p in range(N_params):
        Cs[p] = C_grids[p].flat[r] # .flat[r] returns the r value of the 1D version array.
        # It loops over all coefs in all sets, creating a set of coefficeints for each run  

    args = (*Cs, 3.3, 0.5)  # What is this? Input for the solver (args: Ca1, Ca2, Ce1,  Ce2,  Smax, f)
	sol = solve_ivp(rans_ode, t_span, y0, t_eval=t_eval, args=args) 

# 	Step 2.C. Compute array of distances. 
    if sol.success is True:
    	dist[r] = np.sum((sol.y[0] - k_ref)**2)**0.5 
    else:
    	dist[r] = np.inf   

tend = time.time()
print(f'Loop took {tend-tstart:0.2f} s') 

# 	Step 2.D. Estimate the distribution P(d) from array of distances. 
# This must be done by using Gaussian kernel density estimation (KDE). Correct? 
# In that case... Do not lose time with it now and wait for Colin´s new code. 


# 	Step 2.E. Define tolerance ϵ such that P(d≤ϵ)=r.
r = 0.0005	# This is the acceptance rate, in this case it is set at 0.05%. 
N_accept = int(r * N_runs)  # 0.05% sets of N_runs must be accepted 
R_sort = np.argsort(dist)   # this is an array of indices that would sort dist
epsilon = dist[R_sort[N_accept-1]]  # dist <= epsilon for all accepted runs.
# Is it ok like this, without using P itself? I think so. 





# 	Step 2.F. Randomly choose c0 from ci parameters with d≤ϵ.
# What does this mean? Why should I choose that randomly? 
# Btm, I´ll choose the one with the shortest distance. 
# Copy accepted C values from uniform grid
C_accept = np.empty((N_params, N_accept))
for p in range(N_params):
    C_accept[p] = C_grids[p].flat[R_sort[:N_accept]] 
# Select the first one (min distance)
c0= C_accept[0]


# 	Step 2.G. Adjust prior based on variance of parameters with d≤ϵ.
# Should I calculate the variance of parameters here? Then how do I relate that? Check Olga´s paper. 
# Method for computing variance of parameters
# It computes the variance of a dataset, but we have one dataset for each of the parameters so.. 
# how do we compute the total variance? Or should we compute one for each? 
statistics.pvariance(C_accept)	# Population variance (n)
statistics.variance(C_accept)	# Sample variance (n-1)  
# Which one of this 2 should I use? 


# 	Step 2.H. Estimate covariance C0 from parameters with d≤ϵ.
statistics.covariance(x, y) 
# Covariance is performed between 2 datasets, but we have multiple accepted sets of parameters so...
# How should I proceed in here? 


# 	Step 2.I. Return c0, C0 & ϵ.
# Should I create a class for the calibration, or even a function? 
# Yeah, create a function after knowing it works. 

# Step 3. MCMC (c0, C0, e, m, Na)
# c0; accepted parameters from calibration step. 
# C0; covariance calculated in calibration step.
# Na; is it the number of chains? how should I define this? 
Na=10

i=0
while i<Na:
	# Step 3A. Sample c' from proposal. This time, it is not uniform but P(c), right? 
	# Is this a set of sets of coefficients, like in ABC? OR just one set? 
	C= ? # If set of sets, create a loop before this. 
	args = (*C, 3.3, 0.5)

	# Step 3B. Calculate D'=F(c') from model. 
	# If it is one set, ok. If its a set of sets, should I run all of them like in ABC? 
	sol = solve_ivp(rans_ode, t_span, y0, t_eval=t_eval, args=args

	# Step 3C. Calculate the model summary statistic 
	S = sol.y[0]
	S_ref = k_ref

	# Step 3D. Calculate statistical distance 
	if sol.success is True:
		dist[r] = np.sum((S - S_ref)**2)**0.5 
	else:
		dist[r] = np.inf  

	# Step 3E. If dist<e, accept with probability, increment i, set ci = c' and update covariance (Ci)
	# Ask Peter to explain me the concept of h. 

# Step 4. Return list of accepted ci and estimate posterior joint pdf. 







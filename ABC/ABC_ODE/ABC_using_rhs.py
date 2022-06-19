"""
This is a test of ABC using a system of ordinary differential equations (ODEs)
rather than LES, so that it is much faster and can be run on a laptop.

Also, Olga performed these same tests and published results here:
https://arc.aiaa.org/doi/pdf/10.2514/1.J060308


Numba is a "Just-in-Time" compiler that converts Python math functions into
fast machine code (as fast as optimized C). The `@njit` decorator is a
shorthand way of asking Numba to JIT compile the defined function the first
time it is called in the code (which is inside `solve_ivp`). Using Numba is
good practice when a simple function like `rans_rhs` will be called thousands
of times in rapid succession (the `solve_ivp` time-integration loop)

"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from numba import jit, njit

from matplotlib import pyplot as plt
from matplotlib import colors


# Step 0. Define the system of equations
# -----------------------------------------------------------------------------
@njit(nogil=True, cache=True)
def rans_rhs(t, y, Ca1, Ca2, Ce1, Ce2, Smax, f):
    """The Hamlington and Dahm Non-Equilibrium RANS model.

    For this test, only S_01 = S_10 and a_01 = a_10 = y[2] are non-zero tensor
    entries.
    """
    k = y[0]
    eps = y[1]
    aij = y[2]
    P = 0.0
    dy_dt = np.empty_like(y)
    Sij = 0.5 * Smax * np.sin(Smax * f * t)
    P = -2 * k * aij * Sij

    dy_dt[0] = P - eps
    dy_dt[1] = (eps/k) * (Ce1*P - Ce2*eps)
    dy_dt[2] = (Ca2 - 4/3)*Sij - (P/eps - 1 + Ca1)*(eps/k)*aij

    return dy_dt


@njit(nogil=True, cache=True)
def norm(x):
    return np.mean(x**2)**0.5


@njit(nogil=True, cache=True)
def RK45(fun, t_span, y0, args=[], rtol=1e-4, atol=1e-7):
    """The Dormand-Prince RK45 method used by ode45 in Matlab and solve_ivp in
    Scipy

    """
    y0 = np.asarray(y0)  # tells numba this is an ndarray

    n_stages = 6
    C = np.array([1/5, 3/10, 4/5, 8/9, 1], np.float64)
    A = np.array([
                 [1/5, 0, 0, 0, 0],
                 [3/40, 9/40, 0, 0, 0],
                 [44/45, -56/15, 32/9, 0, 0],
                 [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
                 [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
                 ], np.float64)
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
                 np.float64)
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40], np.float64)

    y = np.empty_like(y0)
    y_new = np.empty_like(y0)
    K = np.empty((n_stages+1, y0.size), dtype=y0.dtype)

    t = t_span[0]
    tlimit = t_span[1]

    nt = 500  # the default case only requires 47 outputs, so this is plenty
    t_save = np.empty(nt, np.float64)             # pre-allocate memory
    y_save = np.empty((nt, y0.size), np.float64)  # pre-allocate memory

    t_save[0] = t
    y_save[0] = y0

    y[:] = y0
    f0 = fun(t, y0, *args)

    # --------------------------------------------------------------
    # Estimate initial dt
    # --------------------------------------------------------------
    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    f1 = fun(t + h0, y0 + h0 * f0, *args)
    d2 = norm((f1 - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2))**0.2

    dt = min(100.0 * h0, h1)
    dt_min = 10**(np.round(np.log10(dt)) - 5)

    # --------------------------------------------------------------
    # Time integration loop
    # --------------------------------------------------------------
    steps = 0
    i = 1
    rejected = False
    while t < tlimit:
        dt = min(tlimit - t, dt)

        K[0] = f0
        for s in range(1, n_stages):
            dy = np.dot(K[:s].T, A[s-1, :s]) * dt
            K[s] = fun(t + C[s-1] * dt, y + dy, *args)

        y_new[:] = y + np.dot(K[:-1].T, B) * dt
        K[-1] = fun(t + dt, y_new, *args)

        scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
        err = norm((np.dot(K.T, E) * dt) / scale)

        if err < 1.0:
            factor = min(10.0, 0.9 * err**-0.2)

            if rejected:
                factor = min(1.0, factor)
            rejected = False

            t += dt
            y[:] = y_new
            f0[:] = K[-1]
            dt *= factor
            t_save[i] = t
            y_save[i] = y
            i += 1

        else:
            dt *= max(0.2, 0.9 * err**-0.2)
            rejected = True

        steps += 1
        if dt < dt_min or i == nt:
            break

    if t < tlimit:
        out = None, None, i
    else:
        out = t_save[:i], y_save[:i], steps  # return just what was saved

    return out


# Step 1. Generate reference data D
# -----------------------------------------------------------------------------
# For verification purposes, generate the ref data with baseline coefficients:
t_span = (0.0, 50/3.3)
y0 = (1., 1., 0.)
args = (1.5, 0.8, 1.44, 1.83, 3.3, 0.5)  # = Ca1, Ca2, Ce1, Ce2, Smax, f

# sol = solve_ivp(rans_rhs, t_span, y0, args=args, rtol=1e-4, atol=1e-7)
# k_ref = sol.y[0]

t, y, steps = RK45(rans_rhs, t_span, y0, args=args)
k_ref = y[:, 0]

t_interp = np.linspace(0.0, 50/3.3, 101)
k_ref = interp1d(t, k_ref, kind='cubic')(t_interp)

print(f'Correct solution: Ca1={1.5:0.2f}, '
      f'Ca2={0.8:0.2f},  Ce1={1.44:0.2f},  Ce2={1.83:0.2f}')

# Step 2 Define a uniform prior with uniform sampling
# -----------------------------------------------------------------------------
# Coefficient bounds imposed by Olga in her paper
C_limits = ((0.5, 5), (0, 2), (0, 3), (1, 2.5))
N_params = len(C_limits)
N = 60
C = np.empty([N_params, N])
for i in range(N_params):
    C[i] = np.linspace(C_limits[i][0], C_limits[i][1], N)

C_grids = np.meshgrid(*C, indexing='ij')

# Step 3. Run RANS model for all sets of coefficients
# -----------------------------------------------------------------------------
N_runs = C_grids[0].size  # should be equal to N^N_params = 60**4
dist = np.empty(N_runs)
Cs = np.empty(N_params)

tstart = time.time()

for r in range(N_runs):
    for p in range(N_params):
        Cs[p] = C_grids[p].flat[r]

    try:
        t, y, steps = RK45(rans_rhs, t_span, y0, args=(*Cs, 3.3, 0.5))
    except Exception as e:
        print(f'Warning, run failed due to a {type(e)} exception')
        t = None

    if t is not None:
        k = interp1d(t, y[:, 0], kind='cubic')(t_interp)
        dist[r] = np.sum((k - k_ref)**2)**0.5  # L2 norm of k

    else:
        dist[r] = np.inf

tend = time.time()
print(f'Loop took {tend-tstart:0.2f} s')

# Since its a uniform gridded sampling, can use minimum distance as an
# estimator for the optimal coefficients
r_min = np.argmin(dist)
c_min = np.empty(N_params)
for p in range(N_params):
    c_min[p] = C_grids[p].flat[r_min]

print(f'Coefficients via minimum distance: Ca1={c_min[0]:0.2f}, '
      f'Ca2={c_min[1]:0.2f},  Ce1={c_min[2]:0.2f},  Ce2={c_min[3]:0.2f}')

# Step 4. Accept X% of all runs
# -----------------------------------------------------------------------------
N_accept = int(0.005 * N_runs)  # X% of N_runs
R_sort = np.argsort(dist)   # this is an array of indices that would sort dist
epsilon = dist[R_sort[N_accept-1]]  # dist <= epsilon for all accepted runs

# copy accepted C values from uniform grid
C_accept = np.empty((N_params, N_accept))
for p in range(N_params):
    C_accept[p] = C_grids[p].flat[R_sort[:N_accept]]


# Step 5. Get Kernel Density Estimation (KDE) for accepted data
# -----------------------------------------------------------------------------
kde = gaussian_kde(C_accept)
positions = np.vstack([C.ravel() for C in C_grids])
hist = np.reshape(kde(positions), C_grids[0].shape)

# Step 6. Get the Maximum A Posteriori (MAP) set of coefficients from the KDE
# -----------------------------------------------------------------------------
map0, map1, map2, map3 = np.where(hist == hist.max())
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

print(f'Coefficients via MAP of the KDE: Ca1={c_map[0]:0.2f}, '
      f'Ca2={c_map[1]:0.2f},  Ce1={c_map[2]:0.2f},  Ce2={c_map[3]:0.2f}')

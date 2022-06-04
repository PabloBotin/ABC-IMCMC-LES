


############### Step 1. Generate reference data D #############################
# -----------------------------------------------------------------------------
# This task must be separately performed in advance. 
# Def ref coefs; Trained on sigma and production, 4 parameters.
C0=-0.0318
C1=-0.0144
C2=-0.198
C3=-0.198


# Step 2 Define a uniform prior with uniform sampling
# -----------------------------------------------------------------------------
C_limits = ((C0-C0/2, C0+C0/2), (C1-C1/2, C1+C1/2), (C2-C2/2, C2+C2/2), (C3-C3/2, C3+C3/2))
N_params = len(C_limits)
N = 7	# Proposed by Peter 
C = np.empty([N_params, N])
for i in range(N_params):
    C[i] = np.linspace(C_limits[i][0], C_limits[i][1], N)
C_grids = np.meshgrid(*C, indexing='ij')


# Step 3. Run spectralLES for all sets of coefficients
# -----------------------------------------------------------------------------
N_runs = C_grids[0].size 
dist = np.empty(N_runs)
Cs = np.empty(N_params)

tstart = time.time()

for r in range(N_runs):
    for p in range(N_params):
        Cs[p] = C_grids[p].flat[r]

    try:
        # RUN spectralLES here 
    except Exception as e: # If it does not run properly...
        print(f'Warning, run failed due to a {type(e)} exception')
        t = None

    if t is not None:
        # Compute Summary statistics and distance here or...
        # Save 3D velocities outut to file PETER's plan. For that, wait untill the next version of spectralLES. 

    else:
        dist[r] = np.inf
        # If the spectralLES run is not succesful, we consider that set of coefs as a bad result, assigning a high distance. 
        # Why? What if the good set of coefficients does not actually compute? 

tend = time.time()
print(f'Loop took {tend-tstart:0.2f} s')


# If we indeed compute the distances, we can select the optimal set of coefficients by 
# selecting the one with the smallest distance. 

r_min = np.argmin(dist)
c_min = np.empty(N_params)
for p in range(N_params):
    c_min[p] = C_grids[p].flat[r_min]

print(f'Coefficients via minimum distance: Ca1={c_min[0]:0.2f}, '
      f'Ca2={c_min[1]:0.2f},  Ce1={c_min[2]:0.2f},  Ce2={c_min[3]:0.2f}')










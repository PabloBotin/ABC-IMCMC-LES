import time
import numpy as np
from scipy.integrate import odeint
from spectralles import Config, SpectralLES 


############### Step 1. Generate reference data D #############################
# -----------------------------------------------------------------------------
# For verification purposes, generate the ref data with baseline coefficients.
# Start by running a Dynamic Smagorinsky case with random initial condition
    # in order to get the "baseline" LES comparison and a better solution
    # field from which to restart all of the ABC runs.
config = Config(pid='dyn_smag', model='dyn_smag', test_filter='gaussian',
                tlimit=0.8, dt_stat=1.0)

sim = SpectralLES(config)  # get new LES instance
sim.run_quiet()  # ignore the results

# Run a GEV test case for generating ref data. 
# Def ref coefs; Trained on sigma and production, 4 parameters.
C0=-0.0318
C1=-0.0144
C2=-0.198
C3=-0.198
# Print ref coefs.
print(f'Ref solution: C1={C0}, C2={C1},  C3={C2},  C4={C3}')

# Set up configuration of ref LES run. 
config = Config(pid='abc_run1', model='4term',
                C0=C0, C1=C1, C2=C2, C3=C3,		
                init_cond='file', init_file='dyn_smag.checkpoint.h5', 
                tlimit=4.0, dt_stat=1.0)

ref = SpectralLES(config)  # get new LES instance
results = ref.run_verbose()

# Process the results into an ABC distance metric
fh = h5py.File(results)			# Does this create or open the h5py file?


# This run is set at 64^3 and is taking ages.. 
# I can't imagin how much will it take for the forward run. 

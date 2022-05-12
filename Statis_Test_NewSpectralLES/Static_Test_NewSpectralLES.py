### Static Test of the New spectralLES ###

#Import modules 
from mpi4py import MPI  # must always be imported first

import numpy as np
import h5py

import spectralles

comm = MPI.COMM_WORLD


# Set of Parameters. From case A (3 param; training in Sigma)
C = np.array([-0.069, 0.07, 0.0056, 0])


def main_abc_program():

    # Start by running a Dynamic Smagorinsky case with random initial condition
    # in order to get the "baseline" LES comparison and a better solution
    # field from which to restart all of the ABC runs.
    config = spectralles.Config(pid='dyn_smag', model='dyn_smag', test_filter='gaussian',
                    tlimit=8.0, dt_stat=1.0)

    sim = spectralles.SpectralLES(config)  # get new LES instance
    sim.run_verbose()  # ignore the results

    # Run a GEV test case for debugging.
    # NOTE to Pablo: Replace this part with your ABC algorithm.
    config = spectralles.Config(pid='abc_run1', model='gev',
                    C0=-0.069, C1=0.07, C2=0.0056, C3=0,
                    init_cond='file', init_file='dyn_smag.checkpoint.h5',
                    tlimit=8.0, dt_stat=1.0)
    sim = spectralles.SpectralLES(config)  # get new LES instance
    results = sim.run_verbose()

    # Process the results into an ABC distance metric
    fh = h5py.File(results)
    Ek = np.zeros(sim.num_wavemodes)
    for grp in fh:
        Ek += grp['Ek']

    # ...

    return


###############################################################################
# if __name__ == "__main__":
main_abc_program()











from mpi4py import MPI  # must always be imported before the first numpy import

import numpy as np
import h5py

from spectralles import Config, SpectralLES  # only works when run.py and spectralles.py are in the same directory!


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

    return


###############################################################################
if __name__ == "__main__":
    main_abc_program()

import nuympy as np

##############	CALIBRATION STEP  #############
# 1. SAMPLE Nc parameters Ci from prior distribution pi(c)
# Which sampling method should I use? 

def sampling_random(N_total, C_limits):
    """ Random sampling with uniform distribution.
    :return: list of lists of sampled parameters
    """
    N_params = len(C_limits)
    C_array = np.random.random(size=(N_total, N_params))
    for i in range(N_params):
        C_array[:, i] = C_array[:, i] * (C_limits[i, 1] - C_limits[i, 0]) + C_limits[i, 0]
    C_array = C_array.tolist()
    return C_array



### Run Babe Run ###
# What should the C_limits be? An array of arrays? 
# C_array is a list of lists of sampled parameters 
N_total	= 10	# Ntotal= Number of samples (How do I select this? ) 
C_limits = ('c1', 'c2', 'c3', 'c4')			# BIen? 

samples = sampling_random(N_total, C_limits)	 

print (samples)



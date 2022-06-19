import numpy as np
import h5py
import math

# Ek_ref
filename = 'abc_run1.statistics.h5'
f = h5py.File(filename, 'r') 
print(list(f['001/S_diag'].keys()))









# shape = f['001/Ek'].shape[0]	# returns teh number of elements of the dataset in int value. Wavenumbers, cells? 
# Ek_ref = f['001/Ek'][:shape]	# <HDF5 dataset "Ek": shape (28,), type "<f8">
# print ('Ek_ref:', Ek_ref)

























# # Ek
# filename = 'abc_run2.statistics.h5'
# f = h5py.File(filename, 'r') 
# Ek = f['001/Ek'][:shape]	# <HDF5 dataset "Ek": shape (28,), type "<f8">
# # # print ('Ek:', Ek)

# # Compute distance 
# print (type(Ek[5]))
# print (type(Ek[5]))

# d_log=np.empty(shape)
# for i in range(shape):
# 	d_log[i]= (np.absolute((np.log(Ek_ref[i])-np.log(Ek[i]))))**2
# 	if math.isnan(d_log[i]):
# 		d_log[i]=0

# d= np.sqrt(np.mean(d_log))
# print (d)



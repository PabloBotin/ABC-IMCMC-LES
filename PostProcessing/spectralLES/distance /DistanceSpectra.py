# This file will try to compute the distance between the spectra of 2 different LES runs.
import numpy as np
import h5py
import matplotlib.pyplot as plt

# 2. Load data from h5py files
class LES_S_spectra:
	# This class automatically computes S from spectra given the filename. 

	def __init__(self, filename):	# Automatically loads spectra

		self.filename= filename

		# Load file 
		f = h5py.File(self.filename, 'r') # Readonly

		# Load data 
		shape = f['001/Ek'].shape[0]	# returns teh number of elements of the dataset in int value. Wavenumbers, cells? 
		Ek = f['001/Ek'][:shape]	# <HDF5 dataset "Ek": shape (28,), type "<f8"> 



def Ek_from_h5py (filename):

	f = h5py.File(self.filename, 'r') # Readonly 
	shape = f['001/Ek'].shape[0]	# returns teh number of elements of the dataset in int value. Wavenumbers, cells? 
	Ek = f['001/Ek'][:shape]	# <HDF5 dataset "Ek": shape (28,), type "<f8"> 
	return Ek

def dist_Ek (Ek_ref, Ek):

	d_log=np.empty(shape)
	for i in range(shape):
		d_log[i]= (np.absolute((np.log(Ek_ref[i])-np.log(Ek[i]))))**2
		if math.isnan(d_log[i]):	# In case I obtain nan values(from 0s), ignore them. Is this correct? 
			d_log[i]=0
	d= np.sqrt(np.mean(d_log))

	return d




# # Compute distance (use ref statistic). I have to somehow introduce it in the code from external source! 
# d= np.sqrt(np.mean(np.absolute(np.log10(S_ref)-np.log10(self.S_LES))**2)) 
# return d


# ### Run baby run ###
filename1 = 'abc_run1.statistics.h5'
filename2 = 'abc_run2.statistics.h5'
d= dist_Ek (Ek_from_h5py (filename1), Ek_from_h5py (filename2))
print(d)


# LES1=LES_spectra (filename1)
# LES2=LES_simulation (filename2)

# distance=LES1.




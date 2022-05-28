# This file will try to compute the distance between the summary statistkics of 2 different LES runs.
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Definition of some methods to be used later on...
TINY_log = np.log(10e-8)

def take_safe_log(x):
    log_fill = np.empty_like(x)         # Why this? 
    log_fill.fill(TINY_log)
    TINY= np.empty_like(x)
    TINY[:] = 10e-8               
    log = np.log(x, out=log_fill, where=x > TINY)
    return log

# 1. Run 2 LES (from a different code). By the moment, they compute Pi, si that will be our summary statistic.
# 	Done, files are 'abc_run1.statistics.h5' and 'abc_run2.statistics.h5'

# 2. Load data from h5py files and compute their logpdf's. 

class LES_simulation:
	def __init__(self, filename, color):
		self.filename= filename
		self.color= color

	def logpdf(self):
		f = h5py.File(self.filename, 'r') # Readonly
		# print(list(f.keys())) # --> ['000', '001'] This are datasets
		# We are interested in '001', which is the final snapchot, right? 
		dset_001 = f['001']
		# print(list(dset_001.keys())) # --> ['Ek', 'Pi', 'S_diag', 'Ssq', 'nuT']
		Pi = dset_001['Pi']	
		# print(list(Pi.keys())) # -->['edges', 'hist'] # Edges are the x s 
		edges= Pi['edges']
		edges= edges[...]
		x = (edges[1:] + edges[:-1]) / 2
		hist = Pi['hist']
		# Compute log of pdf 
		# Normalize 
		norm = np.divide(np.sum(hist),len(edges))
		hist_pdf= np.divide (hist,norm)
		# Log
		hist_log_pdf= take_safe_log(hist_pdf)	# It might need 2 arrays
		plt.plot(x, hist_log_pdf, ls='-', c= 'b', lw= 2)
		plt.ylim([-5,5])


filename1 = 'abc_run1.statistics.h5'
filename2 = 'abc_run2.statistics.h5'
LES1=LES_simulation (filename1, 'b')
LES2=LES_simulation (filename2, 'r')
plt.show()
Pi1=LES1.logpdf()
Pi2=LES1.logpdf()

# 4. Compute distance.

def distance (S_LES, S_DNS):
	d= np.sqrt(np.absolute(np.log(S_DNS)-np.log(S_LES))**2)

d=distance (Pi1, Pi2)
print(d)



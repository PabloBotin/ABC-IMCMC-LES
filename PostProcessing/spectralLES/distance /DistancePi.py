# This file will try to compute the distance between the S (Pi and Sigmas) of 2 different LES runs.
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math

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
####################### LES CLASS  #############################
# The idea is to automatically log all data from the statistics file and from there, have the ability 
# to compute summary statistics
class LES_simulation:
	def __init__(self, filename):
		
		self.filename= filename
		self.f = h5py.File(self.filename, 'r') # Readonly

	def logpdf_Pi(self):

		# Load histogram
		edges= self.f['001/Pi/edges']
		edges= edges[...]
		x = (edges[1:] + edges[:-1]) / 2
		hist = self.f['001/Pi/hist']

		# Compute log of pdf 
		norm = np.divide(np.sum(hist),len(edges))
		hist_pdf= np.divide (hist,norm)
		hist_log_pdf= take_safe_log(hist_pdf)	

		# Store it in a dictionary (for easy access)
		values = dict()
		values ['hist']= hist_log_pdf
		values ['x']= x

		return values


####################### Plot Pi method ##############################
	def plot_Pi (self, hist, x, color, coefs):

		plt.style.use('fivethirtyeight')
		plt.title('log pdf of LES and DNS production rate ')
		plt.plot(x, hist, label= coefs, ls='-', c= color, lw= 2) 
		plt.ylim([-5,5])
		plt.legend()


###################### COMPUTE DISTANCE #############################
def dist_S (S_ref, S):

	d= np.square(np.subtract(S_ref,S)).mean()
	return d


# I discharged this because I can't perform log of negative values. 
	# shape = S_ref.shape[0]	
	# d_log=np.empty(shape)
	# for i in range(shape):
	# 	d_log[i]= (np.absolute((np.log(S_ref[i])-np.log(S[i]))))**2 #invalid value encountered in log
	# 	if math.isnan(d_log[i]):	# In case I obtain nan values(from 0s), ignore them. Is this correct? 
	# 		d_log[i]=0
	# d= np.sqrt(np.mean(d_log))
	# return d



# Para que esto funcione con los histogramas, tengo que cerciorarme de que los bins (x) son los mismos. 
# Como se generan los edges? Check spectralLES

#########################  RUN BABY RUN  #############################
##############  LOAD DATA ############################################
filename1 = 'abc_run1.statistics.h5'
filename2 = 'abc_run2.statistics.h5'
LES1=LES_simulation (filename1)
LES2=LES_simulation (filename2)

##############  COMPUTE LOGPDF  ######################################
Pi1=LES1.logpdf_Pi()
Pi2=LES2.logpdf_Pi()

##############  Plot logpdfPi  #######################################
# Overpose as many as you want like this:
# plot_Pi(Pi1['hist'], Pi1['x'], 'b', '[0.01, 0, 0, 0]')
# plot_Pi(Pi2['hist'], Pi2['x'], 'r', '[0.37, 9.7, 33.4, 2.4]')

# plt.show()

#################  COMPUTE DISTANCE  #################################
print ('hist1', Pi1['hist'])
print ('hist2', Pi2['hist'])
print ('Distance between Pi summary statistics =', dist_S(Pi1['hist'], Pi2['hist']))

# 4. Compute distance.

# def distance (S_LES, S_DNS):
# 	d= np.sqrt(np.absolute(np.log(S_DNS)-np.log(S_LES))**2)

# d=distance (Pi1, Pi2)
# print(d)



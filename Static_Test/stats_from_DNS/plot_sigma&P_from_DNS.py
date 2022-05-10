## This code takes the log of pdf of sigma and P from DNS datafiles and plot them. 
# Files:	
#			- Sum_stats_true (4, 100) = values of the log pdfs (3 first rows correspond to sigma, 4th row to P).
#			- Sum_stats_bins (2,100) = values of bins / x coordinates, (1st row -> sigma / 2nd row -> P).

import numpy as np
import os
import matplotlib.pyplot as plt

y_true = np.loadtxt(os.path.join('/Users/pablo/Documents/Pablo_Stats/stats_from_DNS', 'sum_stat_true'))
x_true = np.loadtxt(os.path.join('/Users/pablo/Documents/Pablo_Stats/stats_from_DNS', 'sum_stat_bins'))


# Compute log of pdf sigma and plot 
def plot_logpdf_sigma():
	titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
	fig, axarr = plt.subplots(nrows=1, ncols=3, sharey=True)
	for i in range (3):
		axarr[i].set_xlabel(titles[i], labelpad=1)
		# plt.plot(x_true[0], y_true[i])
		plt.ylim([-5,2])
		# plt.title("log pdf of sigma_11")
		axarr[i].plot(x_true[0], y_true[i])
		
	plt.show()


# Compute log of pdf sigma and plot 
def plot_logpdf_production():
	plt.plot(x_true[1], y_true[3])
	plt.ylim([-5,4])
	plt.title("log pdf of production rate")	
	plt.show()


########### RUN BABE RUN #########
# plot_logpdf_sigma()
plot_logpdf_production()






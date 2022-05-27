# This file will try to compute the distance between the summary statistkics of 2 different LES runs.
import numpy as np
import h5py
import matplotlib.pyplot as plt
# 1. Run 2 LES (from a different code). By the moment, they compute Pi, si that will be our summary statistic.
# 	Check

# 2. Load data from both files. 
# I should modify this function in order to extract the Pis 
f = h5py.File('abc_run1.statistics.h5', 'r')
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
# Plot pdf (normalized histogram)
# plt.hist(hist, edges, density=11) 
# plt.show()

# Compute log of pdf 
# Normalize 
norm = np.divide(np.sum(hist),len(edges))
hist_pdf= np.divide (hist,norm)
# Log
TINY_log = np.log(10e-8)

def take_safe_log(x):
    log_fill = np.empty_like(x)         # Why this? 
    log_fill.fill(TINY_log)
    TINY= np.empty_like(x)
    TINY[:] = 10e-8               
    log = np.log(x, out=log_fill, where=x > TINY)
    return log

hist_log_pdf= take_safe_log(hist_pdf)	# It might need 2 arrays
plt.plot(x, hist_log_pdf, ls='-', c= 'b', lw= 2)
plt.ylim([-5,5])
plt.show()
# print (edges)
# print (hist_log_pdf)

# print (hist[...])
# print (edges[...])
# print(Pi.shape)



# 3. Compute the log of the pdf of their summary statistics (S_LES & S_DNS).

# def normalize_hist(hist, bins, range):	# range= [-15,5]
#     # the values of the edges with hist? 
#     x = (edges[1:] + edges[:-1]) / 2
#     norm = np.divide(np.sum(hist),bins)
#     # norm = np.sum(pdf)/bins
#     return np.divide (hist,norm)
#     # return pdf/norm

# # def pdf_from_array_with_x(array, bins, range):
# #     pdf, edges = np.histogram(array, bins=bins, range=range, density=1)
# #     # How does it asociate the values of the edges with hist? 
# #     x = (edges[1:] + edges[:-1]) / 2
# #     return x, pdf

# TINY_log = np.log(10e-8)

# def take_safe_log(x):
#     log_fill = np.empty_like(x)         # Why this? 
#     log_fill.fill(TINY_log)
#     TINY= np.empty_like(x)
#     TINY[:] = 10e-8               
#     log = np.log(x, out=log_fill, where=x > TINY)
#     return log

# # Plot log pdf of production rate
# def LES_logpdf_production():
#     production_pdf= normalize_hist(hist, 100, [-15, 5])
#     production_logpdf = take_safe_log(hist)
#     # print (production_logpdf.shape)
#     plt.style.use('fivethirtyeight')
#     plt.title('log pdf of LES and DNS production rate ')
#     # plt.plot(x_dns[1], y_dns[3], label='DNS', c= 'k', lw= 1.5)
#     # plt.plot( x_dns[1], p_Smag_logpdf, label='Smagorinsky', c= 'g', lw= 1.5)
#     plt.plot(production_logpdf, ls='--', c= 'r', lw= 2) 
#     plt.ylim([-5,5])
#     plt.legend()
#     plt.show()
#     # plt.hist(self.production_data, bins='auto', range=[-1.2, 0.01], density=1)

# LES_logpdf_production()




# 4. Compute distance.
# def distance (S_LES, S_DNS):
#     d= np.sqrt(np.absolute(np.log(S_DNS)-np.log(S_LES))**2)
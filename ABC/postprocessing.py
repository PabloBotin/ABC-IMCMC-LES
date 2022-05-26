# The objective of this code is postprocessing the output of spetralLES 
# The output file is abc_run1.statistics.h5
# 1. Understand the type of file. Learn how to use it. 
# 2. Understand the data within the file. And plot it. 


import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import the file for reading 

fh = h5py.File('abc_run1.statistics.h5', 'r')
# print (list (fh.keys()))
#print (fh['001'])
# print ('Pi shape:', fh['000/Pi/hist'].shape)
# print ('X shape:', fh['000/Pi/edges'].shape)

# Pi = np.zeros(fh['000/Pi/hist'].shape)
# for step in fh:
# 	Pi += fh[f'{step}/Pi/hist']
# 	# x += fh[f'{step}/Pi/edges']
# # print ('x:', x)
# # print ('Pi:', Pi)
# plt.plot(Pi)	
# plt.show()


Pi = np.zeros(fh['000/Pi/hist'].shape)
for step in fh:
	Pi += fh[f'{step}/Pi/hist']

TINY_log = np.log(10e-8)

def take_safe_log(x):
    log_fill = np.empty_like(x)         # Why this? 
    log_fill.fill(TINY_log)
    TINY= np.empty_like(x)
    TINY[:] = 10e-8               
    log = np.log(x, out=log_fill, where=x > TINY)
    return log

def pdf_from_array(array, bins, range):
    pdf, edges = np.histogram(array.flatten(), bins=bins, range=range)
    x = (edges[1:] + edges[:-1]) / 2
    norm = np.divide(np.sum(pdf),bins)
    # norm = np.sum(pdf)/bins
    return np.divide (pdf,norm)
    # return pdf/norm

def LES_logpdf_production():
    production_pdf= pdf_from_array(Pi, 100, [-5, 5])
    production_logpdf = take_safe_log(production_pdf)
    plt.style.use('fivethirtyeight')
    plt.title('log pdf of LES production rate ')
    plt.plot(production_logpdf, ls='--', c= 'r', lw= 2) 
    plt.ylim([-20,5])
    plt.show()
    # plt.hist(self.production_data, bins='auto', range=[-1.2, 0.01], density=1)

# Run 
if __name__ == "__main__":
	LES_logpdf_production()

# Ek = np.zeros(sim.num_wavemodes)
# x= np.zeros(fh['000/Pi/edges'].shape)
# for step in fh:  
#     Ek += fh[f'{step}/Ek']
#     x += fh[f'{step}/Pi/edges']
# plt.plot(Ek, x)	
# plt.show()


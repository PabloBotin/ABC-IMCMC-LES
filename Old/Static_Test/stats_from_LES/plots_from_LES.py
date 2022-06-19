import matplotlib as mpl
import gc
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import os
from importlib import reload
import sys

# import pyabc.utils as utils
# import pyabc.glob_var as g
# import data

# Set Matplotlib parameters
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.labelsize'] = mpl.rcParams['font.size']
mpl.rcParams['axes.titlesize'] = 1.5 * mpl.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = mpl.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = mpl.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = mpl.rcParams['font.size']
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['axes.linewidth'] = 1
mpl.rc('text', usetex=True)

plt = reload(plt)

# Selection of the folder where the files are at
folder = '/Users/pablo/Documents/Pablo /stats_from_LES/datafiles '

# Definition of methods used later for plotting. 
# Calculation of pdf from array
# class LES_Output:


# def pdf_from_array_with_x(array, bins, range):
#     pdf, edges = np.histogram(array, bins=bins, range=range, density=1)
#     x = (edges[1:] + edges[:-1]) / 2
#     return x, pdf

# Reshape data from binary .rst datafile to array of floats 
def rst_to_array(file):
    array= np.reshape(np.fromfile(file, dtype=np.float64), 262144)
    return array

# Plot arrays in one or several plots at the same time and save it into a .pdf file
def imagesc(Arrays, titles, name=None):
    axis = [0, 2*np.pi, 0, 2*np.pi]
    cmap = plt.cm.jet  # define the colormap
    # norm = mpl.colors.Normalize(vmin=np.min(Arrays), vmax=np.max(Arrays))
    norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)  # for sigma
    # norm = mpl.colors.Normalize(vmin=-4, vmax=4)    # for velocity

    if len(Arrays) > 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), sharey=True, figsize=(12, 4))
        k = 0
        for ax in axes.flat:
            im = ax.imshow(Arrays[k].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest", extent=axis)
            ax.set_title(titles[k])
            ax.axis('equal')
            ax.set_xlabel(r'$x$')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            k += 1
        axes[0].set_ylabel(r'$y$')
        cbar_ax = fig.add_axes([0.95, 0.18, 0.017, 0.68])  # ([0.85, 0.15, 0.05, 0.68])
        fig.subplots_adjust(left=0.04, right=0.92, wspace=0.1, bottom=0.12, top=0.94)
        fig.colorbar(im, cax=cbar_ax, ax=axes.ravel().tolist())
    else:
        norm = mpl.colors.Normalize(vmin=-5, vmax=5)    # for velocity
        fig = plt.figure(figsize=(6.5, 5))
        ax = plt.gca()
        im = ax.imshow(Arrays[0].T, origin='lower', cmap=cmap,  norm=norm, interpolation="nearest", extent=axis)
        plt.colorbar(im, fraction=0.05, pad=0.04)
    if name:
        # pickle.dump(ax, open(self.folder + name, 'wb'))
        fig.savefig(name)
    del ax, im, fig, cmap
    gc.collect()

# """Takes natural logarithm and put g.TINY number where x = 0"""
TINY_log = -8
TINY = 10e-8
def take_safe_log(x):
    log_fill = np.empty_like(x)         # Why this? 
    log_fill.fill(TINY_log)                  # from Olga's code
    log = np.log(x, out=log_fill, where=x > TINY)
    return log

# Definition of methods to compute and plot.

# Compute all 9 stress tensors (tau_ij) from the files test-tauij_001.rst
# And plot 3 of them (11, 12, 13) in 3 joint plots 
# i.e. plot_production()
def plot_tau_ij():
    # end = str(end)
    data_shape = (128, 128, 128)
    tau_ij = dict()
    for i in range(3):
        for j in range(3):
            file = os.path.join(folder, f'test-tau{i+1}{j+1}_001.rst')
            data = np.fromfile(file, dtype=np.float64)
            shape = int(np.round(data.shape[0]**(1/3), 0))
            tau = np.reshape(data, (shape, shape, shape))
            tau_ij[f'{i+1}{j+1}'] = np.swapaxes(tau, 0, 2)  # to put x index in first place
    # print(np.all(tau_ij['11'], np.zeros_like(tau_ij['11'])))
    imagesc([tau_ij['11'][:, int(shape/2),:], tau_ij['12'][:, int(shape/2),:], tau_ij['13'][:,int(shape/2),:]],
            [r'$\tau_{11}$', r'$\tau_{12}$', r'$\tau_{13}$'],
            'tau_1j_001.pdf')


# Compute and plot production rate (P) from tau_ij data files and save it into a .pdf file 
# i.e. 
def plot_production():
    data_shape = (128, 128, 128)
    file = os.path.join(folder, f'test-Production_001.rst')
    data = np.fromfile(file, dtype=np.float64)
    shape = int(np.round(data.shape[0]**(1/3), 0))
    production = np.swapaxes(np.reshape(np.fromfile(file, dtype=np.float64), (shape, shape, shape)), 0, 2)
    production[:,int(shape/2),:]
    imagesc([production[:,int(shape/2),:]],
        [ r'$\tau_{ij}S_{ij}$'],
        'production.pdf') 


# Calculate and plot the production rate pdf from 'test-Production_001.rst' 
# i.e. plot_pdf_production()
# The pdf range is set at [-5, 5] based on Olga's paper
def plot_pdf_production ():
    file = os.path.join(folder, f'test-Production_001.rst')
    prod_rate= rst_to_array(file)
    # prod_rate_pdf = pdf_from_array_with_x(prod_rate, 100, [-5, 5])
    # plt.hist(prod_rate_pdf[1], prod_rate_pdf[0]) 
    plt.hist(prod_rate, bins='auto', range=[-1.2, 0.01], density=1)
    plt.title("pdf of the production rate")
    plt.show()


# Calculate and plot the stress tensor pdf from 'test-tau11_001.rst'
# i.e. plot_pdf_tau()
# Question; I am plotting here a histogram of the tau_, not the tau_pdf, does it make sense? 
def plot_pdf_sigma():
    file = os.path.join(folder, f'test-tau11_001.rst')
    sigma= rst_to_array(file)
    # tau_pdf = pdf_from_array_with_x(tau_, 100, [np.min(tau_), np.max(tau_)])
    plt.hist(sigma,bins='auto', range=[-0.1, 0.1], density=1)
    plt.title("pdf of the deviatoric part of the stress tensor")
    plt.show()

# Calculate the ln of the pdf and plot it. 
# ValueError: supplied range of [-inf, 2.720233363877793] is not finite
def plot_pdf_ln_sigma():
    file = os.path.join(folder, f'test-tau11_001.rst')
    sigma= rst_to_array(file)
    # tau_pdf = pdf_from_array_with_x(tau_, 100, [(np.min(tau_)), (np.max(tau_))])
    sigma_ln= take_safe_log(sigma)
    plt.hist(sigma_ln, bins=200, range=[-7.9, -2], density=1)
    plt.title("pdf of logarithmic of the deviatoric part of the stress tensor")
    plt.show()


# Next one: Do the same plotting 3 or all 9 pdfs. 
# Try to recycle this:
# Calculate and plot the stress tensor (tau_ij) pdf. 
# In this case, I need to calculate 9 different pdfs: tau_11, tau_12, tau_13...
# But first, I'm gonna try to do it with just one of 'em. 
# def plot_pdf_tau(file): 
#     tau_= 
#     for i in range (3):
#         for j in range(3): 
#             tau_00= rst_to_array(file)
#             tau_00_pdf = pdf_from_array_with_x(tau_00, 100, [np.min(tau_00), np.max(tau_00)])
#             fig = plt.figure()
#             ax1 = fig.add_subplot(2, 1, 1)
#             ax2 = fig.add_subplot(2, 1, 2)
#             plt.hist(tau_00,bins='auto')
#     plt.show()


# Calculate and plot pdf of any tensor.
# !!! Not Working due to file path problems !!!
# file: complete pathname of the .rst file.
# stat: name of the statistic.
# i.e. plot_pdf('/Users/pablo/Documents/Pablo /plot_from_ABC_static_test/datafiles/test-tau11_001.rst', 'stress tensor')
# sys.path.append('/Users/pablo/Documents/Pablo /plot_from_ABC_static_test/datafiles ')
# def plot_pdf(file, stat): 
#     import file
#     tensor= rst_to_array(file)
#     tensor_pdf = pdf_from_array_with_x(tensor, 100, [np.min(tensor), np.max(tensor)])
#     plt.hist(tensor,bins='auto')
#     plt.title('pdf of '+stat)                    # i.e. statistic= 'production rate'
#     plt.show()

#def plot_sigma11():



def main():
	# plot_tau_ij()
    # plot_production()   
    # plot_pdf_production() 
    # plot_pdf_sigma()
    plot_pdf_ln_sigma()
    

if __name__ == '__main__':
    main()





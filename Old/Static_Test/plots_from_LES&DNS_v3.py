
import matplotlib as mpl
# import gc
import numpy as np
import matplotlib.pyplot as plt
import os
# from geobipy import Histogram1D

# from importlib import reload
# from matplotlib.lines import Line2D

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

# plt = reload(plt)

# Other parameters . . . 
single_column = 255
oneandhalf_column = 397
double_column = 539
fig_width_pt = double_column  # Get this from LaTeX using "The column width is: \the\columnwidth \\"
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean       # height in inches
fig_size = [fig_width, fig_height]

# Useful functions...
# Reshape data from binary .rst datafile to array of floats 
def rst_to_array(file):
    array= np.reshape(np.fromfile(file, dtype=np.float64), 262144)
    return array

def pdf_from_array_with_x(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, density=1)
    x = (edges[1:] + edges[:-1]) / 2
    return x, pdf

def pdf_from_array(array, bins, range):
    pdf, edges = np.histogram(array.flatten(), bins=bins, range=range)
    x = (edges[1:] + edges[:-1]) / 2
    norm = np.divide(np.sum(pdf),bins)
    # norm = np.sum(pdf)/bins
    return np.divide (pdf,norm)
    # return pdf/norm

# """Takes natural logarithm and put g.TINY number where x = 0"""

TINY_log = np.log(10e-8)

def take_safe_log(x):
    log_fill = np.empty_like(x)         # Why this? 
    log_fill.fill(TINY_log)
    TINY= np.empty_like(x)
    TINY[:] = 10e-8               
    log = np.log(x, out=log_fill, where=x > TINY)
    return log


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

# FROM DNS 
## Plot log of pdf of sigma and P from DNS datafiles. 
# Files:    
#           - Sum_stats_true (4, 100) = values of the log pdfs (3 first rows correspond to sigma, 4th row to P).
#           - Sum_stats_bins (2,100) = values of bins / x coordinates, (1st row -> sigma / 2nd row -> P).

y_true = np.loadtxt(os.path.join('/Users/pablo/Documents/Pablo_Stats', 'sum_stat_true'))
x_true = np.loadtxt(os.path.join('/Users/pablo/Documents/Pablo_Stats', 'sum_stat_bins'))




# FROM ABC_static_test
class plots_from_LES:

    def __init__(self, folder, case):
        self.case= case
        self.folder = folder
        self.production_data = rst_to_array(os.path.join(self.folder, f'test-Production_001.rst'))
        self.file_sigma = [[],[],[]] 
        for i in range(3):
            self.file_sigma[i] = os.path.join(self.folder, 'test-tau1{}_001.rst'.format(i+1))

        self.sigma_data = [[],[],[]] 
        for i in range(3):
            self.sigma_data[i] = rst_to_array(self.file_sigma[i])


    # # Plot deviatoric part of the stress tensors (sigma) in a .pdf file
    def plot_sigma_ij(self):
        data_shape = (128, 128, 128)
        sigma_ij = dict()
        for i in range(3):
            for j in range(3):
                file = os.path.join(self.folder, f'test-tau{i+1}{j+1}_001.rst')
                data = np.fromfile(file, dtype=np.float64)
                shape = int(np.round(data.shape[0]**(1/3), 0))
                sigma = np.reshape(data, (shape, shape, shape))
                sigma_ij[f'{i+1}{j+1}'] = np.swapaxes(sigma, 0, 2)  # to put x index in first place
        imagesc([sigma_ij['11'][:, int(shape/2),:], sigma_ij['12'][:, int(shape/2),:], sigma_ij['13'][:,int(shape/2),:]],
            [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$'],
            'sigma_1j_001.pdf')

# Compute and plot production rate (P) from tau_ij data files and save it into a .pdf file  
    def plot_production(self):
        self.file_production = os.path.join(self.folder, f'test-Production_001.rst')
        data_shape = (128, 128, 128)
        data = np.fromfile(self.file_production, dtype=np.float64)
        shape = int(np.round(data.shape[0]**(1/3), 0))
        production = np.swapaxes(np.reshape(np.fromfile(self.file_production , dtype=np.float64), (shape, shape, shape)), 0, 2)
        production[:,int(shape/2),:]
        imagesc([production[:,int(shape/2),:]],
            [ r'$\tau_{ij}S_{ij}$'],
            'production.pdf') 

# # Plot pdf of production rate
#     def plot_pdf_production(self):
#         plt.hist(self.production_data, bins='auto', range=[-1.2, 0.01], density=1)
#         plt.title("pdf of the production rate")
#         plt.show()

# Plot log pdf of production rate
    def LES_logpdf_production(self):
        production_pdf= pdf_from_array(self.production_data, 100, [-5, 5])
        production_logpdf = take_safe_log(production_pdf)
        plt.style.use('fivethirtyeight')
        plt.title('log pdf of LES and DNS production rate '+ self.case)
        plt.plot(x_true[1], y_true[3], label='DNS', c= 'k', lw= 1.5)
        plt.plot(x_true[1], production_logpdf, label= 'MAP', ls='--', c= 'r', lw= 2)     
        plt.ylim([-5,5])
        plt.show()
        # plt.hist(self.production_data, bins='auto', range=[-1.2, 0.01], density=1)
        

# Plot log of pdf of sigma. 
    def LES_logpdf_sigma(self):

        sigma_pdf=[[],[],[]]
        sigma_logpdf=[[],[],[]]

        for i in range(3):
            sigma_pdf[i]= pdf_from_array(self.sigma_data[i], 100, [-0.3, 0.3])
            sigma_logpdf[i] = take_safe_log(sigma_pdf[i])

        titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
        fig, axarr = plt.subplots(nrows=1, ncols=3, sharey=True)
        for i in range (3):
            axarr[i].set_xlabel(titles[i], labelpad=1)       
            axarr[i].plot(x_true[0], sigma_logpdf[i], label= 'MAP', ls='--', c= 'r', lw= 2)
            axarr[i].plot(x_true[0], y_true[i], label='DNS', c= 'k', lw= 1.5)

        plt.style.use('fivethirtyeight')
        plt.legend()  
        plt.suptitle('log pdf of LES and DNS sigma '+ self.case)   
        plt.ylim([-5,5]) 
        plt.show()


##### RUN FOREST RUN ######
folder1 = '/Users/pablo/Documents/Pablo_Stats/data'
# folder2 = '/Users/pablo/Documents/Pablo_Stats/data_B'
ABC_static_test = plots_from_LES (folder1, 'case A')
# ABC_static_test_B = plots_from_LES (folder2, 'case B')

# Production 
# ABC_static_test.plot_production()

# Sigma 
# ABC_static_test.plot_sigma()

# Log pdf Production
ABC_static_test.LES_logpdf_production()
# ABC_static_test_B.LES_logpdf_production('4 param')


# Log pdf Sigma [11 12 13]
ABC_static_test.LES_logpdf_sigma()
# ABC_static_test_B.LES_logpdf_sigma('4 param')

# folder = '/Users/pablo/Documents/Pablo_Stats/data'
# production_data = rst_to_array(os.path.join(folder, f'test-Production_001.rst'))
# production_pdf= pdf_from_array(production_data, 100, [-5, 5])
# print (np.shape (production_pdf[0]))
# print (np.shape (production_pdf[1]))




# # Plot log of pdf of sigma. 
#     def LES_logpdf_sigma(self, label):

#         sigma_pdf=[[],[],[]]
#         sigma_logpdf=[[],[],[]]

#         for i in range(3):
#             sigma_pdf[i]= pdf_from_array_with_x(self.sigma_data[i], 100, [-0.3, 0.3])
#             sigma_logpdf[i] = take_safe_log(sigma_pdf[i][1])

#         titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$']
#         fig, axarr = plt.subplots(nrows=1, ncols=3, sharey=True)
#         for i in range (3):
#             axarr[i].set_xlabel(titles[i], labelpad=1)       
#             axarr[i].plot(sigma_pdf[i][0], sigma_logpdf[i], label= label, ls='--', c= 'r', lw= 2)
#             axarr[i].plot(sigma_pdf[i][0], y_true[i], label='DNS', c= 'k', lw= 1.5)

#         plt.style.use('fivethirtyeight')
#         plt.legend()  
#         plt.suptitle('log pdf of LES and DNS sigma '+ self.case)   
#         plt.ylim([-5,5]) 
#         plt.show()












# This is intended to be the forward run algorithm of LES parameters calibration with ABC IMCMC

# 1. Calculate reference summary statistic S from D.
#		Which one? Look in RANS paper: The reference, S, and modeled, S0, summary statistics for both the
#		 verification and validation cases are given by the specific values of the turbulence kinetic energy.
#			ki = k(ti), at the times t=ti 
#				Should I use the same summary statistics? ti are several times or just one moment? 
#					so.. it basically consists on calculating the kinetic energy of each simulation at a specific moment no? 
# 		But... the DNS data is just one run so.. the summary statistic will just be the kinetic energy at a specified moment? 
# 			Is that an accurate summary statistic? Really? 
# 		What is spect_1024?

import os
import sys
import numpy as np
from numpy.fft import fftfreq, fftn, ifftn
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

# 1. Based on Olga's spectra_DNS.py

# def shell_average(E3, N_points):
#     """
#     Compute the 1D, shell-averaged, spectrum of the 3D Fourier-space
#     variable E3.

#     Arguments:
#         km   - wavemode of each n-D wavevector
#         E3   - 3-dimensional complex or real Fourier-space scalar
#     """
#     k0 = np.fft.fftfreq(N_points) * N_points
#     k1 = np.fft.fftfreq(N_points) * N_points
#     k2 = np.fft.fftfreq(N_points) * N_points
#     K = np.array(np.meshgrid(k0, k1, k2, indexing='ij'), dtype=int)
#     Ksq = np.sum(np.square(K), axis=0)
#     km = np.floor(np.sqrt(Ksq)).astype(int)

#     nz, nny, nk = E3.shape
#     E1 = np.zeros(nk, dtype=E3.dtype)
#     zeros = np.zeros_like(E3)

#     if km[0, 0, 0] == 0:
#         # this mpi task has the DC mode, only works for 1D domain decomp
#         E1[0] = E3[0, 0, 0]

#     for k in range(1, nk):
#         E1[k] = 1/2*np.sum(np.where(km==k, E3, zeros))

#     return km, E1

# def spectral_density(vel_array, dx, N_points, fname):
#     """
#     Write the 1D power spectral density of var to text file. Method
#     assumes a real input in physical space.
#     """
#     k = 2*np.pi*np.array([fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])])
#     spect3d = 0
#     for array in vel_array:
#         fft_array = fftn(array)
#         spect3d += np.real(fft_array * np.conj(fft_array))
#     # spect3d[..., 0] *= 0.5
#     x, y = shell_average(spect3d, N_points[0])
#     fh = open(fname + '.spectra', 'w')
#     fh.writelines(["%s\n" % item for item in y])
#     fh.close()
#     return y

# def load_DNS_data(data_folder):

#     N_points = 256
#     lx = 2*np.pi
#     datafile = dict()
#     datafile['u'] = os.path.join(data_folder, 'HIT_u.bin')
#     datafile['v'] = os.path.join(data_folder, 'HIT_v.bin')
#     datafile['w'] = os.path.join(data_folder, 'HIT_w.bin')
#     type_of_bin_data = np.float32

#     HIT_data = dict()
#     data_shape = (256, 256, 256)
#     for i in ['u', 'v', 'w']:
#         HIT_data[i] = np.reshape(np.fromfile(datafile[i], dtype=type_of_bin_data), data_shape)
#         HIT_data[i] = HIT_data[i]
#     for key, value in HIT_data.items():
#         HIT_data[key] = np.swapaxes(value, 0, 2)  # to put x index in first place


#     dx = [lx / N_points] * 3
#     print('calculating DNS spectra... Could take a while...')
#     out = os.path.join('./', 'DNS_spectra')
#     spectra = spectral_density([HIT_data['u'], HIT_data['v'], HIT_data['w']], dx, (N_points, N_points, N_points), out)
#     return spectra

# spectra = np.loadtxt(os.path.join(plot_folder, '{}.spectra'.format(fold)))


# data_folder = '/Users/pablo/Documents/ABCIMCMC/valid_data'
# spectra = load_DNS_data(data_folder)
# print (spectra)


# 2. From file 'JHU_DNS.spectra'
# 2.1. Set plotting properties
# single_column = 255
# oneandhalf_column = 397
# double_column = 539
# fig_width_pt = double_column
# inches_per_pt = 1.0/72.27               # Convert pt to inches
# golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
# fig_width = fig_width_pt*inches_per_pt  # width in inches
# fig_height = fig_width*golden_mean       # height in inches
# fig_size = [fig_width, fig_height]

# # mpl.rcParams['figure.figsize'] = 6.5, 2.2
# # plt.rcParams['figure.autolayout'] = True

# mpl.rcParams['font.size'] = 10
# mpl.rcParams['axes.titlesize'] = 1 * plt.rcParams['font.size']
# mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
# mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
# mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
# mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']

# mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rc('text', usetex=True)


# # plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
# mpl.rcParams['xtick.major.size'] = 3
# mpl.rcParams['xtick.minor.size'] = 3
# mpl.rcParams['xtick.major.width'] = 1
# mpl.rcParams['xtick.minor.width'] = 0.5
# mpl.rcParams['ytick.major.size'] = 3
# mpl.rcParams['ytick.minor.size'] = 3
# mpl.rcParams['ytick.major.width'] = 1
# mpl.rcParams['ytick.minor.width'] = 1
# # mpl.rcParams['legend.frameon'] = False
# # plt.rcParams['legend.loc'] = 'center left'
# plt.rcParams['axes.linewidth'] = 1

# # colors = ['red', 'cyan', 'yellow', 'green']
# colors = ['k', 'b', 'r']


# fig = plt.figure(figsize=(fig_width, fig_height))
# ax = plt.gca()

# #2 

data_folder = '/Users/pablo/Documents/ABCIMCMC'
jhu_data = 'JHU_DNS.spectra'

x2 = np.loadtxt(jhu_data, skiprows=2)[:, 0]
spect_1024 = np.loadtxt(jhu_data, skiprows=2)[:, 1]

ax.loglog(x2, spect_1024, label=r'$1024^3$ data')


# ax.set_ylabel(r'E(k)')
# ax.set_xlabel(r'k')
# ax.axis(ymin=1e-10)
# plt.legend(loc=0)

# fig.subplots_adjust(left=0.17, right=0.95, bottom=0.17, top=0.95)
# fig.savefig('./spectra_plots/1024_256_spectra')
# plt.close('all')


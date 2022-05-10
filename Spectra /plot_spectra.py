import os
import string
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# plt.style.use('dark_background')
# paper twocolumn elsevair
single_column = 252
oneandhalf_column = 397
double_column = 522
text_height = 682/ 72.27

def fig_size(width_column):
    fig_width_pt = width_column
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    return fig_width, fig_height

mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

# colors = ['red', 'cyan', 'yellow', 'green']
colors = ['k', 'b', 'r']

# # thesis
# single_column = 235
# oneandhalf_column = 352
# double_column = 470




def plot_MAP_spectra(plot_folder):

    spectra_DNS_1024 = np.loadtxt('./JHU_DNS.spectra', skiprows=2)
    files = ['smagorinsky', '3params_prod',  '3params_both', '3params_sigma',
               '4params_prod', '4params_both', '4params_sigma']
    spectra = np.empty((7, 33))

    spectra[0] = np.loadtxt(os.path.join(plot_folder, '{}.spectra'.format(files[0])))
    # grab spectra of 0 run (MAP)
    for i, file in enumerate(files[1:]):
        spectra[i+1] = np.loadtxt(os.path.join(plot_folder, '{}.spectra'.format(file)))[0]
    # colors = ['w', 'c', 'y', 'm', 'c', 'y', 'm', 'r']
    colors = ['b', 'orange', 'g', 'r', 'orange', 'g', 'r', 'k']
    linestyle = ['-', '--', '--', '--', '-', '-', '-']
    lw = [2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]    # linewidth
    labels = ['Smagorinsky', '3 param. production', '3 param. combined', '3 param. sigma',
              '4 param. production', '4 param. combined', '4 param. sigma']
    f_wigth, f_height = fig_size(single_column)
    fig = plt.figure(figsize=(f_wigth, f_height))
    ax = plt.gca()
    x = np.arange(spectra.shape[1])
    y = 1.6 * np.power(x, -5 / 3)
    ax.loglog(x, y, color=colors[-1], linestyle='--',  label=r'$-5/3$ slope')
    # ax.loglog(spectra_DNS_1024[:, 0], spectra_DNS_1024[:, 1], color=colors[-1], linestyle='-', lw=2, label='JHU DNS')

    for i, y in enumerate(spectra):
        ax.loglog(x, y, color=colors[i], linestyle=linestyle[i], lw=lw[i], label=labels[i])

    # ax.set_title('Forward runs spectra')
    ax.set_ylabel(r'$\widehat{u_i}^*\widehat{u_i}$', labelpad=2)
    ax.set_xlabel(r'$k$', labelpad=1.5)
    ax.axis(ymin=1e-5, ymax=1)
    ax.axis(xmin=1, xmax=29)
    # ax.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left', frameon=False)
    ax.legend(fontsize=8, labelspacing=0.25, handletextpad=0.8, borderaxespad=0, frameon=0)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.94)
    fig.savefig(plot_folder + 'spectra_MAP')
    print(plot_folder + 'spectra_MAP')
    plt.close('all')


def plot_all_spectra_IC(plot_folder):
    print('\n\nPlotting all IC spectra')

    colors = ['k', 'b', 'r', 'g', 'c']
    lw = 1  # base linewidth
    title = ['trained on sigma', 'trained on production', 'trained on sigma+prod']
    fig_width, fig_height = fig_size(double_column)
    fig, axarr = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True, figsize=(fig_width, 0.85 * fig_height))

    for p, n_param in enumerate([3, 4]):
        for n, fold in enumerate(['IC_' + (str(n_param) + i) for i in ['params_sigma', 'params_prod', 'params_both']]):
            print('{}'.format(fold))
            spectra = np.loadtxt(os.path.join(plot_folder, '{}.spectra'.format(fold)))

            x = np.arange(spectra.shape[1])
            y = 1.6 * np.power(x, -5 / 3)
            axarr[p, n].loglog(x, y, 'k--', label=r'$-5/3$ slope')

            counter = 0
            for i, y in enumerate(spectra[1:]):
                counter += 1
                axarr[p, n].loglog(x, y, 'm-', linewidth=lw)

    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.05, hspace=0.1, bottom=0.12, top=0.88)
    for i in range(2):
        axarr[i, 0].set_ylabel(r'$\widehat{u_i}^*\widehat{u_i}$')
        axarr[i, 0].text(-0.3, 0.92, string.ascii_lowercase[i] + ')', transform=axarr[i, 0].transAxes, size=10,
                         weight='bold')

    for i in range(3):
        axarr[1, i].set_xlabel(r'k')
        axarr[0, i].set_title(title[i])
    axarr[0, 0].axis(ymin=1e-6, ymax=10)
    axarr[0, 0].axis(xmin=1, xmax=29)
    # custom_lines = [Line2D([0], [0], color=colors[0], linestyle='--', lw=1),
    #                 Line2D([0], [0], color=colors[0], linestyle='-', lw=1),
    #                 Line2D([0], [0], color=colors[1], linestyle='-', lw=1),
    #                 Line2D([0], [0], color=colors[2], linestyle='-', lw=1), ]
    # axarr[0, 1].legend(custom_lines, ['-5/3 slope', 'DNS JHU', 'Smagorinsky', 'MAP'], loc='upper center',
    #                    bbox_to_anchor=(0.5, 1.4), frameon=False,
    #                    fancybox=False, shadow=False, ncol=4)
    fig.savefig(os.path.join(plot_folder, 'spectra_all_IC'))
    plt.close('all')


def plot_all_spectra(limit, plot_folder, high_probability_only=0.0):
    """ Plot for only stable spectra with parameters probability = high_probability_only,
    :param limit: if need to plot only part of spectra (from beginning to limit)
    :param plot_folder: path to save plots
    :param high_probability_only: lower limit to sampled parameters probability
    """
    print('\n\nPlotting all spectra')
    spectra_DNS_1024 = np.loadtxt(os.path.join(plot_folder, 'JHU_DNS.spectra'), skiprows=2)
    unstable = np.load(os.path.join(plot_folder, 'unstable.npz'),  allow_pickle=True)['unstable']

    # colors = ['k', 'b', 'r', 'g', 'c']
    colors = ['r', 'w', 'b', 'g', 'c']
    lw = 1   # base linewidth
    title = ['trained on sigma', 'trained on production', 'trained on sigma+prod']
    fig_width, fig_height = fig_size(double_column)
    fig, axarr = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True, figsize=(fig_width, 0.85*fig_height))

    spectra_smag = np.loadtxt(os.path.join(plot_folder, 'smagorinsky.spectra'))
    for p, n_param in enumerate([3, 4]):
        for n, fold in enumerate([(str(n_param) + i) for i in ['params_sigma', 'params_prod', 'params_both']]):
            print('{}'.format(fold))
            spectra = np.loadtxt(os.path.join(plot_folder, '{}.spectra'.format(fold)))
            probability = np.loadtxt(os.path.join(plot_folder, '{}.probability'.format(fold)))
            probability = probability[1:]/probability[0]

            x = np.arange(spectra.shape[1])
            # y = 1.6 * np.power(x, -5 / 3)
            # axarr[p, n].loglog(x, y, color=colors[0], linestyle='--', label=r'$-5/3$ slope')

            counter = 0
            for i, y in enumerate(spectra[1:]):
                if counter < limit:   # common minimum number for 0.7
                    if i+1 not in unstable[p, n] and probability[i] > high_probability_only:
                        counter += 1
                        axarr[p, n].loglog(x, y, 'm-', linewidth=probability[i] * lw, alpha=probability[i])
            print('{} lines plotted with probability > {}'.format(counter, high_probability_only))
            axarr[p, n].loglog(spectra_DNS_1024[:, 0], spectra_DNS_1024[:, 1], colors[0], linewidth=1.5 * lw, label='DNS JHU')
            axarr[p, n].loglog(x, spectra_smag, color=colors[1], linewidth=1 * lw, label='Smagorinsky')
            axarr[p, n].loglog(x, spectra[0], color=colors[2], linewidth=1.5 * lw, label='MAP')

    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.05, hspace=0.08, bottom=0.12, top=0.85)
    for i in range(2):
        axarr[i, 0].set_ylabel(r'$\widehat{u_i}^*\widehat{u_i}$')
        axarr[i, 0].text(-0.3, 0.92, string.ascii_lowercase[i]+')', transform=axarr[i, 0].transAxes, size=10, weight='bold')

    for i in range(3):
        axarr[1, i].set_xlabel(r'$k$')
        axarr[0, i].set_title(title[i])
    axarr[0, 0].axis(ymin=1e-6, ymax=10)
    axarr[0, 0].axis(xmin=1, xmax=29)
    custom_lines = [#Line2D([0], [0], color=colors[0], linestyle='--', lw=1),
                    Line2D([0], [0], color=colors[0], linestyle='-', lw=1),
                    Line2D([0], [0], color=colors[1], linestyle='-', lw=1),
                    Line2D([0], [0], color=colors[2], linestyle='-', lw=1),]
    axarr[0, 1].legend(custom_lines, [#'-5/3 slope',
                                      'DNS JHU', 'Smagorinsky', 'MAP'], loc='upper center',
                       bbox_to_anchor=(0.5, 1.45), frameon=False,
                       fancybox=False, shadow=False, ncol=4)
    fig.savefig(os.path.join(plot_folder, 'spectra_all_{}'.format(int(high_probability_only*100))))
    plt.close('all')


def plot_all_percentile(percent, limit, plot_folder, high_probability_only=0.0):
    """ Plot confidence interval for only stable spectra with parameters probability = high_probability_only,
    :param percent: confidence percent(shaded area on plot)
    :param limit: if need to plot only part of spectra (from beginning to limit)
    :param plot_folder: path to save plots
    :param high_probability_only: lower limit to sampled parameters probability
    """
    print('\n\nPlotting percentile')
    spectra_DNS_1024 = np.loadtxt('./JHU_DNS.spectra', skiprows=2)
    unstable = np.load(os.path.join(plot_folder, 'unstable.npz'),  allow_pickle=True)['unstable']
    spectra_smag = np.loadtxt(os.path.join(plot_folder, 'smagorinsky.spectra'))

    # colors = ['k', 'b', 'r', 'g', 'c']
    colors = ['r', 'w', 'b', 'g', 'c']
    title = ['trained on sigma', 'trained on production', 'trained on sigma+prod']
    fig_width, fig_height = fig_size(double_column)
    fig, axarr = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True, figsize=(fig_width, 0.85*fig_height))

    for p, n_param in enumerate([3, 4]):
        for n, fold in enumerate([(str(n_param) + i) for i in ['params_sigma', 'params_prod', 'params_both']]):
            print('{}'.format(fold))
            spectra = np.loadtxt(os.path.join(plot_folder, '{}.spectra'.format(fold)))
            spectra_map = spectra[0]
            probability = np.loadtxt(os.path.join(plot_folder, '{}.probability'.format(fold)))
            probability = probability[1:] / probability[0]
            # remove unstable and low probability spectra (lower than high_probability_only value)
            N_end = spectra.shape[0]
            ind = np.arange(N_end)
            ind = ind[np.where(probability[:N_end] > high_probability_only)]
            ind_unstable = []
            for i in range(len(ind)):
                if ind[i] in unstable[p, n]:
                    ind_unstable.append(i)
            ind = np.delete(ind, ind_unstable)
            ind = ind[1:limit+1]

            spectra = spectra[ind, :]

            x = np.arange(spectra.shape[1])
            # y = 1.6 * np.power(x, -5 / 3)
            # axarr[p, n].loglog(x, y, 'k--', label=r'$-5/3$ slope')

            spectra_down = np.percentile(spectra[1:], 100-percent, axis=0)
            spectra_up = np.percentile(spectra[1:], percent, axis=0)
            spectra_mean = np.mean(spectra[1:], axis=0)
            spectra_median = np.median(spectra[1:], axis=0)

            axarr[p, n].loglog(x, spectra_up, 'm-', linewidth=1.5, label='75')
            axarr[p, n].loglog(x, spectra_down, 'm-', linewidth=1.5, label='25')
            axarr[p, n].loglog(x, spectra_mean, colors[3], linewidth=1.5, label='mean')
            axarr[p, n].loglog(x, spectra_median, colors[4], linewidth=1.5, label='median')
            axarr[p, n].fill_between(x, spectra_down, spectra_up, facecolor='magenta', alpha=0.5)
            axarr[p, n].loglog(spectra_DNS_1024[:, 0], spectra_DNS_1024[:, 1], 'k-', linewidth=1.5, label='DNS JHU')
            axarr[p, n].loglog(x, spectra_smag, color=colors[1], linewidth=1.5, label='Smagorinsky')
            axarr[p, n].loglog(x, spectra_map, color=colors[2], linewidth=1.5, label='MAP')

    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.05, hspace=0.08, bottom=0.12, top=0.79)
    for i in range(2):
        axarr[i, 0].set_ylabel(r'$\widehat{u_i}^*\widehat{u_i}$')
        axarr[i, 0].text(-0.3, 0.92, string.ascii_lowercase[i]+')', transform=axarr[i, 0].transAxes, size=10, weight='bold')

    for i in range(3):
        axarr[1, i].set_xlabel(r'$k$')
        axarr[0, i].set_title(title[i])
    axarr[0, 0].axis(ymin=1e-6, ymax=10)
    axarr[0, 0].axis(xmin=1, xmax=29)
    custom_lines = [#Line2D([0], [0], color=colors[0], linestyle='--', lw=1),
                    Line2D([0], [0], color=colors[0], linestyle='-', lw=1),
                    Line2D([0], [0], color=colors[1], linestyle='-', lw=1),
                    Line2D([0], [0], color=colors[2], linestyle='-', lw=1),
                    Line2D([0], [0], color=colors[3], linestyle='-', lw=1),
                    Line2D([0], [0], color=colors[4], linestyle='-', lw=1),]
    axarr[0, 1].legend(custom_lines, [#'-5/3 slope',
                                      'DNS JHU', 'Smagorinsky', 'MAP', 'mean', 'median'], loc='upper center',
                       bbox_to_anchor=(0.5, 1.7), frameon=False,
                       fancybox=False, shadow=False, ncol=3)
    fig.savefig(os.path.join(plot_folder+'spectra_{}percentile_{}'.format(percent, int(high_probability_only*100))))
    plt.close('all')


base_path = '/Users/pablo/ABC_spectralLES/spectralLES/model_dev/'
plot_folder = os.path.join(base_path, 'spectra_plots/')

plot_MAP_spectra(plot_folder)
# to plot all stable spectra
# plot_all_spectra(limit=100, plot_folder=plot_folder)
# # to plot only spectra with probability > 70%
# # Warning: has limit to plot only 16 spectra
# plot_all_spectra(limit=100, plot_folder=plot_folder, high_probability_only=0.7)
#
# for percentile in [75, 90, 95, 99, 100]:
#     # to plot all stable spectra
#     plot_all_percentile(percentile, limit=100, plot_folder=plot_folder)
#     # to plot only spectra with probability > 70%
#     # Warning: has limit to plot only 16 spectra
#     plot_all_percentile(percentile, limit=100,  plot_folder=plot_folder, high_probability_only=0.7)
#
# plot_all_spectra_IC(plot_folder)

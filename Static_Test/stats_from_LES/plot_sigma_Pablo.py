import os
import string
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 1 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 1
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
plt.style.use('dark_background')
# colors = ['red', 'cyan', 'yellow', 'green']
colors = ['r', 'w', 'b', 'g', 'c']

# single_column = 255
# oneandhalf_column = 397
# double_column = 539
# thesis
single_column = 235
oneandhalf_column = 352
double_column = 470

def fig_size(width_column):
    fig_width_pt = width_column
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    return fig_width, fig_height


def plot_sum_stat(N_params, limit, plot_folder, high_probability_only=0.0):

    titles = [r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{13}$', r'$\sigma_{ij}S_{ij}$']
    lw = 1
    fig_width, fig_height = fig_size(double_column)
    fig, axarr = plt.subplots(nrows=3, ncols=4, sharey=True, figsize=(fig_width, fig_height))

    y_true = np.loadtxt(os.path.join(plot_folder, 'sum_stat_true'))
    x_true = np.loadtxt(os.path.join(plot_folder, 'sum_stat_bins'))
    if N_params == 3:
        unstable = np.load(os.path.join(plot_folder, 'unstable.npz'), allow_pickle=True)['unstable'][0]
    else:
        unstable = np.load(os.path.join(plot_folder, 'unstable.npz'), allow_pickle=True)['unstable'][1]

    files = [str(N_params) + 'params_sigma', str(N_params) + 'params_prod', str(N_params) + 'params_both']
    sigma_pdf_sm = np.load(os.path.join(plot_folder, 'smag_sigma_pdf.npz'))['pdf']
    # x = np.load(os.path.join(folder_smag, 'sigma_pdf_sm.npz'))['x']
    x = x_true[0]
    prod_pdf_sm = np.load(os.path.join(plot_folder, 'smag_prod_pdf.npz'))['pdf']
    # x_prod = np.load(os.path.join(folder_smag, 'prod_pdf_sm.npz'))['x']
    x_prod = x_true[1]

    for i, file in enumerate(files):
        print('\n', file)
        sigma_pdf = np.load(os.path.join(plot_folder, file+'_sigma_pdf.npz'))['pdf']
        print(sigma_pdf.shape)
        prod_pdf = np.load(os.path.join(plot_folder, file+'_prod_pdf.npz'))['pdf']
        print(prod_pdf.shape)
        probability = np.loadtxt(os.path.join(plot_folder, file+'.probability'))
        probability = probability[1:] / probability[0]

        for j, pdf_ij in enumerate(sigma_pdf[:3]):
            counter = 0
            for k, pdf in enumerate(pdf_ij[1:]):
                if counter < limit:   # common minimum number for 0.7
                    if k+1 not in unstable[i] and probability[k] > high_probability_only:
                        counter += 1
                        axarr[i, j].plot(x, pdf, 'm-', linewidth=probability[k]*lw, alpha=probability[k])
            print('{} lines plotted with probability > {}'.format(counter, high_probability_only))
            axarr[i, j].plot(x_true[0], y_true[j], color=colors[0], linewidth=lw, label='DNS')
            axarr[i, j].plot(x, pdf_ij[0], color=colors[2], linewidth=lw, label='MAP')
            print('check:{} {} {}, sum = {}'.format(i, j, np.sum(np.exp(y_true[j])), np.sum(np.exp(pdf_ij[0]))))
            axarr[i, j].plot(x, sigma_pdf_sm[j], color=colors[1], linewidth=lw, label='Smag.')

        counter = 0
        for k, pdf in enumerate(prod_pdf[1:]):
            if counter < limit:  # common minimum number for 0.7
                if k + 1 not in unstable[i] and probability[k] > high_probability_only:
                    counter += 1
                    axarr[i, 3].plot(x_prod, pdf, 'm-', linewidth=probability[k]*lw, alpha=probability[k])
        print('{} lines plotted with probability > {}'.format(counter, high_probability_only))

        axarr[i, 3].plot(x_true[1], y_true[3], color=colors[0], linewidth=lw, label='DNS')
        axarr[i, 3].plot(x_prod, prod_pdf[0], color=colors[2], linewidth=lw, label='MAP')
        axarr[i, 3].plot(x_prod, prod_pdf_sm, color=colors[1], linewidth=lw, label='Smag.')


    # ax.loglog(x, data[0], 'c-', linewidth=1.5*lw, label='Smagorinsky')
    # ax.loglog(x, data[1], 'y-', linewidth=1.5*lw, label='best')

    for i in range(4):
        axarr[2, i].set_xlabel(titles[i], labelpad=1)
        axarr[2, i].axis(ymin=-7)
    for i in range(3):
        axarr[i, 0].set_ylabel(r'ln(pdf)', labelpad=0)
        axarr[i, 3].set_ylim(bottom=-7)
        axarr[i, 3].set_xlim([-5, 5])
        axarr[i, 3].xaxis.set_major_locator(ticker.MultipleLocator(2))
        axarr[i, 3].tick_params(direction="in")
        if i < 2:
            axarr[i, 3].xaxis.set_major_formatter(plt.NullFormatter())
        for ind in range(3):
            axarr[i, 0].text(-0.4, 0.92, string.ascii_lowercase[i]+')', transform=axarr[i, 0].transAxes, size=10, weight='bold')
            axarr[i, ind].set_xlim([-0.3, 0.3])
            # axarr[i, ind].set_ylim([-7, 4])
            # axarr[i, ind].set_xticks([])

            axarr[i, ind].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            axarr[i, ind].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            axarr[i, ind].tick_params(direction="in", which='major')
            axarr[i, ind].tick_params(direction="in", which='minor')
            if i < 2:
                axarr[i, ind].xaxis.set_major_formatter(plt.NullFormatter())

    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.05, hspace=0.1, bottom=0.1, top=0.93)

    custom_lines = [Line2D([0], [0], color=colors[0], lw=1),
                    Line2D([0], [0], color=colors[1], linestyle='-', lw=1),
                    Line2D([0], [0], color=colors[2], linestyle='-', lw=1)]
    axarr[0, 1].legend(custom_lines, ['DNS', 'Smagorinsky', 'MAP'], loc='upper center',
                       bbox_to_anchor=(0.99, 1.35), frameon=False,
                       fancybox=False, shadow=False, ncol=3)
    fig.savefig(os.path.join(plot_folder, '{}_sigma_pdf_{}'.format(N_params, int(high_probability_only*100))))
    plt.close('all')

base_filename = '/Users/pablo/Documents/ABC_spectralLES/spectralLES/model_dev'
plot_folder = os.path.join(base_filename, 'sigma_plots')
plot_sum_stat(N_params=3, limit=100, plot_folder=plot_folder)
plot_sum_stat(N_params=4, limit=100, plot_folder=plot_folder)

plot_sum_stat(N_params=3, limit=100, plot_folder=plot_folder, high_probability_only=0.7)
plot_sum_stat(N_params=4, limit=100, plot_folder=plot_folder, high_probability_only=0.7)
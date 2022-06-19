import os
import glob
import string
import numpy as np


def grab_spectra(sample_folder, avg):
    """
    Go to sample_folder and
    calculate the mean spectra over avg last spectra outputs
    :param sample_folder: folder of one forward run
    :param avg: number of spectra to average over
    :return: 1d array of spectra
    """
    spectra = None
    for root, dirs, files in os.walk(sample_folder):
        if root == os.path.join(sample_folder, 'analysis'):
            f_array = sorted(files)[-avg - 1:-1]
            data = np.empty((len(f_array), 33))
            for k in range(len(f_array)):
                f = open(os.path.join(root, f_array[k]), 'r')
                data[k] = np.array(f.readlines()[1:]).astype(np.float)
            spectra = np.mean(data, axis=0)
    return spectra


def find_unstable(base_path, N_start, N_end):
    unstable_all = []
    percent_all = []
    for p, N_params in enumerate([3, 4]):
        folders = [str(N_params) + 'params_sigma/', str(N_params) + 'params_prod/', str(N_params) + 'params_both/']
        unstable_param = []
        percent_param = []
        for i, folder in enumerate(folders):
            print('\n', folder)
            folder = os.path.join(base_path, folder)
            probability = np.loadtxt(os.path.join(folder, 'probability_from_posterior_'+str(N_params)))
            probability /= probability[0]
            unstable = []
            percent = []
            for j in range(N_start[p, i], N_end[p, i] + 1):
                sample_folder = os.path.join(folder, str(j))
                for root, dirs, files in os.walk(sample_folder):
                    if root == os.path.join(sample_folder, 'data'):
                        end = sorted(files)[-1][-7:-4]
                        if end != '032':
                            unstable.append(j)
                            percent.append(int(100*probability[N_start[p, i]+j]))
                            print('unstable run {}, probability {}%'.format(j, int(100*probability[N_start[p, i]+j])))
            unstable_param.append(unstable)
            percent_param.append(percent)
        unstable_all.append(unstable_param)
        percent_all.append(percent_param)
    np.savez(os.path.join(plot_folder, 'unstable'), unstable=np.array(unstable_all), percent=np.array(percent_all))
    return np.array(unstable_all)


def make_txt_spectra(base_path, avg, N_end, plot_folder):
    """ Goes through all folders, average spectra over last
        'avg' outputs and write them into .txt file.
    :param base_path:
    :param avg: number of outputs to average over
    :param N_end:
    :param plot_folder: path for writing .txt file
    :return:
    """
    N_end = N_end + 1

    for p, n_param in enumerate([3, 4]):
        for n, fold in enumerate([(str(n_param) + i) for i in ['params_sigma/', 'params_prod/', 'params_both/']]):
            print('{}'.format(fold))
            folder = os.path.join(base_path, fold)

            spectra = np.zeros((N_end[p, n], 33))
            for i in range(N_end[p, n]):
                spectra[i] = grab_spectra(os.path.join(folder, str(i)), avg)
            spectra /= 64 ** 6
            np.savetxt(os.path.join(plot_folder, '{}.spectra'.format(fold[:-1])), spectra)
    spectra_smag = grab_spectra(os.path.join(base_path, 'smagorinsky'), avg) / 64 ** 6
    np.savetxt(os.path.join(plot_folder, 'smagorinsky.spectra'), spectra_smag)


def make_txt_spectra_IC(base_path, avg, N_ic, plot_folder):
    """ Goes through all IC folders, average spectra over last
    'avg' outputs and write them into .txt file.
    :param base_path:
    :param avg: number of outputs to average over
    :param N_end:
    :param plot_folder: path for writing .txt file
    :return:
    """
    for p, n_param in enumerate([3, 4]):
        for n, fold in enumerate([(str(n_param) + i) for i in ['params_sigma/', 'params_prod/', 'params_both/']]):
            print('{}'.format(fold))
            folder = os.path.join(base_path, fold)
            spectra = np.zeros((N_ic, 33))
            for i in range(N_ic):
                print(os.path.join(folder, 'IC_run'+str(i)))
                spectra[i] = grab_spectra(os.path.join(folder, 'IC_run'+str(i+1)), avg)
            spectra /= 64 ** 6
            np.savetxt(os.path.join(plot_folder, 'IC_{}.spectra'.format(fold[:-1])), spectra)


def find_unstable_IC(base_path, N_ic):
    unstable_all = []
    for p, N_params in enumerate([3, 4]):
        folders = [str(N_params) + 'params_sigma/', str(N_params) + 'params_prod/', str(N_params) + 'params_both/']
        unstable_param = []
        for i, folder in enumerate(folders):
            print('\n', folder)
            folder = os.path.join(base_path, folder)
            unstable = []
            percent = []
            for j in range(N_ic):
                sample_folder = os.path.join(folder, 'IC_run{}'.format(j+1))
                for root, dirs, files in os.walk(sample_folder):
                    if root == os.path.join(sample_folder, 'data'):
                        end = sorted(files)[-1][-7:-4]
                        if end != '032':
                            unstable.append(j)
                            print('unstable run {}'.format(j+1))
            unstable_param.append(unstable)
        unstable_all.append(unstable_param)
    np.savez(os.path.join(plot_folder, 'unstable_IC'), unstable=np.array(unstable_all))
    return np.array(unstable_all)


base_path = '/home/olga/teslapy/spectralLES/model_dev/'
plot_folder = os.path.join(base_path, 'spectra_plots/')
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)
averaging = 20      # average spectra over last 20 outputs


# N_start = np.array([[0, 0, 0], [0, 0, 0]])
# N_end_arr = np.array([[100, 100, 100], [100, 100, 100]])
# make_txt_spectra(base_path, averaging, N_end_arr, plot_folder)
# # plot_MAP_spectra(base_path=base_path, avg=averaging, plot_folder=plot_folder)
# unstable = find_unstable(base_path, N_start, N_end_arr)


make_txt_spectra_IC(base_path=base_path, avg=averaging, N_ic=20, plot_folder=plot_folder)
find_unstable_IC(base_path=base_path, N_ic=20)

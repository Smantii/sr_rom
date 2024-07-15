import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sr_rom.data.data import process_data, smooth_data, split_data
from scipy.interpolate import interp1d
from sr_rom.code.two_scale_dd_vms_rom_closure import main
import os
import shutil
from ray.util.multiprocessing import Pool
from functools import partial


def get_dir(task, folder_name, w, test_perc):
    main_dir = '/home/smanti/SR-ROM/src/sr_rom/'
    task_folder = main_dir + "results_" + task + "/"
    dir = task_folder + folder_name + '/results_' + \
        str(test_perc) + "/results_w_" + str(w) + "_n_2/"
    # if os.path.exists(dir):
    #    shutil.rmtree(dir)
    # os.makedirs(dir)
    return dir


def f(idx_Re):
    _, _, _, _, sq_avgerr_L2, sq_avgrelerr_L2 = main(
        int(Re[idx_Re]), method, dir, idx_Re, False)
    return idx_Re, sq_avgerr_L2, sq_avgrelerr_L2


# load data
t = np.linspace(500., 520., 2001)
t_full = np.linspace(500., 520., 20001)


test_perc_list = [20, 40, 60, 80]
windows = [3, 5, 7]
method = "NN"
shuffle = False
task = shuffle*"interpolation" + (1-shuffle)*"extrapolation"

for test_perc in test_perc_list:
    for w in windows:
        print(f"Collecting results for {test_perc}% test and window {w}", flush=True)
        Re, A, B, tau, a_FOM, X = process_data(5, "2dcyl/Re200_300")
        A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=w, num_smoothing=2, r=5)

        # split in training and test
        train_data, val_data, train_val_data, test_data = split_data(
            Re, A_conv, B_conv, tau_conv, a_FOM, X, test_perc/100, True)

        train_Re_idx = train_val_data.y["idx"]

        if method == "LI":
            folder_name = "li_tau"
            dir = get_dir(task, folder_name, w, test_perc)
            # linearly interpolate w.r.t. Reynolds and time
            tau_interp = np.zeros((61, 20001, 5))*np.nan
            for i in range(5):
                tau_conv_interp_Re = interp1d(
                    Re[train_Re_idx], tau_conv[train_Re_idx, :, i], axis=0, fill_value="extrapolate")
                tau_Re = tau_conv_interp_Re(Re)
                tau_conv_interp_t = interp1d(
                    t, tau_Re, axis=1, fill_value="extrapolate")
                tau_interp[:, :, i] = tau_conv_interp_t(t_full)
            np.save(dir + "tau_interp.npy", tau_interp)
        elif method == "SR":
            folder_name = "sr"
            dir = get_dir(task, folder_name, w, test_perc)
        elif method == "NN":
            folder_name = "nn"
            dir = get_dir(task, folder_name, w, test_perc)

        # save mean and std of X and y
        for i in range(5):
            y_train = train_val_data.y["tau"][:, :, i].flatten("F")

            # Standardization
            mean_std_X_train = [np.mean(train_val_data.y["X"][:, 1:], axis=0),
                                np.std(train_val_data.y["X"][:, 1:], axis=0)]
            mean_std_train_comp = [np.mean(y_train, axis=0),
                                   np.std(y_train, axis=0)]
            np.save(dir + "mean_std_X_train.npy", mean_std_X_train)
            np.save(dir + "mean_std_train_comp_" + str(i) + ".npy", mean_std_train_comp)

        l2_error = np.zeros(len(Re))
        l2_rel_error = np.zeros(len(Re))

        pool = Pool(2)
        for result in pool.map(f, range(2)):
            l2_error[result[0]] = result[1]
            l2_rel_error[result[0]] = result[2]
            print(result)

        np.save(dir + "l2_error.npy", l2_error)
        np.save(dir + "l2_rel_error.npy", l2_rel_error)

    print("Done!")

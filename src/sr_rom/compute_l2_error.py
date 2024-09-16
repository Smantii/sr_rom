import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sr_rom.data.data import process_data, smooth_data, split_data
from scipy.interpolate import interp1d
from sr_rom.code.two_scale_dd_vms_rom_closure import main
import os
from ray.util.multiprocessing import Pool
from functools import partial
import joblib
from sklearn.linear_model import LinearRegression


def get_dir(task, folder_name, w, test_perc, make_dir=False):
    main_dir = '/home/smanti/SR-ROM/src/sr_rom/'
    task_folder = main_dir + "results_" + task + "/"
    dir = task_folder + folder_name + '/results_' + \
        str(test_perc) + "/results_w_" + str(w) + "_n_2/"
    if make_dir:
        os.makedirs(dir)
    return dir


def f(idx_Re):
    _, avgerr_L2, _, avgrelerr_L2, _, _, _ = main(
        int(Re[idx_Re]), method, dir, idx_Re, False)
    return idx_Re, avgerr_L2, avgrelerr_L2


# load data
t = np.linspace(500., 520., 2001)
t_full = np.linspace(500., 520., 20001)


test_perc_list = [40, 55, 70, 85]
windows = [3, 5, 7]
method = "SR"
shuffle = False
task = shuffle*"interpolation" + (1-shuffle)*"extrapolation"
t_sample = 200
r = 2

Re, A, B, tau, a_FOM, X, X_sampled, residual = process_data(
    r, "2dcyl/Re200_400", t_sample)

for test_perc in test_perc_list:
    for w in windows:
        print(f"Collecting results for {test_perc}% test and window {w}", flush=True)
        A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=w, num_smoothing=2, r=r)
        r_2 = np.zeros(r)
        # u_coefs = np.zeros((61, 2001, r+1))

        # split in training and test
        train_data, val_data, train_val_data, test_data = split_data(
            Re, A_conv, B_conv, tau_conv, a_FOM, X, X_sampled, residual, 0.85, shuffle)

        train_Re_idx = train_val_data.y["idx"]
        test_Re_idx = test_data.y["idx"]

        if method == "LI":
            folder_name = "li_tau"
            dir = get_dir(task, folder_name, w, test_perc)
            # linearly interpolate w.r.t. Reynolds and time
            tau_interp = np.zeros((61, 20001, r))*np.nan
            for i in range(r):
                tau_conv_interp_Re = interp1d(
                    Re[train_Re_idx], tau_conv[train_Re_idx, ::t_sample, i], axis=0, fill_value="extrapolate")
                tau_Re = tau_conv_interp_Re(Re)
                tau_conv_interp_t = interp1d(
                    t[::t_sample], tau_Re, axis=1, fill_value="extrapolate")
                tau_interp[:, :, i] = tau_conv_interp_t(t_full)
                r_2[i] = 1 - np.sum((tau_interp[test_Re_idx, ::10, i] - tau_conv[test_Re_idx, :, i])**2)/np.sum(
                    (tau_conv[test_Re_idx, :, i] - np.mean(tau_conv[test_Re_idx, :, i]))**2)

            np.save(dir + "tau_interp.npy", tau_interp)
            np.savetxt(dir + "r_2_test.txt", r_2)
        elif method == "SR":
            folder_name = "sr"
            dir = get_dir(task, folder_name, w, test_perc)
        elif method == "NN":
            folder_name = "nn"
            dir = get_dir(task, folder_name, w, test_perc)

        elif method == "LR":
            folder_name = "lr"
            dir = get_dir(task, folder_name, w, test_perc, True)
            for i in range(r):
                y_train = train_val_data.y["tau"][:, :, i].flatten("F")
                y_train_sampled = train_val_data.y["tau"][:, ::t_sample, i].flatten("F")
                y_test = test_data.y["tau"][:, :, i].flatten("F")

                # Standardization
                mean_std_X_train = [np.mean(train_val_data.y["X_sampled"][:, 1:], axis=0),
                                    np.std(train_val_data.y["X_sampled"][:, 1:], axis=0)]
                mean_std_train_comp = [np.mean(y_train, axis=0),
                                       np.std(y_train, axis=0)]
                X_train_norm = (train_val_data.y["X"][:, 1:] -
                                mean_std_X_train[0])/mean_std_X_train[1]
                X_sampled_train_norm = (
                    train_val_data.y["X_sampled"][:, 1:] - mean_std_X_train[0])/mean_std_X_train[1]
                y_train_norm = (y_train - mean_std_train_comp[0]) / \
                    mean_std_train_comp[1]
                y_sampled_train_norm = (y_train_sampled - mean_std_train_comp[0]) / \
                    mean_std_train_comp[1]
                X_test_norm = (test_data.y["X"][:, 1:] -
                               mean_std_X_train[0])/mean_std_X_train[1]
                y_test_norm = (y_test - mean_std_train_comp[0])/mean_std_train_comp[1]

                # reshuffling
                p_train = np.random.permutation(len(X_sampled_train_norm))
                X_sampled_train_norm = X_sampled_train_norm[p_train]
                y_sampled_train_norm = y_sampled_train_norm[p_train]

                reg = LinearRegression()
                reg.fit(X_sampled_train_norm, y_sampled_train_norm)
                r_2[i] = reg.score(X_test_norm, y_test_norm)
                joblib.dump(reg, dir + "model_" + str(i) + ".pkl")

            np.savetxt(dir + "r_2_test.txt", r_2)

        # save mean and std of X and y
        for i in range(r):
            y_train = train_val_data.y["tau"][:, ::t_sample, i].flatten("F")

            # Standardization
            mean_std_X_train = [np.mean(train_val_data.y["X_sampled"][:, 1:], axis=0),
                                np.std(train_val_data.y["X_sampled"][:, 1:], axis=0)]
            mean_std_train_comp = [np.mean(y_train, axis=0),
                                   np.std(y_train, axis=0)]
            np.save(dir + "mean_std_X_train.npy", mean_std_X_train)
            np.save(dir + "mean_std_train_comp_" + str(i) + ".npy", mean_std_train_comp)

        l2_error = np.zeros(len(Re))
        l2_rel_error = np.zeros(len(Re))

        pool = Pool(32)
        for result in pool.map(f, range(len(Re))):
            l2_error[result[0]] = result[1]
            l2_rel_error[result[0]] = result[2]
            # u_coefs[result[0]] = result[3]
            print(result)

        np.save(dir + "l2_error.npy", l2_error)
        np.save(dir + "l2_rel_error.npy", l2_rel_error)
        # np.save(dir + "u_coefs.npy", u_coefs)

    print("Done!")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sr_rom.data.data import process_data, smooth_data, split_data
from scipy.interpolate import interp1d
from sr_rom.code.two_scale_dd_vms_rom_closure import main
import os
import shutil


# load data
t = np.linspace(500., 520., 2001)
t_full = np.linspace(500., 520., 20001)

test_perc_list = [20, 40, 60, 80]
windows = [3, 5, 7]

for test_perc in test_perc_list:
    for w in windows:
        print(f"Collecting results for {test_perc}% test and window {w}")
        Re, A, B, tau, a_FOM, X = process_data(5, "2dcyl/Re200_300")
        A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=w, num_smoothing=2, r=5)

        # split in training and test
        train_data, val_data, train_val_data, test_data = split_data(
            Re, A_conv, B_conv, tau_conv, a_FOM, X, test_perc/100, True)

        train_Re_idx = train_val_data.y["idx"]

        dir = os.path.dirname(os.path.realpath(__file__)) + '/results_interpolation/li_A_B/results_' + \
            str(test_perc) + "/results_w_" + str(w) + "_n_2/"
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        # linearly interpolate w.r.t. Reynolds
        for i in range(5):
            A_conv_interp_fun = interp1d(
                Re[train_Re_idx], A_conv[train_Re_idx, :, :], axis=0, fill_value="extrapolate")
            B_conv_interp_fun = interp1d(
                Re[train_Re_idx], B_conv[train_Re_idx, :, :, :], axis=0, fill_value="extrapolate")
            A_conv_interp = A_conv_interp_fun(Re)
            B_conv_interp = B_conv_interp_fun(Re)

        np.save(dir + "A_conv_interp.npy", A_conv_interp)
        np.save(dir + "B_conv_interp.npy", B_conv_interp)

        l2_error = []
        l2_rel_error = []
        for idx, Re in enumerate(Re):
            print(Re)
            _, _, _, _, sq_avgerr_L2, sq_avgrelerr_L2 = main(
                int(Re), "LI", dir, idx, False)
            print(sq_avgerr_L2, sq_avgrelerr_L2)
            l2_error.append(sq_avgerr_L2)
            l2_rel_error.append(sq_avgrelerr_L2)

        np.save(dir + "l2_error.npy", l2_error)
        np.save(dir + "l2_rel_error.npy", l2_rel_error)

        print("Done!")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sr_rom.data.data import process_data, smooth_data, split_data
from scipy.interpolate import interp1d
from sr_rom.code.two_scale_dd_vms_rom_closure import main
import os

# load data
Re, A, B, tau, a_FOM, X = process_data(5, "2dcyl/Re200_300")
A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=3, num_smoothing=2, r=5)
t = np.linspace(500., 520., 2001)
t_full = np.linspace(500., 520., 20001)

# split in training and test
train_data, val_data, train_val_data, test_data, (train_Re_idx, test_Re_idx) = split_data(
    Re, A_conv, B_conv, tau_conv, a_FOM, X, 0.2, True)

# linearly interpolate w.r.t. Reynolds and time
tau_interp = np.zeros((61, 20001, 5))*np.nan
for i in range(5):
    tau_conv_interp_Re = interp1d(
        Re[train_Re_idx], tau_conv[train_Re_idx, :, i], axis=0, fill_value="extrapolate")
    tau_Re = tau_conv_interp_Re(Re)
    tau_conv_interp_t = interp1d(t, tau_Re, axis=1, fill_value="extrapolate")
    tau_interp[:, :, i] = tau_conv_interp_t(t_full)

test_Re = Re[test_Re_idx]
l2_error = []

# load DD VMS-ROM csv file
true_l2_error = np.loadtxt(
    "/home/smanti/SR-ROM/src/sr_rom/results_20/results_w_3_n_2/vmsrom_l2_error.csv", delimiter=",", skiprows=1)

# for Re, idx in zip(test_Re, test_Re_idx):
# for idx, Re in enumerate(Re):
#    istep, err_l2_avg, err_h10_avg = main(int(Re), tau_interp, idx)
#   l2_error.append(err_l2_avg)

# np.save("l2_error_test.npy", l2_error)

l2_error = np.load("l2_error_test.npy")

test_ord = np.argsort(test_Re)


plt.plot(Re, true_l2_error[:, 1], c="#e41a1c", marker='o', label="DD VMS-ROM")
plt.plot(Re, l2_error, c='#377eb8', marker='o', label="VMS-ROM interp")
plt.scatter(test_Re, 0.1*np.ones_like(test_Re), marker="*", c="#e41a1c")
# plt.xticks(Re)
plt.xlabel("Re")
plt.ylabel(r"$\epsilon_{L^2}$")
plt.ylim(bottom=0.093)

plt.legend()
plt.savefig("error_plot_20.png", dpi=300)


print(
    f"Mean +/- std L2 test error DD VMS-ROM = {np.mean(true_l2_error[test_Re_idx,1])} +/- {np.std(true_l2_error[test_Re_idx,1])}")
print(f"Mean +/- std L2 error = {np.mean(l2_error)} +/- {np.std(l2_error)}")

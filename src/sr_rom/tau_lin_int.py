import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sr_rom.data.data import process_data, smooth_data, split_data
from scipy.interpolate import interp1d

# load data
Re, A, B, tau, a_FOM, X = process_data(5, "2dcyl/Re200_300")
A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=3, num_smoothing=2, r=5)

# split in training and test
train_data, val_data, train_val_data, test_data, (train_Re_idx, test_Re_idx) = split_data(
    Re, A_conv, B_conv, tau_conv, a_FOM, X, 0.2, True)

# linearly interpolate w.r.t. Reynolds
tau_conv_interp_fun = interp1d(
    Re[train_Re_idx], tau_conv[train_Re_idx, :, 0], axis=0, fill_value="extrapolate")
tau_conv_interp = tau_conv_interp_fun(Re)

# plot the results
t = np.arange(2001)
re_mesh, t_mesh = np.meshgrid(Re, t)

fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={
                       "projection": "3d"},  figsize=(20, 10))

# Plot the surface.
surf = ax[0].plot_surface(re_mesh, t_mesh, tau_conv[:, :, 0].T, cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
surf_comp = ax[1].plot_surface(re_mesh, t_mesh, tau_conv_interp.T, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
ax[0].set_xlabel(r"$Re$")
ax[0].set_ylabel(r"time index")
ax[1].set_xlabel(r"$Re$")
ax[1].set_ylabel(r"time index")
ax[0].set_title(r"VMS-ROM Closure term")
ax[1].set_title(r"Linearly interpolated closure term")
plt.colorbar(surf, shrink=0.5)
plt.colorbar(surf_comp, shrink=0.5)
plt.show()

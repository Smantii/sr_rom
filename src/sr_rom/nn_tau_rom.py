from sr_rom.data.data import process_data, split_data, smooth_data
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import cm


class NeuralNetwork(nn.Module):
    def __init__(self, r, hidden_units, dropout_rate=0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        # define hidden_layers
        hidden_layers = [nn.Linear(r+1, hidden_units[0]),
                         nn.LeakyReLU(), nn.Dropout(dropout_rate)]
        for i in range(1, len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.LeakyReLU())
            hidden_layers.append(nn.Dropout(dropout_rate))
        # append last layer
        hidden_layers.append(nn.Linear(hidden_units[-1], 1))

        # nn stack
        self.linear_relu_stack = nn.Sequential(*hidden_layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


output_path = sys.argv[1]
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.", flush=True)
else:
    print("No GPU available. Training will run on CPU.", flush=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush=True)

t_sample = 200
r = 2
Re, A, B, tau, a_FOM, X, X_sampled, residual = process_data(
    r, "2dcyl/Re200_400", t_sample=t_sample)
A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=7, num_smoothing=2, r=r)
train_data, val_data, train_val_data, test_data = split_data(
    Re, A_conv, B_conv, tau_conv, a_FOM, X, X_sampled, residual, 0.8, shuffle_test=False)

num_Re = len(Re)
num_t = tau.shape[1]

t = np.linspace(500, 520, 2001)

for i in range(r):
    idx_train = np.argsort(train_val_data.y["X_sampled"][:, 0])
    idx_test = np.argsort(test_data.y["X"][:, 0])
    y_train = train_val_data.y["tau"][:, ::t_sample, i].flatten("F")[idx_train]
    y_test = test_data.y["tau"][:, :, i].flatten("F")[idx_test]

    # Standardization
    mean_std_X_train = [np.mean(train_val_data.y["X_sampled"][idx_train, 1:], axis=0),
                        np.std(train_val_data.y["X_sampled"][idx_train, 1:], axis=0)]
    X_train_norm = train_val_data.y["X_sampled"][idx_train]
    X_test_norm = test_data.y["X"][idx_test]
    X_train_norm[:, 0] /= 1000
    X_test_norm[:, 0] /= 1000
    X_train_norm[:, 1:] = (X_train_norm[:, 1:] -
                           mean_std_X_train[0])/mean_std_X_train[1]
    X_test_norm[:, 1:] = (X_test_norm[:, 1:] - mean_std_X_train[0])/mean_std_X_train[1]
    y_train_norm = y_train
    y_test_norm = y_test

    X_train_norm = torch.from_numpy(X_train_norm).to(torch.float32)
    y_train_norm = torch.from_numpy(y_train_norm.reshape(-1, 1)).to(torch.float32)
    X_test_norm = torch.from_numpy(X_test_norm).to(torch.float32)
    y_test_norm = torch.from_numpy(y_test_norm.reshape(-1, 1)).to(torch.float32)

    model = NeuralNetRegressor(module=NeuralNetwork, batch_size=512, verbose=0,
                               optimizer=torch.optim.Adam, max_epochs=300,
                               train_split=None, device="cuda", iterator_train__shuffle=True)

    params = {'lr': [1e-4, 1e-3],
              'optimizer__weight_decay': [1e-5, 1e-4, 1e-3],
              'module__hidden_units': [[64, 128, 256, 128, 64],
                                       [64, 128, 256, 512, 256, 128, 64],
                                       [128, 256, 512, 1024, 512, 256, 128]],
              'module__dropout_rate': [0.4, 0.5],
              'module__r': [2]
              }

    tic = time.time()
    gs = GridSearchCV(model, params, cv=2, verbose=3,
                      scoring="neg_mean_squared_error", refit=True, n_jobs=3, return_train_score=True)
    gs.fit(X_train_norm, y_train_norm)
    print(f"Completed in {time.time() - tic}", flush=True)
    print(f"The best parameters are {gs.best_params_}")

    r2_train = 1 + gs.score(X_train_norm, y_train_norm) / \
        torch.mean((y_train_norm - torch.mean(y_train_norm))**2)
    r2_test = 1 + gs.score(X_test_norm, y_test_norm) / \
        torch.mean((y_test_norm - torch.mean(y_test_norm))**2)

    print(f"The R^2 in the training set is {r2_train.item()}", flush=True)
    print(f"The R^2 in the test set is {r2_test.item()}", flush=True)

    # plot prediction
    X_scaled = X
    X_scaled[:, 0] /= 1000
    X_scaled[:, 1:] = (X[:, 1:] - mean_std_X_train[0])/mean_std_X_train[1]

    model_out = gs.predict(torch.from_numpy(X_scaled).to(torch.float32).cuda())

    # rescale out back
    pred = model_out.reshape((num_Re, num_t), order="F")

    re_mesh, t_mesh = np.meshgrid(Re, t)

    fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={
        "projection": "3d"},  figsize=(20, 10))

    # Plot the surface.
    surf = ax[0].plot_surface(re_mesh, t_mesh, tau_conv[:, :, i].T, cmap=cm.coolwarm,
                              linewidth=0, antialiased=False)
    surf_comp = ax[1].plot_surface(re_mesh, t_mesh, pred.T, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
    ax[0].set_xlabel(r"$Re$")
    ax[0].set_ylabel(r"time")
    ax[1].set_xlabel(r"$Re$")
    ax[1].set_ylabel(r"time")
    ax[0].set_title(r"VMS-ROM Closure term")
    ax[1].set_title(r"NN-ROM Closure term")
    plt.colorbar(surf, shrink=0.5)
    plt.colorbar(surf_comp, shrink=0.5)
    plt.savefig(output_path + "nn_rom_tau_" + str(i) + ".png", dpi=300)

    model_out_reshaped = model_out.reshape((num_Re, num_t), order="F")
    np.save(output_path + "model_pred_" + str(i) + ".npy", model_out_reshaped)
    gs.best_estimator_.save_params(output_path + "model_param_" + str(i) + ".pkl")

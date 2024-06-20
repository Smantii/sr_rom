
from sr_rom.data.data import process_data, split_data, smooth_data
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import cm


class CustomDataset(Dataset):
    def __init__(self, Re, targets):
        self.Re = Re
        self.targets = targets

    def __getitem__(self, idx):
        Re_idx = self.Re[idx]
        target_idx = self.targets[idx]
        return Re_idx, target_idx

    def __len__(self):
        return len(self.targets)


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.flatten = nn.Flatten()
        # define hidden_layers
        hidden_layers = [nn.Linear(6, hidden_units[0]), nn.ReLU()]
        for i in range(1, len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        # append last layer
        hidden_layers.append(nn.Linear(hidden_units[-1], 1))

        # nn stack
        self.linear_relu_stack = nn.Sequential(*hidden_layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


output_path = sys.argv[1]

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.", flush=True)
else:
    print("No GPU available. Training will run on CPU.", flush=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, flush=True)

Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")
A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=3, num_smoothing=2, r=5)
train_data, val_data, train_val_data, test_data = split_data(
    Re, A_conv, B_conv, tau_conv, a_FOM, 0.2)

num_Re = len(Re)
num_t = tau.shape[1]

t = np.linspace(500, 520, 2001)

Re_grid, t_grid = np.meshgrid(Re, t)
X = np.zeros((61*2001, 6))
X[:, 0] = Re_grid.flatten()
for i in range(5):
    X[:, i+1] = a_FOM[:, :, i].flatten('F')

train_val = np.zeros((len(train_val_data.X)*2001, 6))
test = np.zeros((len(test_data.X)*2001, 6))
for i in range(2001):
    train_val[len(train_val_data.X)*i:len(train_val_data.X)
              * (i+1)] = X[train_val_data.y["idx"] + 61*i]
    test[len(test_data.X)*i:len(test_data.X)*(i+1)] = X[test_data.y["idx"] + 61*i]

X_train_val = train_val
y_train_val = train_val_data.y["tau"][:, :, 0].flatten('F')
X_test = test
y_test = test_data.y["tau"][:, :, 0].flatten('F')

# Standardization
mean_std_train_Re = [np.mean(X_train_val, axis=0), np.std(X_train_val, axis=0)]
mean_std_train_comp = [np.mean(y_train_val, axis=0), np.std(y_train_val, axis=0)]
train_Re_norm = (X_train_val - mean_std_train_Re[0])/mean_std_train_Re[1]
train_comp_norm = (y_train_val - mean_std_train_comp[0]) / \
    mean_std_train_comp[1]
test_Re_norm = (X_test - mean_std_train_Re[0])/mean_std_train_Re[1]
test_comp_norm = (y_test - mean_std_train_comp[0])/mean_std_train_comp[1]

train_Re_norm = torch.from_numpy(train_Re_norm).to(torch.float32)
train_comp_norm = torch.from_numpy(train_comp_norm.reshape(-1, 1)).to(torch.float32)
test_Re_norm = torch.from_numpy(test_Re_norm).to(torch.float32)
test_comp_norm = torch.from_numpy(test_comp_norm.reshape(-1, 1)).to(torch.float32)

model = NeuralNetRegressor(module=NeuralNetwork, batch_size=512, verbose=0,
                           optimizer=torch.optim.Adam, max_epochs=50,
                           train_split=None, device="cuda")

params = {
    'lr': [1e-4, 1e-3],
    'optimizer__weight_decay': [1e-5, 1e-4],
    'module__hidden_units': [[64, 128, 256, 512, 256, 128, 64],
                             [128, 256, 512, 1024, 512, 256, 128]]
}

tic = time.time()
gs = GridSearchCV(model, params, cv=3, verbose=3,
                  scoring="neg_mean_squared_error", refit=True, n_jobs=3, return_train_score=True)
gs.fit(train_Re_norm, train_comp_norm)
print(f"Completed in {time.time() - tic}", flush=True)

r2_train = 1 + gs.score(train_Re_norm, train_comp_norm) / \
    torch.mean((train_comp_norm - torch.mean(train_comp_norm))**2)
r2_test = 1 + gs.score(test_Re_norm, test_comp_norm) / \
    torch.mean((test_comp_norm - torch.mean(test_comp_norm))**2)

print(f"The R^2 in the training set is {r2_train.item()}", flush=True)
print(f"The R^2 in the test set is {r2_test.item()}", flush=True)

# plot prediction
X_scaled = (X - mean_std_train_Re[0])/mean_std_train_Re[1]

model_out = gs.predict(torch.from_numpy(X_scaled).to(torch.float32).cuda())

# rescale out back
model_out_np = model_out*mean_std_train_comp[1] + mean_std_train_comp[0]
pred = model_out_np.reshape((num_Re, num_t), order="F")

re_mesh, t_mesh = np.meshgrid(Re, t)

fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={
                       "projection": "3d"},  figsize=(20, 10))


# Plot the surface.
surf = ax[0].plot_surface(re_mesh, t_mesh, tau_conv[:, :, 0].T, cmap=cm.coolwarm,
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
plt.savefig(output_path + "nn_rom_tau_0.png", dpi=300)

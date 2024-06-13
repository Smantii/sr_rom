import os
from sr_rom.data.data import process_data, split_data, smooth_data
from torch.utils.data import Dataset
from torch import nn
import torch
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import sys


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
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def save_results(reg, X_train_val, y_train_val, X_test, y_test,
                 mean_std_train_Re, mean_std_train_comp, prb_name, ylabel):

    train_val_neg_mse = reg.score(X_train_val, y_train_val)
    test_neg_mse = reg.score(X_test, y_test)
    # compute r2
    train_val_score = 1 + train_val_neg_mse / \
        torch.mean((y_train_val - torch.mean(y_train_val))**2)
    test_score = 1 + test_neg_mse / \
        torch.mean((y_test - torch.mean(y_test))**2)

    # reconstruct the full dataset (concatenate and sort to account for order)
    Re_data_norm = torch.sort(torch.concatenate((X_train_val, X_test)),
                              axis=0)[0].view(-1, 1).detach().numpy()
    prediction_norm = reg.predict(Re_data_norm)

    # revert data scaling
    Re_data = mean_std_train_Re[1]*Re_data_norm + mean_std_train_Re[0]
    X_train_val = mean_std_train_Re[1]*X_train_val + mean_std_train_Re[0]
    X_test = mean_std_train_Re[1]*X_test + mean_std_train_Re[0]
    prediction = mean_std_train_comp[1]*prediction_norm + mean_std_train_comp[0]
    y_train_val = mean_std_train_comp[1]*y_train_val + mean_std_train_comp[0]
    y_test = mean_std_train_comp[1]*y_test + mean_std_train_comp[0]

    with open("scores.txt", "a") as text_file:
        text_file.write(prb_name + " " + str(train_val_score.item()) +
                        " " + str(test_score.item()) + "\n")

    np.save(prb_name + "_pred", prediction)

    plt.scatter(X_train_val, y_train_val,
                c="#b2df8a", marker=".", label="Training data")
    plt.scatter(X_test, y_test,
                c="#b2df8a", marker="*", label="Test data")
    plt.scatter(Re_data, prediction, c="#1f78b4", marker='x',
                label="Best sol", linewidths=0.5)
    plt.xlabel(r"$Re$")
    plt.ylabel(ylabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    plt.savefig(prb_name, dpi=300)

    plt.clf()

    Re_full_norm = np.linspace(Re_data_norm[0], Re_data_norm[-1], 1001).reshape(-1, 1)
    prediction_norm = reg.predict(Re_full_norm)

    # revert data scaling
    Re_full = mean_std_train_Re[1]*Re_full_norm + mean_std_train_Re[0]
    prediction = mean_std_train_comp[1]*prediction_norm + mean_std_train_comp[0]

    plt.scatter(X_train_val, y_train_val,
                c="#b2df8a", marker=".", label="Training data")
    plt.scatter(X_test, y_test,
                c="#b2df8a", marker="*", label="Test data")
    plt.plot(Re_full, prediction, c="#1f78b4",
             label="Best sol", linewidth=0.5)
    plt.xlabel(r"$Re$")
    plt.ylabel(ylabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    plt.savefig(prb_name + "_cont", dpi=300)
    print(f"{prb_name} learned!", flush=True)

    plt.clf()


def nn_rom(train_val_data, test_data, output_path):
    os.chdir(output_path)

    with open("scores.txt", "a") as text_file:
        text_file.write("Name" + " " + "R^2_train" + " " + "R^2_test\n")

    max_epochs = 1000

    params = {
        'lr': [1e-4, 1e-3, 1e-2],
        'optimizer__weight_decay': [1e-5, 1e-4],
    }

    print("Started training procedure for A", flush=True)
    # training procedure for A
    for i in range(5):
        for j in range(5):
            train_Re = torch.from_numpy(train_val_data.X).view(-1, 1).to(torch.float32)
            train_comp = torch.from_numpy(
                train_val_data.y['A'][:, i, j]).view(-1, 1).to(torch.float32)
            test_Re = torch.from_numpy(test_data.X).view(-1, 1).to(torch.float32)
            test_comp = torch.from_numpy(
                test_data.y['A'][:, i, j]).view(-1, 1).to(torch.float32)

            # Standardization
            mean_std_train_Re = [torch.mean(train_Re), torch.std(train_Re)]
            mean_std_train_comp = [torch.mean(train_comp), torch.std(train_comp)]
            train_Re_norm = (train_Re - mean_std_train_Re[0])/mean_std_train_Re[1]
            train_comp_norm = (train_comp - mean_std_train_comp[0]) / \
                mean_std_train_comp[1]
            test_Re_norm = (test_Re - mean_std_train_Re[0])/mean_std_train_Re[1]
            test_comp_norm = (test_comp - mean_std_train_comp[0])/mean_std_train_comp[1]

            model = NeuralNetRegressor(module=NeuralNetwork, batch_size=-1, verbose=0,
                                       optimizer=torch.optim.Adam, max_epochs=max_epochs, train_split=None)

            gs = GridSearchCV(model, params, cv=5, verbose=3,
                              scoring="neg_mean_squared_error", refit=True, n_jobs=-1)
            gs.fit(train_Re_norm, train_comp_norm)

            save_results(gs, train_Re_norm, train_comp_norm, test_Re_norm, test_comp_norm,
                         mean_std_train_Re, mean_std_train_comp,
                         "A_" + str(i) + str(j), r"$A_{ij}$")


if __name__ == "__main__":
    # load and process data
    Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")
    A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=3, num_smoothing=2, r=5)

    _, _, train_val_data, test_data = split_data(
        Re, A_conv, B_conv, tau_conv, a_FOM, test_size=0.4)

    output_path = sys.argv[1]
    nn_rom(train_val_data, test_data, output_path)

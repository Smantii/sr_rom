import os
from sr_rom.data.data import process_data, split_data, smooth_data
from torch.utils.data import Dataset, DataLoader
from torch import tensor, nn
import torch
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV


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


def nn_rom(train_val_data, test_data, output_path):
    os.chdir(output_path)

    with open("scores.txt", "a") as text_file:
        text_file.write("Name" + " " + "R^2_train" + " " + "R^2_test\n")

    max_epochs = 1000

    params = {
        'lr': [1e-4, 1e-3, 1e-2],
        'optimizer__weight_decay': [1e-5, 1e-4, 1e-3],
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
            train_Re_norm = (train_Re - torch.mean(train_Re))/torch.std(train_Re)
            train_comp_norm = (train_comp - torch.mean(train_comp)) / \
                torch.std(train_comp)
            test_Re_norm = (test_Re - torch.mean(train_Re))/torch.std(train_Re)
            test_comp_norm = (test_comp - torch.mean(train_comp))/torch.std(train_comp)

            # wrap datasets
            training = CustomDataset(train_Re_norm, train_comp_norm)
            test = CustomDataset(test_Re_norm, test_comp_norm)

            train_dataloader = DataLoader(
                training, batch_size=len(train_Re_norm), shuffle=False)
            test_dataloader = DataLoader(
                test, batch_size=len(test_Re_norm), shuffle=False)

            model = NeuralNetRegressor(module=NeuralNetwork, batch_size=-1, verbose=0,
                                       optimizer=torch.optim.Adam, max_epochs=1000, train_split=None)

            gs = GridSearchCV(model, params, cv=5, verbose=3,
                              scoring="neg_mean_squared_error", refit=True, n_jobs=6)
            gs.fit(train_Re_norm, train_comp_norm)


if __name__ == "__main__":
    # load and process data
    Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")
    A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=3, num_smoothing=2, r=5)

    _, _, train_val_data, test_data = split_data(
        Re, A_conv, B_conv, tau_conv, a_FOM, test_size=0.4)

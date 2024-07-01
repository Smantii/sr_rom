import numpy as np
import numpy.typing as npt
from typing import Tuple
from alpine.data import Dataset
from sklearn.model_selection import train_test_split as ttsplit
import os
import math


def toy_data(k: float, r: int) -> Tuple[npt.NDArray, npt.NDArray]:
    A = np.zeros((r, r))
    B = np.zeros((r, r, r))
    # fill A
    for j in np.arange(r):
        # fill even rows
        A[::2, j] = np.sum((-k)**(np.arange(j+2)))
        # fill odd rows
        A[1::2, j] = -np.sum((-k)**(np.arange(j+2)))
    # fill B
    for l in np.arange(r):
        B[:, :, l] = (0.5)**l*A
    return {'A': A, 'B': B}


def generate_toy_data(r):
    k_list = list(range(1, 11))
    A_B_list = []
    for k in k_list:
        A_B_k = toy_data(k, r)
        A_B_list.append(A_B_k)

    return k_list, A_B_list


def split_data(Re, A, B, tau, a_FOM, X, test_size=0.2, shuffle_test=False):
    num_data = len(Re)
    # num_test = round(test_size*num_data)
    # half_num_test = int(num_test/2)
    # center = int((num_data - 1)/2)

    # idx_test = np.arange(center - half_num_test, center + half_num_test)
    # idx_train_val = np.concatenate(
    #    (np.arange(center - half_num_test), np.arange(center + half_num_test, num_data)))
    # Re_train_val = Re[idx_train_val]
    # Re_test = Re[idx_test]

    # FIXME: fix hardcoded numbers!

    Re_train_val, Re_test, idx_train_val, idx_test = ttsplit(
        Re, np.arange(num_data), test_size=test_size, random_state=42, shuffle=shuffle_test)

    X_train_val = np.zeros((len(Re_train_val)*2001, 6))
    X_test = np.zeros((len(Re_test)*2001, 6))
    for i in range(2001):
        X_train_val[len(Re_train_val)*i:len(Re_train_val)
                    * (i+1)] = X[idx_train_val + 61*i]
        X_test[len(Re_test)*i:len(Re_test)*(i+1)] = X[idx_test + 61*i]

    y_train_val = tau[idx_train_val]
    y_test = tau[idx_test]

    # FIXME: adapt to this part to the new dataset
    Re_train, Re_val, idx_train,  idx_val = ttsplit(
        Re_train_val, idx_train_val, test_size=2/8, random_state=42, shuffle=True)

    A_train = A[idx_train]
    A_train_val = A[idx_train_val]
    A_val = A[idx_val]
    A_test = A[idx_test]

    B_train = B[idx_train]
    B_train_val = B[idx_train_val]
    B_val = B[idx_val]
    B_test = B[idx_test]

    tau_train = tau[idx_train]
    tau_train_val = tau[idx_train_val]
    tau_val = tau[idx_val]
    tau_test = tau[idx_test]

    a_FOM_train = a_FOM[idx_train]
    a_FOM_train_val = a_FOM[idx_train_val]
    a_FOM_val = a_FOM[idx_val]
    a_FOM_test = a_FOM[idx_test]

    data_train = {'A': A_train, 'B': B_train, 'tau': tau_train,
                  'a_FOM': a_FOM_train, 'idx': idx_train}
    data_train_val = {'A': A_train_val, 'B': B_train_val,
                      'tau': tau_train_val, 'a_FOM': a_FOM_train_val,
                      'idx': idx_train_val}
    data_val = {'A': A_val, 'B': B_val, 'tau': tau_val,
                'a_FOM': a_FOM_val, 'idx': idx_val}
    data_test = {'A': A_test, 'B': B_test, 'tau': tau_test,
                 'a_FOM': a_FOM_test, 'idx': idx_test}

    train_data = Dataset("Re_data", Re_train, data_train)
    train_val_data = Dataset("Re_data", Re_train_val, data_train_val)
    val_data = Dataset("Re_data", Re_val, data_val)
    test_data = Dataset("Re_data", Re_test, data_test)

    return train_data, val_data, train_val_data, test_data


def process_data(r: int, bench_name: str):
    data_path = os.path.dirname(os.path.realpath(__file__))
    bench_path = os.path.join(data_path, bench_name)

    dir_list = sorted(os.listdir(bench_path))
    num_Re = len(dir_list)
    num_t = 2001

    Re = np.zeros(num_Re)
    t = np.linspace(500, 520, num_t)

    A = np.zeros((num_Re, r, r))
    B = np.zeros((num_Re, r, r, r))
    tau = np.zeros((num_Re, num_t, r))
    a_FOM = np.zeros((num_Re, num_t, r))

    for i, directory in enumerate(dir_list):
        directory_path = os.path.join(bench_path, directory)
        curr_Re = float(directory.replace("Re", ""))
        curr_tau = np.loadtxt(directory_path+"/vmsrom_clousre_N5",
                              delimiter=',', usecols=range(r))
        curr_A = np.loadtxt(directory_path+"/tildeA_N5",
                            delimiter=',', usecols=range(r))
        curr_B = np.loadtxt(directory_path+"/tildeB_N5",
                            delimiter=',', usecols=range(r**2)).reshape((r, r, r))
        uk = np.loadtxt(directory_path+"/uk", delimiter=',')
        curr_a_FOM = uk.reshape((num_t, 41))[:, 1:(r+1)]

        Re[i] = curr_Re
        A[i, :, :] = curr_A
        B[i, :, :, :] = curr_B
        tau[i, :, :] = curr_tau
        a_FOM[i, :, :] = curr_a_FOM

    Re_grid, t_grid = np.meshgrid(Re, t)

    # fill matrix of data
    X = np.zeros((num_Re*num_t, r+1))
    X[:, 0] = Re_grid.flatten()
    for i in range(5):
        X[:, i+1] = a_FOM[:, :, i].flatten('F')

    return Re, A, B, tau, a_FOM, X


def smooth_data(A, B, tau, w, num_smoothing, r):
    A_conv = A.copy()
    B_conv = B.copy()

    tau_conv = np.zeros_like(tau)

    for _ in range(num_smoothing):

        for i in range(r):
            for j in range(r):
                A_conv[:, i, j] = np.convolve(A[:, i, j], np.ones(w), 'same') / w

        for i in range(r):
            for j in range(r):
                for k in range(r):
                    B_conv[:, i, j, k] = np.convolve(
                        B[:, i, j, k], np.ones(w), 'same') / w

        A = A_conv
        B = B_conv

        for i in range(r):
            for j in range(2001):
                tau_conv[:, j, i] = np.convolve(tau[:, j, i], np.ones(w), 'same') / w

        tau = tau_conv

    return A_conv, B_conv, tau_conv


if __name__ == "__main__":
    # k_array, A_B_list = generate_toy_data(3)
    # data = split_data(k_array, A_B_list)
    process_data(5, "2dcyl/Re200_300")

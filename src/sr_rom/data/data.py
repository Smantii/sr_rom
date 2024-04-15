import numpy as np
import numpy.typing as npt
from typing import Tuple
from alpine.data import Dataset
from sklearn.model_selection import train_test_split as ttsplit
import os


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
    k_array = np.arange(1, 11)
    A_B_list = []
    for k in k_array:
        A_B_k = toy_data(k, r)
        A_B_list.append(A_B_k)

    return k_array, A_B_list


def split_data(k_array, A_B_list):
    k_train_val, k_test, A_B_train_val, A_B_test = ttsplit(
        k_array, A_B_list, test_size=0.2, random_state=42, shuffle=True)
    k_train, k_val, A_B_train,  A_B_val = ttsplit(
        k_train_val, A_B_train_val, test_size=3/8, random_state=42, shuffle=True)

    train_data = Dataset("k_A_B", k_train, A_B_train)
    val_data = Dataset("k_A_B", k_val, A_B_val)
    test_data = Dataset("k_A_B", k_test, A_B_test)

    return train_data, val_data, test_data


def process_data(r: int, bench_name: str) -> Tuple[Dataset, Dataset, Dataset]:
    data_path = os.path.dirname(os.path.realpath(__file__))
    bench_path = os.path.join(data_path, bench_name)
    k_list = []
    A_B_list = []
    for directory in os.listdir(bench_path):
        directory_path = os.path.join(bench_path, directory)
        k = float(directory.replace("Re", ""))
        A = np.loadtxt(directory_path+"/tildeA_N5", delimiter=',', usecols=range(r))
        B = np.loadtxt(directory_path+"/tildeB_N5", delimiter=',', usecols=range(r))
        k_list.append(k)
        A_B_list.append({'A': A, 'B': B})
    return np.array(k_list), A_B_list


if __name__ == "__main__":
    # k_array, A_B_list = generate_toy_data(3)
    # data = split_data(k_array, A_B_list)
    process_data(5, "2dcyl")

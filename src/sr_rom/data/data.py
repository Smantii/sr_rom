import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from alpine.data import Dataset
from sklearn.model_selection import train_test_split as ttsplit
import os


def split_data(Re, A, B, tau, a_FOM, X, X_sampled, residual, test_size=0.2, shuffle_test=False):
    num_data = len(Re)

    Re_train_val, Re_test, idx_train_val, idx_test = ttsplit(
        Re, np.arange(num_data), test_size=test_size, random_state=42, shuffle=shuffle_test)

    # validation percentage = 10% of the entire data
    Re_train, Re_val, idx_train,  idx_val = ttsplit(
        Re_train_val, idx_train_val, test_size=1/(10*(1 - test_size)), random_state=42, shuffle=True)

    num_Re = len(Re)

    num_t_points_sampled = int(X_sampled.shape[0]/num_Re)

    X_train = np.zeros((len(Re_train)*2001, X.shape[1]))
    X_val = np.zeros((len(Re_val)*2001, X.shape[1]))
    X_train_val = np.zeros((len(Re_train_val)*2001, X.shape[1]))
    X_test = np.zeros((len(Re_test)*2001, X.shape[1]))
    X_train_sampled = np.zeros((len(Re_train)*num_t_points_sampled, X.shape[1]))
    X_val_sampled = np.zeros((len(Re_val)*num_t_points_sampled, X.shape[1]))
    X_train_val_sampled = np.zeros((len(Re_train_val)*num_t_points_sampled, X.shape[1]))
    X_test_sampled = np.zeros((len(Re_test)*num_t_points_sampled, X.shape[1]))

    # fill X_train_val and X_test
    for i in range(2001):
        X_train[len(Re_train)*i:len(Re_train)
                * (i+1)] = X[idx_train + num_Re*i]
        X_val[len(Re_val)*i:len(Re_val)
              * (i+1)] = X[idx_val + num_Re*i]
        X_train_val[len(Re_train_val)*i:len(Re_train_val)
                    * (i+1)] = X[idx_train_val + num_Re*i]
        X_test[len(Re_test)*i:len(Re_test)*(i+1)] = X[idx_test + num_Re*i]

    # fill X_train_val and X_test sampled
    for i in range(num_t_points_sampled):
        X_train_sampled[len(Re_train)*i:len(Re_train)
                        * (i+1)] = X_sampled[idx_train + num_Re*i]
        X_val_sampled[len(Re_val)*i:len(Re_val)
                      * (i+1)] = X_sampled[idx_val + num_Re*i]
        X_train_val_sampled[len(Re_train_val)*i:len(Re_train_val)
                            * (i+1)] = X_sampled[idx_train_val + num_Re*i]
        X_test_sampled[len(Re_test)*i:len(Re_test)
                       * (i+1)] = X_sampled[idx_test + num_Re*i]

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

    residual_train = residual[idx_train]
    residual_val = residual[idx_val]
    residual_train_val = residual[idx_train_val]
    residual_test = residual[idx_test]

    # FIXME: since we are not doing holdout, this are not updated
    data_train = {'A': A_train, 'B': B_train, 'tau': tau_train,
                  'a_FOM': a_FOM_train, 'idx': idx_train, "X": X_train,
                  "X_sampled": X_train_sampled, "residual": residual_train}
    data_val = {'A': A_val, 'B': B_val, 'tau': tau_val,
                'a_FOM': a_FOM_val, 'idx': idx_val, "X": X_val,
                "X_sampled": X_val_sampled, "residual": residual_val}

    data_train_val = {'A': A_train_val, 'B': B_train_val,
                      'tau': tau_train_val, 'a_FOM': a_FOM_train_val,
                      'idx': idx_train_val, "X": X_train_val,
                      "X_sampled": X_train_val_sampled, "residual": residual_train_val}

    data_test = {'A': A_test, 'B': B_test, 'tau': tau_test,
                 'a_FOM': a_FOM_test, 'idx': idx_test, "X": X_test, "X_sampled": X_test_sampled, "residual": residual_test}

    train_data = Dataset("Re_data", Re_train, data_train)
    train_val_data = Dataset("Re_data", Re_train_val, data_train_val)
    val_data = Dataset("Re_data", Re_val, data_val)
    test_data = Dataset("Re_data", Re_test, data_test)

    return train_data, val_data, train_val_data, test_data


def process_data(r: int, bench_name: str, Re_list: List | str, num_t: List):
    """Function that transform ROM data in numpy arrays.

    Args:
        r: number of modes.
        bench_name: name of the directory where the data are located
        Re_list: list of Reynolds subdirectories to be used to build the
            ROM numpy arrays.
        t: numpy array equal to np.linspace(t_0, t_1, num_t_points), where
            [t_0,t_1] is the interval in which the ROM operators are defined.
        t_sample: time sampling index.

    Returns:
        the Reynolds, tau and a_FOM numpy arrays (first three numpy arrays); one
        numpy matrix of shape (num_Re, num_t) rows and r+1 columns, in which each row
        is the value of Re (first column) and a_FOM in such Re and given time.
        The last numpy array is built on the same fashion of X but has dimension 
        (num_Re, num_t_sampled).
    """

    data_path = os.path.dirname(os.path.realpath(__file__))
    bench_path = os.path.join(data_path, bench_name)

    dir_list = sorted(os.listdir(bench_path))
    Re, tau, a_FOM = [], [], []

    for directory in dir_list:
        directory_path = os.path.join(bench_path, directory)
        curr_Re = float(directory.replace("Re", ""))
        uk = np.loadtxt(directory_path+"/uk", delimiter=',')
        num_basis_functions = int(len(uk)/num_t)
        curr_a_FOM = uk.reshape((num_t, num_basis_functions))[:, 1:(r+1)]

        if curr_Re in Re_list:
            Re.append(curr_Re)
            a_FOM.append(curr_a_FOM)
            curr_tau = np.loadtxt(directory_path+"/vmsrom_clousre_N"+str(r)+"_all.txt",
                                  delimiter=',', usecols=range(r))
            tau.append(curr_tau)

    Re = np.array(Re)
    tau = np.array(tau)
    a_FOM = np.array(a_FOM)

    return Re, tau, a_FOM

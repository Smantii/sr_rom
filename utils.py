import numpy as np
import os
import re

def find_snapshot_rank(path):
    # Read the file
    path_gu = os.path.join(path, "gu")
    gu = np.loadtxt(path_gu)

    # Reshape the array into a square matrix
    n = int(np.sqrt(len(gu)))
    gu = gu.reshape((n, n))

    # Compute the rank of the matrix
    R = np.linalg.matrix_rank(gu)

    return R

def load_rom_ops(path):

    mb = np.loadtxt(path + "/nb")
    mb = int(mb)  # Ensure mb is an integer

    # load stiffness matrix
    a0_full = np.loadtxt(path + "/au")
    a0_full = a0_full.reshape((mb + 1, mb + 1), order='F')

    # load mass matrix
    b0_full = np.loadtxt(path + "/bu")
    b0_full = b0_full.reshape((mb + 1, mb + 1), order='F')

    # load advection tensor
    cu_full = np.loadtxt(path + "/cu")
    cu_full = cu_full.reshape((mb, mb + 1, mb + 1), order='F')

    return a0_full, b0_full, cu_full, mb

def get_rom_ops_r_dim(a0_full, b0_full, cu_full, nb):
    index  = np.arange(nb + 1)
    index1 = np.arange(nb)
    index2 = np.arange(1, nb + 1)

    au0 = a0_full[0:nb+1, 0:nb+1]

    bu0 = b0_full[0:nb+1, 0:nb+1]

    cutmp = cu_full[0:nb, 0:nb+1, 0:nb+1]
    cutmp1 = cu_full[0:nb, 1:nb+1, 1:nb+1]
    cu = cutmp1.reshape((nb * nb, nb))

    au = au0[1:, 1:]
    bu = bu0[1:, 1:]

    return au0, bu0, au, bu, cu, cutmp, cutmp1

def load_initial_condition(path, nb):
    # load initial condition
    index = np.arange(nb + 1)
    u0_full = np.loadtxt(path + "/u0")
    u0 = u0_full[index]

    return u0

def load_coefficients_training_data(path, mb):
    # load ns
    ns = int(np.loadtxt(path + "/ns"))

    # load uk
    uk = np.loadtxt(path + "/uk")
    uk = uk.reshape((mb + 1, ns), order='F').T

    return uk, ns

def constructMatrixAndTensor(directory, index):
    # Get list of all *_pred.npy files in the specified directory
    files = [f for f in os.listdir(directory) if f.endswith('_pred.npy')]
    
    # Initialize dictionaries for A matrix and B tensor components
    A_components = {}
    B_components = {}

    # Regular expression patterns for file matching
    pattern_A = r'A_(\d{2})_pred\.npy'
    pattern_B = r'B_(\d{3})_pred\.npy'

    # Load A matrix components
    for file in files:
        matches_A = re.match(pattern_A, file)
        matches_B = re.match(pattern_B, file)
        
        if matches_A:
            component = matches_A.group(1)
            data = np.load(os.path.join(directory, file))  # Include directory path
            A_components[component] = data[index]
        
        if matches_B:
            component = matches_B.group(1)
            data = np.load(os.path.join(directory, file))  # Include directory path
            B_components[component] = data[index]

    # Construct the A matrix
    A = np.zeros((5, 5))  # Assuming A is a 5x5 matrix
    for i in range(5):
        for j in range(5):
            component = '{:02d}'.format(i * 10 + j)
            if component in A_components:
                A[i, j] = A_components[component]
            else:
                print(f"Missing component A_{component}")

    # Construct the B tensor
    B = np.zeros((5, 5, 5))  # Assuming B is a 5x5x5 tensor
    for i in range(5):
        for j in range(5):
            for k in range(5):
                component = '{:03d}'.format(i * 100 + j * 10 + k)
                if component in B_components:
                    B[i, j, k] = B_components[component]
                else:
                    print(f"Missing component B_{component}")

    # Display the A matrix and B tensor
    print('A matrix:')
    print(A)
    print('B tensor:')
    print(B)

    return A, B

def two_scale_quadratic_ansatz_gen(tensor_r, tensor_d, a_snap_r, a_snap_d):
    rhs_mat = construct_closure(tensor_r, tensor_d, a_snap_r, a_snap_d)
    lhs_mat = construct_approximated_g(a_snap_r)
    
    return rhs_mat, lhs_mat

def construct_closure(tensor_r, tensor_d, a_snap_r, a_snap_d):
    # Assemble the vector f in the ddf ROM
    # see the paper for details

    M, r = a_snap_r.shape
    _, d = a_snap_d.shape

    tau_r_ = np.zeros((r, M))
    
    for j in range(M):
        tmp = np.reshape(np.dot(tensor_d, a_snap_d[j, :]), (d, d)).dot(a_snap_d[j, :])
        tmp1 = np.reshape(np.dot(tensor_r, a_snap_r[j, :]), (r, r)).dot(a_snap_r[j, :])
        tau_r_[:, j] = tmp[:r] - tmp1
    
    tau_r = tau_r_.T
    
    return tau_r

def construct_approximated_g(a_snap):
    """
    Assemble the matrix E in the ddf ROM.

    Parameters:
    a_snap : numpy.ndarray
        The matrix of size M*r.

    Returns:
    numpy.ndarray
        The matrix E of size M*(r + r^2 + r^3).
    """
    
    M, r = a_snap.shape
    ansatz_option = 2
    
    if ansatz_option == 1:
        # Linear ansatz
        matrix_e = np.zeros((M, r))
        for j in range(M):
            matrix_e[j, :] = a_snap[j, :]
    
    elif ansatz_option == 2:
        # Quadratic ansatz
        matrix_e = np.zeros((M, r + r**2))
        for j in range(M):
            aj = a_snap[j, :]
            A1 = aj
            B1 = np.outer(aj, aj).reshape(-1)
            matrix_e[j, :] = np.concatenate((A1, B1))
    
    elif ansatz_option == 3:
        # Cubic ansatz
        matrix_e = np.zeros((M, r + r**2 + r**3))
        for j in range(M):
            aj = a_snap[j, :]
            A1 = aj
            B1 = np.outer(aj, aj).reshape(-1)
            C1 = np.kron(aj, np.kron(aj, aj))
            matrix_e[j, :] = np.concatenate((A1, B1, C1))
    
    return matrix_e

def solve_g_using_tsvd(U, S, V, b_rhs, r, choose_rank):
    # Extract necessary components from SVD
    Ua = U[:, :choose_rank]
    Va = V[:, :choose_rank]
    Sa = np.diag(S[:choose_rank])

    # Compute the inverse of the eigenvalues of Sa
    eigen_Sr = np.diag(Sa)
    S_diagi = 1. / eigen_Sr
    S_inv = np.diag(S_diagi)

    # Calculate B_m using the selected rank
    B_m = Va @ S_inv @ Ua.T @ b_rhs

    # Extract TildeA and TildeB from B_m
    TildeA = B_m[:r, :].T
    TildeB = B_m[r:, :].T

    # Reshape TildeB to a 3D tensor
    TildeB = np.reshape(TildeB, (r, r, r))

    return TildeA, TildeB
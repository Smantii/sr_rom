import numpy as np
from typing import Optional, List
from functools import partial
import torch
import torch.jit as jit
from skorch import NeuralNetRegressor
from skorch.utils import to_device
import ast
import joblib
from sr_rom.roms import *


def eval_tau_with_saved_models(nb, pclosure, method):
    # setup for closure function
    # Load mean and standard deviation files
    mean_std_X_train_file = f"{pclosure}mean_std_X_train.npy"

    # Load models
    if method == "SR":
        # model_files = reg_list
        model_files = f"{pclosure}models.txt"

    elif method == "NN":
        with open(f"{pclosure}params.txt", "r") as file:
            file = file.readlines()
        model_files = []
        for i, best_params in enumerate(file):
            curr_model = {}
            best_params_dict = ast.literal_eval(best_params)
            curr_model["model_path"] = f"{pclosure}model_param_" + str(i)+".pkl"
            curr_model["module__hidden_units"] = best_params_dict["module__hidden_units"]
            curr_model['module__dropout_rate'] = best_params_dict['module__dropout_rate']
            curr_model['module__r'] = best_params_dict['module__r']
            model_files.append(curr_model)

    elif method == "LI":
        model_files = f"{pclosure}/tau_interp.npy"

    elif method == "LR":
        model_files = [f"{pclosure}model_" + str(i) + ".pkl" for i in range(nb)]

    # Create the process_input_and_calculate_tau function with cached mean and std values
    mean_inputs, std_inputs, best_models = load_inputs_models(
        model_files, mean_std_X_train_file, method)

    # define eval tau function
    eval_tau = partial(compute_tau_with_saved_models,
                       mean_inputs=mean_inputs, std_inputs=std_inputs,
                       best_models=best_models, method=method)
    return eval_tau


def eval_tau_with_curr_models(pclosure, models):
    # Load mean and standard deviation files
    mean_std_X_train_file = f"{pclosure}mean_std_X_train.npy"
    # Load mean and standard deviation values for inputs and outputs
    mean_std_X_train = np.load(mean_std_X_train_file)
    mean_inputs = mean_std_X_train[0]
    std_inputs = mean_std_X_train[1]

    # define eval tau function
    eval_tau = partial(compute_tau_with_curr_models,
                       mean_inputs=mean_inputs, std_inputs=std_inputs,
                       best_models=models)
    return eval_tau


def vmsrom_solver_wclosure(au, bu, cu, a0, u0, nsteps, iostep, nu, Dt,
                           nb, method, pclosure, reg_rom=None):

    rhs = np.zeros((nb, 1))
    ext = np.zeros((nb, 3))
    hufac = None

    alphas = np.zeros((3, 3))
    betas = np.zeros((4, 3))

    alphas = np.array([[1., 2., 3],
                      [0., -1., -3.],
                      [0., 0., 1.]])

    betas = np.array([[1., 1.5, 11/6],
                     [-1., -2., -3.],
                     [0., 0.5, 1.5],
                     [0., 0., -1/3]])

    ucoef = np.zeros(((nsteps // iostep) + 1, nb + 1))
    ucoef[0, :] = u0
    u = np.zeros((nb + 1, 3))  # vectors for BDF3/EXT3
    u[:, 0] = u0

    if reg_rom is None:
        eval_tau = eval_tau_with_saved_models(nb, pclosure, method)
    else:
        eval_tau = eval_tau_with_curr_models(pclosure, reg_rom)

    for istep in range(1, nsteps + 1):
        ito = min(istep, 3)
        if istep <= 3:
            hufac = None

        # Compute the right-handed side of the fully discretized system associated to BDFk/EXTk
        rhs = np.zeros((nb, 1))

        ext[:, 2] = ext[:, 1]
        ext[:, 1] = ext[:, 0]
        ext[:, 0] = np.zeros(nb)

        utmp = u
        ext[:, 0] -= np.reshape(cu @ utmp[:, 0], (nb, nb + 1)) @ u[:, 0]
        ext[:, 0] -= nu * a0

        rhs += ext @ alphas[:, ito - 1].reshape(-1, 1)
        rhs -= bu @ (u[1:, :] @ betas[1:, ito - 1]).reshape(-1, 1) / Dt
        rhs -= np.array(eval_tau(u[1:, 0])).reshape(-1, 1)

        if hufac is None:
            h = bu * betas[0, ito - 1] / Dt + au * nu
            hfac = np.linalg.cholesky(h)
        u_new = np.hstack(([1], np.linalg.solve(
            hfac, np.linalg.solve(hfac.T, rhs)).flatten()))

        u[:, 2] = u[:, 1]
        u[:, 1] = u[:, 0]
        u[:, 0] = u_new

        if istep % iostep == 0:
            ucoef[istep // iostep, :] = u[:, 0]
        elif np.isnan(np.sum(u[:, 0])):
            ucoef[(istep // iostep + 1):, :] *= np.nan
            break

    return istep, ucoef


def read_model(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def parse_equation(equation):
    # Replace tau_0, tau_1, etc. with variable names for Python code
    equation = equation.replace('tau_', 'tau[')
    equation = equation.replace(' = ', '] = ')
    equation = equation.replace('^', '**')
    return equation


def calculate_taus_sr(*args, equations):
    # Prepare the context for evaluation
    arg_context = {}
    for i, arg in enumerate(args):
        arg_context['X' + str(i+1)] = arg

    # Include required math functions and placeholder for tau values
    context = arg_context | {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                             'exp': np.exp, 'log': np.log, 'asin': np.arcsin,
                             'acos': np.arccos, 'np': np, 'sqrt': np.sqrt,
                             'tau': [0] * len(equations)}

    # Execute each equation in the context
    for eq in equations:
        exec(eq, context)
    return context['tau']


def process_models_sr(model_file):
    # Read model equations
    equations = read_model(model_file)
    parsed_equations = [parse_equation(eq) for eq in equations]

    return parsed_equations


def process_models_nn(models):
    best_models = []
    for model in models:
        best_model = NeuralNetRegressor(
            module=NeuralNetwork,
            module__hidden_units=model["module__hidden_units"],
            module__dropout_rate=model["module__dropout_rate"],
            module__r=model["module__r"]+1,
            device="cpu")
        best_model.initialize()
        best_model.load_params(f_params=model["model_path"])
        # Ensures model loads on cpu
        best_model.set_params(device='cpu')

        # Moves current model instance to cpu
        to_device(best_model.module_, 'cpu')
        module_ = best_model.module_
        module_.eval()
        module_ = jit.script(module_)
        best_models.append(module_)

    return best_models


def process_models_lr(model_file):
    models = [joblib.load(model) for model in model_file]
    return models


def load_inputs_models(model_file, mean_std_X_train_file, method):
    if method == "SR":
        best_models = process_models_sr(model_file)
        # best_models = model_file
    elif method == "NN":
        best_models = process_models_nn(model_file)
    elif method == "LR":
        best_models = process_models_lr(model_file)

    # Load mean and standard deviation values for inputs and outputs
    mean_std_X_train = np.load(mean_std_X_train_file)
    mean_inputs = mean_std_X_train[0]
    std_inputs = mean_std_X_train[1]

    return mean_inputs, std_inputs, best_models


@torch.inference_mode()
def calculate_taus_nn(adjusted_inputs, best_models):
    tau_values = np.zeros(len(adjusted_inputs))
    adj_inputs_reshaped = torch.from_numpy(
        adjusted_inputs.reshape(-1, 1).T.astype(np.float32))
    with torch.no_grad():
        for i, model in enumerate(best_models):
            model_out = model.forward(adj_inputs_reshaped).flatten()
            tau_values[i] = model_out
    return tau_values


def calculate_taus_lr(adjusted_inputs, best_models):
    tau_values = np.array([model.predict(adjusted_inputs.reshape(1, -1))
                          for model in best_models]).flatten()
    return tau_values


def compute_tau_with_saved_models(input_values, mean_inputs, std_inputs, best_models, method):
    # Adjust input_values using mean and std for inputs
    adjusted_inputs = (input_values - mean_inputs) / std_inputs

    if np.isnan(np.sum(adjusted_inputs)):
        return np.nan*np.zeros_like(adjusted_inputs)

    # Calculate tau values with adj vusted inputs
    if method == "SR":
        tau_values = calculate_taus_sr(*adjusted_inputs, equations=best_models)
        # tau_values = calculate_taus_sr(adjusted_inputs, best_models)
    elif method == "NN":
        tau_values = calculate_taus_nn(adjusted_inputs, best_models)
    elif method == "LR":
        tau_values = calculate_taus_lr(adjusted_inputs, best_models)

    adjusted_tau_values = tau_values

    return adjusted_tau_values


def compute_tau_with_curr_models(input_values, mean_inputs, std_inputs, best_models):
    # Adjust input_values using mean and std for inputs
    adjusted_inputs = (input_values - mean_inputs) / std_inputs

    if np.isnan(np.sum(adjusted_inputs)):
        return np.nan*np.zeros_like(adjusted_inputs)

    # tau_values = np.array([model.predict(adjusted_inputs.reshape(1, -1))
    #                       for model in best_models]).flatten()
    tau_values = np.array(best_models.predict(adjusted_inputs.reshape(1, -1))).flatten()

    return tau_values


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
    au0 = a0_full[0:nb+1, 0:nb+1]

    bu0 = b0_full[0:nb+1, 0:nb+1]

    cutmp = cu_full[0:nb, 0:nb+1, 0:nb+1]
    cutmp1 = cu_full[0:nb, 1:nb+1, 1:nb+1]
    cu = cutmp1.reshape((nb * nb, nb))
    cu_full = cutmp.reshape((nb * (nb+1), nb+1))

    a0 = au0[1:, 0]
    au = au0[1:, 1:]
    bu = bu0[1:, 1:]

    return au0, bu0, au, bu, cu, cutmp, cutmp1, cu_full, a0


def load_initial_condition(path, nb):
    index = np.arange(nb + 1)
    u0_full = np.loadtxt(path + "/u0")
    u0 = u0_full[index]

    return u0


def solve_ode(Re: int, method, sdir, path_to_model, T_final: float = 20, dt: float = 0.001,
              iostep: int = 1, r: int = 5, reg_list: Optional[List] = None):
    """ Main driver to run VMSROM with closure at a given Reynolds number.

    Args:
        Re: Reynolds number
        idx_Re: the index of Re w.r.t. full Reynolds data list (only needed to process lin_int results)
        save_csv: true to save errors in csv files
    """

    # viscosity
    mu = 1. / Re
    # number of time steps
    nsteps = int(T_final / dt)

    # load NekROM operators
    a0_full, b0_full, cu_full, _ = load_rom_ops(sdir)

    # Extract r (mb) dimensional NekROM operators
    _, bu0, au, bu, _, _, _, cufull, a0 = get_rom_ops_r_dim(
        a0_full, b0_full, cu_full, r)
    u0 = load_initial_condition(sdir, r)  # load r-dimensional initial condition

    # Run VMSROM with closure
    _, ucoef = vmsrom_solver_wclosure(
        au, bu, cufull, a0, u0, nsteps, iostep, mu, dt, r, method, path_to_model, reg_list)

    # Compute the energy
    ene_matrix = ucoef @ bu0 @ ucoef.T
    ene = 0.5*np.diag(ene_matrix)

    return ucoef, ene

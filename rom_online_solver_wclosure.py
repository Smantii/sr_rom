import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import math
from torch import nn
from skorch import NeuralNetRegressor
import torch
import time


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_units, dropout_rate=0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        # define hidden_layers
        hidden_layers = [nn.Linear(5, hidden_units[0]), nn.LeakyReLU()]
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


def rom_online_solver_wclosure(a0_full, b0_full, au0, bu0, au, bu, cu, cutmp, u0, uk, nsteps, iostep, mu, Dt, nb, directory, ns, testp=None, ws=None):
    # Initialize error tables
    err_list_l2 = pd.DataFrame()
    err_list_h10 = pd.DataFrame()

    extended_vec = np.zeros_like(uk)
    ucoef = np.zeros(((nsteps // iostep) + 1, nb + 1))
    ucoef[0, :] = u0
    u_n = u0[1:nb + 1]

    # setup for closure function
    pclosure = '/home/smanti/SR-ROM/notebooks'

    '''
    # Load mean and standard deviation files and models
    mean_std_X_train_file = f"{pclosure}/mean_std_X.npy"
    models = []
    mean_std_files = []
    for i in range(5):
        curr_model = {}
        curr_model["model_path"] = f"{pclosure}/model_param_" + str(i)+".pkl"
        models.append(curr_model)
        mean_std_files.append(f"{pclosure}/mean_std_y_" + str(i)+".npy")

    # FIXME: automate this!
    models[0]['module__hidden_units'] = [64, 128, 256, 512, 256, 128, 64]
    models[0]['module__dropout_rate'] = 0.5
    models[1]['module__hidden_units'] = [128, 256, 512, 1024, 512, 256, 128]
    models[1]['module__dropout_rate'] = 0.5
    models[2]['module__hidden_units'] = [128, 256, 512, 1024, 512, 256, 128]
    models[2]['module__dropout_rate'] = 0.3
    models[3]['module__hidden_units'] = [128, 256, 512, 1024, 512, 256, 128]
    models[3]['module__dropout_rate'] = 0.3
    models[4]['module__hidden_units'] = [64, 128, 256, 512, 256, 128, 64]
    models[4]['module__dropout_rate'] = 0.3
    '''
    models = []
    for i in range(5):
        models.append(f"{pclosure}/tau_conv_" + str(i)+".npy")

    # Create the process_input_and_calculate_tau function with cached mean and std values
    mean_values_tau, std_values_tau, mean_inputs, std_inputs, parsed_equations = create_process_input_and_calculate_tau(
        models, 0, 0)

    options = {'xtol': 1e-6, 'ftol': 1e-6, 'maxiter': 1000}

    for istep in range(1, nsteps + 1):
        print(f"istep: {istep}")

        #       tmp, info, exitflag, msg = optimize.fsolve(fsolve_function, u_n, **options, full_output=True)
        #       def fsolve_function(u):
        #           return reduced_F(u, au0, bu, cu, cutmp, u_n, mu, nb, Dt, tildeA, tildeB)
        #       tmp, info, exitflag, msg = fsolve(fsolve_function, u_n)
        tmp, info, exitflag, msg = fsolve(reduced_F_closure, u_n, args=(au0, bu, cu, cutmp, u_n, mu, nb, Dt, process_input_and_calculate_tau, mean_inputs,
                                          std_inputs, mean_values_tau, std_values_tau, parsed_equations, istep), xtol=options['xtol'], maxfev=options['maxiter'], full_output=True)

        if exitflag not in [1, 2, 3]:
            break

        u_n = tmp

        if istep % iostep == 0:
            ucoef[istep // iostep, :] = np.concatenate(([1], u_n))

    extended_vec[:, :nb+1] = ucoef
    err_wproj = extended_vec - uk

    err_l2 = np.diag(np.matmul(np.matmul(err_wproj, b0_full), err_wproj.T))
    err_l2_avg = np.sum(err_l2) / ns

    err_h10 = np.diag(np.matmul(np.matmul(err_wproj, a0_full), err_wproj.T))
    err_h10_avg = np.sum(err_h10) / ns

    err_list_l2['l2'] = err_l2
    err_list_h10['h10'] = err_h10

    # Construct filename suffix based on testp and ws
    if testp is None and ws is None:
        suffix = ""
    else:
        suffix = f"_ws{ws}_testp{testp}"
    # Save coefficients
    coefname = f"/ucoef_{suffix}.txt"
    np.savetxt(directory + coefname, ucoef, fmt='%24.15e')

    # Save final coefficients
    final_ucoef = ucoef[-1, :]
    coefname_final = f"/ufinal_{suffix}.txt"
    np.savetxt(directory + coefname_final, final_ucoef, fmt='%24.15e')

    # Write error tables to CSV files
    err_list_l2.to_csv(
        f"{directory}/err_l2_square_dt{Dt}_nsteps{nsteps}_mu{mu}_N{nb}_{suffix}.csv", index=False)
    err_list_h10.to_csv(
        f"{directory}/err_h10_square_dt{Dt}_nsteps{nsteps}_mu{mu}_N{nb}_{suffix}.csv", index=False)

    return istep, err_l2_avg, err_h10_avg


def reduced_F_closure(u, a, b, c, ctmp, u_n, mu, nb, Dt, tau_, mean_inputs, std_inputs, mean_values_tau, std_values_tau, parsed_equations, istep):
    # Contribution from the stiffness matrix
    b_inv = np.linalg.inv(b)
    F = - mu * b_inv @ a[1:nb+1, 1:nb+1] @ u
    F -= mu * b_inv @ a[1:nb+1, 0]  # Contribution from the stiffness matrix
    # Contribution from the advection tensor
    F -= b_inv @ np.reshape(c @ u, (nb, nb)) @ u
    F += b_inv @ ctmp[:, 0, 0] * 1 * 1
    F -= b_inv @ np.reshape(ctmp[:, 0, :],
                            (nb, nb+1)) @ np.concatenate(([1], u))
    F -= b_inv @ np.reshape(ctmp[:, :, 0],
                            (nb, nb+1)) @ np.concatenate(([1], u))
    F -= tau_(u, mean_inputs, std_inputs,
              mean_values_tau, std_values_tau, parsed_equations, istep)
    G = u - Dt * F - u_n
    return G


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


def calculate_taus(X1, X2, X3, X4, X5, X6, equations):
    # Prepare the context for evaluation
    context = {
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,  # Include required math functions
        'math': math, 'tau': [0] * len(equations)  # Placeholder for tau values
    }

    # Execute each equation in the context
    for eq in equations:
        exec(eq, context)

    return context['tau']


def create_process_input_and_calculate_tau(models, mean_std_files, mean_std_X_train_file):
    # Read model equations
    '''
    best_models = []
    for model in models:
        best_model = NeuralNetRegressor(
            module=NeuralNetwork,
            module__hidden_units=model["module__hidden_units"],
            module__dropout_rate=model["module__dropout_rate"])
        best_model.initialize()
        best_model.load_params(f_params=model["model_path"])
        best_models.append(best_model)

    # Load mean and standard deviation values for outputs
    mean_std_arrays = [np.load(file) for file in mean_std_files]
    mean_values_tau = np.array([mean_std[0] for mean_std in mean_std_arrays])
    std_values_tau = np.array([mean_std[1] for mean_std in mean_std_arrays])

    # Load mean and standard deviation values for inputs
    mean_std_X_train = np.load(mean_std_X_train_file)
    mean_inputs = mean_std_X_train[0]
    std_inputs = mean_std_X_train[1]
    '''
    best_models = [np.load(file) for file in models]

    # return mean_values_tau, std_values_tau, mean_inputs, std_inputs, best_models
    return 0, 0, 0, 0, best_models


def process_input_and_calculate_tau(input_values, mean_inputs, std_inputs, mean_values_tau, std_values_tau, parsed_equations, istep):
    # Adjust input_values using mean and std for inputs
    adjusted_inputs = (input_values - mean_inputs) / std_inputs

    # Calculate tau values with adjusted inputs
    # tau_values = calculate_taus(*adjusted_inputs, parsed_equations)
    adjusted_tau_values = [model[46, istep] for model in parsed_equations]
    # for i, eq in enumerate(parsed_equations):
    #    model_out = eq.forward(torch.from_numpy(
    #        adjusted_inputs.reshape(-1, 1).T).to(torch.float32))
    #    model_out_np = model_out.cpu().detach().numpy().flatten()
    #    adjusted_tau_values.append(
    #       model_out_np * std_values_tau[i] + mean_values_tau[i])

    # Adjust tau_values using mean and std for outputs
    # adjusted_tau_values = tau_values * std_values_tau + mean_values_tau

    return np.array(adjusted_tau_values).flatten()

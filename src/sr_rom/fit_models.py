import numpy as np
from sr_rom.vmsrom_solver import solve_ode
import sr_rom.roms as roms


def fit_lr(X, y, val_idx, sdir, directory, r, Re,
           T_sample, dt, iostep, energy_FOM):
    best_lr_rom = None
    best_score = -np.inf

    # generate alpha values to select
    alpha_values = []
    for i in range(6):
        alpha_values.append(10**(i)*np.arange(1, 10))

    alpha_values = np.array(alpha_values).flatten()
    alpha_values = np.insert(alpha_values, 0, 0)

    for alpha in alpha_values:
        lr_rom = roms.lr_rom(r=r, alpha=alpha)
        lr_rom.fit(X, y)
        # print([lr_rom.reg[i].solver_ for i in range(r)])

        _, energy_lr = solve_ode(Re, "LR", sdir, directory,
                                 T_sample, dt, iostep, r, lr_rom)

        err_FOM = np.mean((energy_lr[val_idx] - energy_FOM[val_idx])**2)
        score = 1 - err_FOM / \
            np.mean((energy_FOM[val_idx] - np.mean(energy_FOM[val_idx]))**2)
        if np.isnan(score):
            score = -1e5
        elif score > best_score:
            best_lr_rom = lr_rom
            best_score = score
            print(score, alpha, err_FOM)
    return best_lr_rom


def fit_sr(X, y, val_idx, sdir, directory, r, Re,
           T_sample, dt, iostep, energy_FOM):
    symbols_list = ['add,sub,mul,constant,variable',
                    'add,sub,mul,sin,constant,variable',
                    'add,sub,mul,exp,sin,constant,variable',
                    'add,sub,mul,sin,cos,constant,variable',
                    'add,sub,mul,exp,sin,cos,constant,variable',
                    'add,sub,mul,exp,sin,cos,square,log,constant,variable']
    best_sr_rom = None
    best_score = -np.inf

    for symbols in symbols_list:
        for max_length in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            for generations in [10, 25, 50, 75, 100]:
                params = {"allowed_symbols": symbols,
                          "generations": generations,
                          "max_length": max_length}
                sr_rom = roms.sr_rom(r=r, model_parameters=params)
                sr_rom.fit(X, y)

                _, energy_sr = solve_ode(Re, "SR", sdir, directory,
                                         T_sample, dt, iostep, r, sr_rom)

                err_FOM = np.mean((energy_sr[val_idx] - energy_FOM[val_idx])**2)
                r_2_FOM = 1 - err_FOM / \
                    np.mean((energy_FOM[val_idx] - np.mean(energy_FOM[val_idx]))**2)
                if r_2_FOM >= best_score:
                    best_sr_rom = sr_rom
                    best_score = r_2_FOM
                    print(err_FOM, r_2_FOM, params, flush=True)

    return best_sr_rom


def fit_nn(X, y, val_idx, sdir, directory, r, Re,
           T_sample, dt, iostep, energy_FOM, device):
    best_nn_rom = None
    best_score = -np.inf
    best_dict = {}
    architectures = [[64, 128, 256, 512, 256, 128, 64],
                     [64, 128, 256, 128, 64],
                     [64, 128, 64]]

    for lr in [1e-4, 1e-3, 1e-2]:
        for optimizer__weight_decay in [1e-5, 1e-4, 1e-3]:
            for module__dropout in [0.3, 0.4, 0.5]:
                for architecture in architectures:
                    params = {"module__hidden_units": architecture,
                              "module__r": r,
                              "module__dropout_rate": module__dropout,
                              "lr": lr,
                              "optimizer__weight_decay": optimizer__weight_decay}

                    nn_rom = roms.nn_rom(r, params, device)
                    nn_rom.fit(X, y)
                    # turn on inference mode
                    nn_rom.inference_mode()
                    _, energy_nn = solve_ode(Re, "NN", sdir, directory,
                                             T_sample, dt, iostep, r, nn_rom)

                    err_FOM = np.mean((energy_nn[val_idx] - energy_FOM[val_idx])**2)
                    r_2_FOM = 1 - err_FOM / \
                        np.mean((energy_FOM[val_idx] - np.mean(energy_FOM[val_idx]))**2)
                    if r_2_FOM >= best_score:
                        best_nn_rom = nn_rom
                        best_score = r_2_FOM
                        best_dict = params
                        print(err_FOM, r_2_FOM, params, flush=True)
    return best_nn_rom, best_dict

from pyoperon.sklearn import SymbolicRegressor
from sr_rom.data.data import process_data, split_data, smooth_data
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import time
import warnings
from matplotlib import cm

# suppress warnings
warnings.filterwarnings("ignore")


def save_results(reg, X, tau, X_train_val, y_train_val, X_test, y_test,
                 mean_std_train_Re, mean_std_train_comp, prb_name, ylabel, output_path, learning_time):

    train_val_score = reg.score(X_train_val, y_train_val)
    test_score = reg.score(X_test, y_test)

    str_model = reg.get_model_string(reg.model_, precision=5)

    # reconstruct the full dataset
    num_t = 2001
    num_re_train_val = int(len(X_train_val[:, 0])/num_t)
    num_re_test = int(len(X_test[:, 0])/num_t)
    num_re = num_re_train_val + num_re_test

    # model prediction
    X_scaled = (X[:, 1:] - mean_std_train_Re[0])/mean_std_train_Re[1]
    model_out = reg.predict(X_scaled)

    # revert data scaling
    model_out = model_out*mean_std_train_comp[1] + mean_std_train_comp[0]
    prediction = model_out.reshape((num_re, num_t), order="F")

    with open(output_path + "models.txt", "a") as text_file:
        text_file.write(prb_name + " = " + str_model + "\n")

    with open(output_path + "scores.txt", "a") as text_file:
        text_file.write(prb_name + " " + str(train_val_score) +
                        " " + str(test_score) + "\n")

    np.save(output_path + prb_name + "_pred", prediction)

    t = np.linspace(500, 520, 2001)
    re_mesh, t_mesh = np.meshgrid(Re, t)

    _, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={
        "projection": "3d"},  figsize=(20, 10))

    # plot the surface.
    surf = ax[0].plot_surface(re_mesh, t_mesh, tau.T, cmap=cm.coolwarm,
                              linewidth=0, antialiased=False)
    surf_comp = ax[1].plot_surface(re_mesh, t_mesh, prediction.T, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
    ax[0].set_xlabel(r"$Re$")
    ax[0].set_ylabel(r"time index")
    ax[1].set_xlabel(r"$Re$")
    ax[1].set_ylabel(r"time index")
    ax[0].set_title(r"VMS-ROM Closure term")
    ax[1].set_title(r"SR-ROM Closure term")
    plt.colorbar(surf, shrink=0.5, label=ylabel)
    plt.colorbar(surf_comp, shrink=0.5, label=ylabel)
    plt.savefig(output_path + prb_name + "_surface.png", dpi=100)

    print(f"{prb_name} learned in {learning_time}s!", flush=True)


def sr_rom_operon(train_val_data, test_data, X, tau, t_sample, output_path):
    with open(output_path + "scores.txt", "a") as text_file:
        text_file.write("Name" + " " + "R^2_train" + " " + "R^2_test\n")

        params = {
            'optimizer_iterations': [10],
            'n_threads': [16],
            'max_evaluations': [int(1e6)],
            'tournament_size': [3],
            'max_length': [20, 60, 100, 140],
            'generations': [10, 25, 50],
            'allowed_symbols': ['add,sub,mul,sin,cos,sqrt,square,acos,asin,exp,log,pow,constant,variable'],

        }

    print("Started training procedure for tau", flush=True)
    idx_train_sampled = np.argsort(train_val_data.y["X_sampled"][:, 0])
    num_trials = 10
    # training procedure for tau
    for i in range(5):
        y_train = train_val_data.y["tau"][:, :, i].flatten("F")
        y_train_sampled = train_val_data.y["tau"][:, ::t_sample, i].flatten("F")[
            idx_train_sampled]
        y_test = test_data.y["tau"][:, :, i].flatten("F")

        # Standardization
        mean_std_X_train = [np.mean(train_val_data.y["X_sampled"][:, 1:], axis=0),
                            np.std(train_val_data.y["X_sampled"][:, 1:], axis=0)]
        mean_std_train_comp = [np.mean(y_train, axis=0),
                               np.std(y_train, axis=0)]
        X_train_norm = (train_val_data.y["X"][:, 1:] -
                        mean_std_X_train[0])/mean_std_X_train[1]
        X_sampled_train_norm = (
            train_val_data.y["X_sampled"][idx_train_sampled, 1:] - mean_std_X_train[0])/mean_std_X_train[1]
        y_train_norm = (y_train - mean_std_train_comp[0]) / \
            mean_std_train_comp[1]
        y_sampled_train_norm = (y_train_sampled - mean_std_train_comp[0]) / \
            mean_std_train_comp[1]
        X_test_norm = (test_data.y["X"][:, 1:] -
                       mean_std_X_train[0])/mean_std_X_train[1]
        y_test_norm = (y_test - mean_std_train_comp[0])/mean_std_train_comp[1]

        gs = GridSearchCV(SymbolicRegressor(), params, cv=5, verbose=0, refit=False,
                          n_jobs=-1, return_train_score=False)
        params_results = {}
        tic = time.time()
        for trial in range(num_trials):
            print(f"GS number {trial}", flush=True)
            gs.fit(X_sampled_train_norm, y_sampled_train_norm)
            if trial == 0:
                params_results["params"] = gs.cv_results_["params"]
                # init mean test scores
                params_results["mean_test_score"] = np.zeros(
                    len(params_results["params"]))

            # update mean test scores
            params_results["mean_test_score"] += 1 / \
                num_trials*gs.cv_results_["mean_test_score"]

        # refit the model with best hyperparams in the entire dataset
        print(params_results["mean_test_score"], flush=True)
        best_params_idx = np.nanargmax(params_results["mean_test_score"])
        best_params = params_results["params"][best_params_idx]
        print(best_params, flush=True)
        reg = SymbolicRegressor(**best_params)
        reg.fit(X_sampled_train_norm, y_sampled_train_norm)
        toc = time.time()

        save_results(reg, X, tau[:, :, i], X_train_norm, y_train_norm, X_test_norm, y_test_norm,
                     mean_std_X_train, mean_std_train_comp,
                     "tau_" + str(i), r"$\tau_i$", output_path, toc-tic)


if __name__ == "__main__":
    output_path = sys.argv[1]
    windows = [3, 5, 7]
    # load data
    t_sample = 200
    Re, A, B, tau, a_FOM, X, X_sampled, residual = process_data(
        5, "2dcyl/Re200_300", t_sample)

    for w in windows:
        print(f"---Collecting results for window size {w}...!---", flush=True)
        new_folder = "results_w_" + str(w) + "_n_2"
        os.mkdir(output_path + new_folder)
        # process data
        A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=w, num_smoothing=2, r=5)

        train_data, val_data, train_val_data, test_data = split_data(
            Re, A_conv, B_conv, tau_conv, a_FOM, X, X_sampled, residual, test_size=0.6, shuffle_test=False)

        sr_rom_operon(train_val_data, test_data, X, tau_conv, t_sample,
                      output_path + new_folder + "/")
        print(f"---Results for window size {w} completed!---", flush=True)

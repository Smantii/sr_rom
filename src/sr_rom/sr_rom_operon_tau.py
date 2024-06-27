from pyoperon.sklearn import SymbolicRegressor
from sr_rom.data.data import process_data, split_data, smooth_data
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.model_selection import GridSearchCV
import time
import warnings
from matplotlib import cm

# suppress warnings
warnings.filterwarnings("ignore")


def save_results(reg, X, X_train_val, y_train_val, X_test, y_test,
                 mean_std_train_Re, mean_std_train_comp, prb_name, ylabel, output_path, learning_time):

    train_val_score = reg.score(X_train_val, y_train_val)
    test_score = reg.score(X_test, y_test)

    str_model = reg.get_model_string(reg.model_, precision=5)

    # reconstruct the full dataset
    num_t = 2001
    num_re_train_val = int(len(X_train_val[:, 0])/num_t)
    num_re_test = int(len(X_test[:, 0])/num_t)
    num_re = num_re_train_val + num_re_test
    idx_train_val = np.arange(num_re_train_val)
    # NOTE: we test extrapolation, so test set is at the end of reynolds interval
    idx_test = np.arange(num_re_train_val, num_re)
    tau = np.zeros((num_re, num_t))
    tau[idx_train_val] = y_train_val.reshape((num_re_train_val, num_t), order="F")
    tau[idx_test] = y_test.reshape((num_re_test, num_t), order="F")

    # model prediction
    X_scaled = (X - mean_std_train_Re[0])/mean_std_train_Re[1]
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

    # plot all the sections
    for i in range(num_re):
        plt.clf()
        plt.figure(num=1, figsize=(8, 5))
        plt.plot(t, tau_conv[i, :], label="Data")
        plt.plot(t, prediction[i, :], label="Prediction")
        plt.xlabel(r"$t$")
        plt.ylabel(ylabel + " section")
        plt.legend(loc="upper right")
        plt.savefig(output_path + prb_name + "_section_" + str(i) + ".png", dpi=100)

    print(f"{prb_name} learned in {learning_time}s!", flush=True)


def sr_rom_operon(train_val_data, test_data, X, symbols, output_path):
    with open(output_path + "scores.txt", "a") as text_file:
        text_file.write("Name" + " " + "R^2_train" + " " + "R^2_test\n")

    print("Started training procedure for tau", flush=True)
    # training procedure for tau
    for i in range(5):
        y_train = train_val_data.y[:, :, i].flatten("F")
        y_test = test_data.y[:, :, i].flatten("F")

        # Standardization
        mean_std_X_train = [np.mean(train_val_data.X, axis=0),
                            np.std(train_val_data.X, axis=0)]
        mean_std_train_comp = [np.mean(y_train, axis=0),
                               np.std(y_train, axis=0)]
        X_train_norm = (train_val_data.X - mean_std_X_train[0])/mean_std_X_train[1]
        y_train_norm = (y_train - mean_std_train_comp[0]) / \
            mean_std_train_comp[1]
        X_test_norm = (test_data.X - mean_std_X_train[0])/mean_std_X_train[1]
        y_test_norm = (y_test - mean_std_train_comp[0])/mean_std_train_comp[1]

        # reshuffling
        p_train = np.random.permutation(len(X_train_norm))
        p_test = np.random.permutation(len(X_test_norm))
        X_train_norm = X_train_norm[p_train]
        y_train_norm = y_train_norm[p_train]
        X_test_norm = X_test_norm[p_test]
        y_test_norm = y_test_norm[p_test]

        reg = SymbolicRegressor(
            allowed_symbols=symbols,
            optimizer_iterations=10,
            n_threads=16,
            max_evaluations=int(1e6),
            generations=50
        )

        params = {
            'max_length': [20, 40, 60, 80, 100],
            'tournament_size': [2, 3],
        }

        gs = GridSearchCV(reg, params, cv=3, verbose=3, refit=True, n_jobs=-1)
        tic = time.time()
        gs.fit(X_train_norm, y_train_norm)
        toc = time.time()

        save_results(gs.best_estimator_, X, X_train_norm, y_train_norm, X_test_norm, y_test_norm,
                     mean_std_X_train, mean_std_train_comp,
                     "tau_" + str(i), r"$\tau_i$", output_path, toc-tic)


if __name__ == "__main__":
    output_path = sys.argv[1]
    windows = [3, 5, 7]
    symbols = 'add,sub,mul,sin,cos,sqrt,square,acos,asin,constant,variable'
    # load data
    Re, A, B, tau, a_FOM, X = process_data(5, "2dcyl/Re200_300")

    for w in windows:
        print(f"---Collecting results for window size {w}...!---", flush=True)
        new_folder = "results_w_" + str(w) + "_n_2"
        os.mkdir(output_path + new_folder)
        # process data
        A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=w, num_smoothing=2, r=5)

        _, _, train_val_data, test_data = split_data(
            Re, A_conv, B_conv, tau_conv, a_FOM, X, test_size=0.2)

        sr_rom_operon(train_val_data, test_data, X, symbols,
                      output_path + new_folder + "/")
        print(f"---Results for window size {w} completed!---", flush=True)

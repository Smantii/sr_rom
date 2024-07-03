from pyoperon.sklearn import SymbolicRegressor
from sr_rom.data.data import process_data, split_data, smooth_data
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.model_selection import GridSearchCV
import time
import warnings

# suppress warnings
warnings.filterwarnings("ignore")


def save_results(reg, X_train_val, y_train_val, X_test, y_test,
                 mean_std_train_Re, mean_std_train_comp, prb_name, ylabel, output_path, learning_time):

    train_val_score = reg.score(X_train_val, y_train_val)
    test_score = reg.score(X_test, y_test)

    str_model = reg.get_model_string(reg.model_, precision=5)

    # reconstruct the full dataset (concatenate and sort to account for order)
    Re_data_norm = np.sort(np.concatenate((X_train_val, X_test)),
                           axis=0).reshape(-1, 1)
    prediction_norm = reg.predict(Re_data_norm)

    # revert data scaling
    Re_data = mean_std_train_Re[1]*Re_data_norm + mean_std_train_Re[0]
    X_train_val = mean_std_train_Re[1]*X_train_val + mean_std_train_Re[0]
    X_test = mean_std_train_Re[1]*X_test + mean_std_train_Re[0]
    prediction = mean_std_train_comp[1]*prediction_norm + mean_std_train_comp[0]
    y_train_val = mean_std_train_comp[1]*y_train_val + mean_std_train_comp[0]
    y_test = mean_std_train_comp[1]*y_test + mean_std_train_comp[0]

    with open(output_path + "models.txt", "a") as text_file:
        text_file.write(prb_name + " = " + str_model + "\n")

    with open(output_path + "scores.txt", "a") as text_file:
        text_file.write(prb_name + " " + str(train_val_score) +
                        " " + str(test_score) + "\n")

    np.save(output_path + prb_name + "_pred", prediction)

    plt.scatter(X_train_val, y_train_val,
                c="#b2df8a", marker=".", label="Training data")
    plt.scatter(X_test, y_test,
                c="#b2df8a", marker="*", label="Test data")
    plt.scatter(Re_data, prediction, c="#1f78b4", marker='x',
                label="Best sol", linewidths=0.5)
    plt.xlabel(r"$Re$")
    plt.ylabel(ylabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    plt.savefig(output_path + prb_name, dpi=300, bbox_inches='tight')

    plt.clf()

    Re_full_norm = np.linspace(Re_data_norm[0], Re_data_norm[-1], 1001).reshape(-1, 1)
    prediction_norm = reg.predict(Re_full_norm)

    # revert data scaling
    Re_full = mean_std_train_Re[1]*Re_full_norm + mean_std_train_Re[0]
    prediction = mean_std_train_comp[1]*prediction_norm + mean_std_train_comp[0]

    plt.scatter(X_train_val, y_train_val,
                c="#b2df8a", marker=".", label="Training data")
    plt.scatter(X_test, y_test,
                c="#b2df8a", marker="*", label="Test data")
    plt.plot(Re_full, prediction, c="#1f78b4",
             label="Best sol", linewidth=0.5)
    plt.xlabel(r"$Re$")
    plt.ylabel(ylabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    plt.savefig(output_path + prb_name + "_cont", dpi=300, bbox_inches='tight')
    print(f"{prb_name} learned in {learning_time}s!", flush=True)

    plt.clf()


def sr_rom_operon(train_val_data, test_data, output_path):
    with open(output_path + "scores.txt", "a") as text_file:
        text_file.write("Name" + " " + "R^2_train" + " " + "R^2_test\n")

    seed = 42
    np.random.seed(seed)

    params = {
        'max_length': [50, 100, 150, 200],
        'tournament_size': [2, 3],
        'allowed_symbols': ['add,sub,mul,sin,cos,sqrt,square,acos,asin,constant,variable',
                            'add,sub,mul,sin,cos,sqrt,square,acos,asin,exp,log,constant,variable',
                            'add,sub,mul,sin,cos,sqrt,square,acos,asin,exp,log,pow,constant,variable']
    }

    # reshuffling
    p_train = np.random.permutation(len(train_val_data.X))
    p_test = np.random.permutation(len(test_data.X))

    print("Started training procedure for A", flush=True)
    # training procedure for A
    for i in range(5):
        for j in range(5):
            X_train_val = train_val_data.X.reshape(-1, 1)
            y_train_val = train_val_data.y["A"][:, i, j]
            X_test = test_data.X.reshape(-1, 1)
            y_test = test_data.y["A"][:, i, j]

            # Standardization
            mean_std_train_Re = [np.mean(X_train_val), np.std(X_train_val)]
            mean_std_train_comp = [np.mean(y_train_val), np.std(y_train_val)]
            train_Re_norm = (X_train_val - mean_std_train_Re[0])/mean_std_train_Re[1]
            train_comp_norm = (y_train_val - mean_std_train_comp[0]) / \
                mean_std_train_comp[1]
            test_Re_norm = (X_test - mean_std_train_Re[0])/mean_std_train_Re[1]
            test_comp_norm = (y_test - mean_std_train_comp[0])/mean_std_train_comp[1]

            # train_Re_norm = train_Re_norm[p_train]
            # train_comp_norm = train_comp_norm[p_train]
            # test_Re_norm = test_Re_norm[p_test]
            # test_comp_norm = test_comp_norm[p_test]

            reg = SymbolicRegressor(
                optimizer_iterations=10,
                n_threads=16,
                epsilon=0,
                max_evaluations=int(1e6),
            )

            gs = GridSearchCV(reg, params, cv=3, verbose=3, refit=True, n_jobs=-1)
            tic = time.time()
            gs.fit(train_Re_norm, train_comp_norm)
            toc = time.time()

            print(gs.best_params_)

            save_results(gs.best_estimator_, train_Re_norm, train_comp_norm, test_Re_norm, test_comp_norm,
                         mean_std_train_Re, mean_std_train_comp,
                         "A_" + str(i) + str(j), r"$A_{ij}$", output_path, toc-tic)

    print("Done!", flush=True)

    # training procedure for B
    print("Started training procedure for B", flush=True)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                X_train_val = train_val_data.X.reshape(-1, 1)
                y_train_val = train_val_data.y["B"][:, i, j, k]
                X_test = test_data.X.reshape(-1, 1)
                y_test = test_data.y["B"][:, i, j, k]

                # Standardization
                train_Re_norm = (X_train_val - np.mean(X_train_val))/np.std(X_train_val)
                train_comp_norm = (y_train_val - np.mean(y_train_val)) / \
                    np.std(y_train_val)
                test_Re_norm = (X_test - np.mean(X_train_val))/np.std(X_train_val)
                test_comp_norm = (y_test - np.mean(y_train_val))/np.std(y_train_val)

                train_Re_norm = train_Re_norm[p_train]
                train_comp_norm = train_comp_norm[p_train]
                test_Re_norm = test_Re_norm[p_test]
                test_comp_norm = test_comp_norm[p_test]

                reg = SymbolicRegressor(
                    optimizer_iterations=10,
                    n_threads=16,
                    epsilon=0,
                    max_evaluations=int(1e6),
                )

                gs = GridSearchCV(reg, params, cv=3, verbose=0, refit=True, n_jobs=-1)
                tic = time.time()
                gs.fit(train_Re_norm, train_comp_norm)
                toc = time.time()

                save_results(gs.best_estimator_, train_Re_norm, train_comp_norm, test_Re_norm, test_comp_norm,
                             mean_std_train_Re, mean_std_train_comp,
                             "B_" + str(i) + str(j) + str(k), r"$B_{ij}$", output_path, toc-tic)

    print("Done!", flush=True)


if __name__ == "__main__":
    output_path = sys.argv[1]
    windows = [3, 5, 7]
    # symbols = 'add,sub,mul,sin,cos,sqrt,square,acos,asin,exp,log,constant,variable'
    # load data
    Re, A, B, tau, a_FOM, X = process_data(5, "2dcyl/Re200_300")

    for w in windows:
        print(f"---Collecting results for window size {w}...---", flush=True)
        new_folder = "results_w_" + str(w) + "_n_2"
        os.mkdir(output_path + new_folder)
        # process data
        A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=w, num_smoothing=2, r=5)

        _, _, train_val_data, test_data = split_data(
            Re, A_conv, B_conv, tau_conv, a_FOM, X, test_size=0.2, shuffle_test=False)

        sr_rom_operon(train_val_data, test_data, output_path + new_folder + "/")
        print(f"---Results for window size {w} completed!---", flush=True)

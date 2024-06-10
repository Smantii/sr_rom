from pyoperon.sklearn import SymbolicRegressor
from sr_rom.data.data import process_data, split_data, smooth_data
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def save_results(reg, X_train_val, y_train_val, X_test, y_test,  prb_name, ylabel):

    train_val_score = reg.score(X_train_val, y_train_val)
    test_pred = reg.predict(X_test)
    if not np.any(np.isnan(test_pred)):
        test_score = reg.score(X_test, y_test)
    else:
        test_score = -1000

    str_model = reg.get_model_string(reg.model_, precision=5)

    Re_data = np.sort(np.concatenate((X_train_val, X_test)),
                      axis=0).reshape(-1, 1)
    prediction = reg.predict(Re_data)

    with open("models.txt", "a") as text_file:
        text_file.write(prb_name + " = " + str_model + "\n")

    with open("scores.txt", "a") as text_file:
        text_file.write(prb_name + " " + str(train_val_score) +
                        " " + str(test_score) + "\n")

    np.save(prb_name + "_pred", prediction)

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
    plt.savefig(prb_name, dpi=300)

    plt.clf()

    k_sample = np.linspace(Re_data[0], Re_data[-1], 1001).reshape(-1, 1)
    prediction = reg.predict(k_sample)

    plt.scatter(X_train_val, y_train_val,
                c="#b2df8a", marker=".", label="Training data")
    plt.scatter(X_test, y_test,
                c="#b2df8a", marker="*", label="Test data")
    plt.plot(k_sample, prediction, c="#1f78b4",
             label="Best sol", linewidth=0.5)
    plt.xlabel(r"$Re$")
    plt.ylabel(ylabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    plt.savefig(prb_name + "_cont", dpi=300)
    print(f"{prb_name} learned!", flush=True)

    plt.clf()


def sr_rom_operon(train_data, val_data, train_val_data, test_data, symbols, output_path):

    os.chdir(output_path)

    with open("scores.txt", "a") as text_file:
        text_file.write("Name" + " " + "R^2_train" + " " + "R^2_test\n")

    print("Started training procedure for A", flush=True)
    # training procedure for A
    for i in range(5):
        for j in range(5):
            X_train = train_data.X.reshape(-1, 1)
            y_train = train_data.y["A"][:, i, j]
            X_val = val_data.X.reshape(-1, 1)
            y_val = val_data.y["A"][:, i, j]
            X_train_val = train_val_data.X.reshape(-1, 1)
            y_train_val = train_val_data.y["A"][:, i, j]
            X_test = test_data.X.reshape(-1, 1)
            y_test = test_data.y["A"][:, i, j]

            reg = SymbolicRegressor(
                allowed_symbols=symbols,
                optimizer_iterations=10,
                max_length=40,
                n_threads=32,
                epsilon=0,
                max_evaluations=int(1e6),
                tournament_size=3)

            reg.fit(X_train_val, y_train_val)

            save_results(reg, X_train_val, y_train_val, X_test, y_test,
                         "A_" + str(i) + str(j), r"$A_{ij}$")

    print("Done!", flush=True)

    # training procedure for B
    print("Started training procedure for B")
    for i in range(5):
        for j in range(5):
            for k in range(5):
                X_train = train_data.X.reshape(-1, 1)
                y_train = train_data.y["B"][:, i, j, k]
                X_val = val_data.X.reshape(-1, 1)
                y_val = val_data.y["B"][:, i, j, k]
                X_train_val = train_val_data.X.reshape(-1, 1)
                y_train_val = train_val_data.y["B"][:, i, j, k]
                X_test = test_data.X.reshape(-1, 1)
                y_test = test_data.y["B"][:, i, j, k]

                reg = SymbolicRegressor(
                    allowed_symbols=symbols,
                    optimizer_iterations=10,
                    max_length=40,
                    n_threads=32,
                    epsilon=0,
                    max_evaluations=int(1e6),
                    tournament_size=3)

                reg.fit(X_train_val, y_train_val)

                save_results(reg, X_train_val, y_train_val, X_test, y_test,
                             "B_" + str(i) + str(j) + str(k), r"$B_{ijk}$")

    print("Done!", flush=True)


if __name__ == "__main__":
    # load and process data
    Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")
    A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=5, num_smoothing=2, r=5)

    train_data, val_data, train_val_data, test_data = split_data(
        Re, 1000*A_conv, 1000*B_conv, tau_conv, a_FOM, test_size=0.6)

    symbols = 'add,sub,mul,sin,cos,sqrt,square,pow,acos,asin,atan,constant,variable'

    output_path = sys.argv[1]
    sr_rom_operon(train_data, val_data, train_val_data,
                  test_data, symbols, output_path)

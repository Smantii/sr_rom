from pyoperon.sklearn import SymbolicRegressor
from sr_rom.data.data import process_data, split_data, smooth_data
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def save_results(best_model, X_train, y_train, X_val,  y_val, X_train_val, y_train_val,
                 X_test, y_test, num_runs, prb_name, ylabel):

    train_val_score = best_model.score(X_train_val, y_train_val)
    test_score = best_model.score(X_test, y_test)

    str_model = best_model.get_model_string(best_model.model_, precision=5)

    Re_data = np.sort(np.concatenate((X_train_val, X_test)),
                      axis=0).reshape(-1, 1)
    prediction = best_model.predict(Re_data)

    with open("models.txt", "a") as text_file:
        text_file.write(prb_name + " = " + str_model + "\n")

    with open("scores.txt", "a") as text_file:
        text_file.write(prb_name + " " + str(train_val_score) +
                        " " + str(test_score) + "\n")

    np.save(prb_name + "_pred", prediction)

    plt.scatter(X_train, y_train,
                c="#b2df8a", marker=".", label="Training data")
    plt.scatter(X_val, y_val,
                c="#b2df8a", marker="s", label="Val data")
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
    prediction = best_model.predict(k_sample)

    plt.scatter(X_train, y_train,
                c="#b2df8a", marker=".", label="Training data")
    plt.scatter(X_val, y_val,
                c="#b2df8a", marker="h", label="Val data")
    plt.scatter(X_test, y_test,
                c="#b2df8a", marker="*", label="Test data")
    plt.plot(k_sample, prediction, c="#1f78b4",
             label="Best sol", linewidth=0.5)
    plt.xlabel(r"$Re$")
    plt.ylabel(ylabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fancybox=True, shadow=True)
    plt.savefig(prb_name + "_cont", dpi=300)
    print(f"{prb_name} learned in {num_runs} runs!")

    plt.clf()


def run(X_train, y_train, X_val, y_val):
    train_score = -np.inf
    val_score = -np.inf
    best_model = None

    num_runs = 0

    while train_score <= 0.8 or val_score <= 0.8:
        num_runs += 1
        reg = SymbolicRegressor(
            allowed_symbols=symbols,
            offspring_generator='basic',
            optimizer_iterations=10,
            max_length=40,
            initialization_method='btc',
            n_threads=32,
            objectives=['r2'],
            epsilon=0,
            random_state=None,
            reinserter='keep-best',
            max_evaluations=int(1e6),
            symbolic_mode=False,
            tournament_size=3,

        )

        reg.fit(X_train, y_train)
        train_score = reg.score(X_train, y_train)
        curr_val_score = reg.score(X_val, y_val)
        if curr_val_score > val_score:
            val_score = curr_val_score
            best_model = reg
            print("Update!", curr_val_score)

    return best_model, num_runs


def sr_rom_operon(train_data, val_data, train_val_data, test_data, symbols, output_path):

    os.chdir(output_path)

    with open("scores.txt", "a") as text_file:
        text_file.write("Name" + " " + "R^2_train" + " " + "R^2_test\n")

    print("Started training procedure for A")
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

            best_model, num_runs = run(X_train, y_train, X_val, y_val)

            save_results(best_model, X_train, y_train, X_val,  y_val,
                         X_train_val, y_train_val, X_test, y_test,
                         num_runs, "A_" + str(i) + str(j), r"$A_{ij}$")

    print("Done!")

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

                best_model, num_runs = run(X_train, y_train, X_val, y_val)

                save_results(best_model, X_train, y_train, X_val,  y_val,
                             X_train_val, y_train_val, X_test, y_test,
                             num_runs, "B_" + str(i) + str(j) + str(k), r"$B_{ijk}$")

    print("Done!")


if __name__ == "__main__":
    # load and process data
    Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")
    A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=3, num_smoothing=2, r=5)

    train_data, val_data, train_val_data, test_data = split_data(
        Re, 1000*A_conv, 1000*B_conv, tau_conv, a_FOM)

    symbols = 'add,sub,mul,sin,cos,sqrt,square,pow,acos,asin,atan,constant,variable'

    output_path = sys.argv[1]
    sr_rom_operon(train_data, val_data, train_val_data,
                  test_data, symbols, output_path)

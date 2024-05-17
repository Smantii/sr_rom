from pyoperon.sklearn import SymbolicRegressor
from sr_rom.data.data import process_data, split_data
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys


def sr_rom_operon(train_data, val_data, train_val_data, test_data, symbols, output_path):
    # training procedure for A

    os.chdir(output_path)

    with open("scores.txt", "a") as text_file:
        text_file.write("Name" + " " + "R^2_train" + " " + "R^2_test\n")

    print("Started!")
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

            val_score = -np.inf
            best_model = None

            num_runs = 0

            while val_score <= 0.5:
                num_runs += 1
                reg = SymbolicRegressor(
                    allowed_symbols=symbols,
                    offspring_generator='basic',
                    optimizer_iterations=10,
                    max_length=50,
                    initialization_max_length=20,
                    initialization_method='btc',
                    n_threads=32,
                    objectives=['mse'],
                    epsilon=0,
                    random_state=None,
                    reinserter='keep-best',
                    max_evaluations=int(1e6),
                    symbolic_mode=False,
                    tournament_size=3,

                )

                reg.fit(X_train, y_train)
                curr_val_score = reg.score(X_val, y_val)
                if curr_val_score >= val_score:
                    val_score = curr_val_score
                    best_model = reg
                    print("Update!", val_score)
                # else:
                #    print(val_score)

            train_val_score = best_model.score(X_train_val, y_train_val)
            test_score = best_model.score(X_test, y_test)

            str_model = best_model.get_model_string(best_model.model_)

            k_data = np.concatenate((X_train_val, X_test)).reshape(-1, 1)
            prediction = best_model.predict(k_data)

            prb_name = "A_" + str(i) + str(j)

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
            plt.scatter(k_data, prediction, c="#1f78b4", marker='x',
                        label="Best sol", linewidths=0.5)
            plt.xlabel(r"$Re$")
            plt.ylabel(r"$A_{ij}$")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       ncol=3, fancybox=True, shadow=True)
            plt.savefig(prb_name, dpi=300)

            plt.clf()

            k_sample = np.linspace(X_train[0], X_test[-1], 1001).reshape(-1, 1)
            prediction = best_model.predict(k_sample)

            plt.scatter(X_train, y_train,
                        c="#b2df8a", marker=".", label="Training data")
            plt.scatter(X_val, y_val,
                        c="#b2df8a", marker="h", label="Val data")
            plt.scatter(X_test, y_test,
                        c="#b2df8a", marker="*", label="Test data")
            plt.plot(k_sample, prediction, c="#1f78b4", label="Best sol", linewidth=0.5)
            plt.xlabel(r"$Re$")
            plt.ylabel(r"$A_{ij}$")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                       ncol=3, fancybox=True, shadow=True)
            plt.savefig(prb_name + "_cont", dpi=300)
            print(f"{prb_name} learned in {num_runs} runs!")

            plt.clf()


if __name__ == "__main__":
    # load and process data
    Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")

    train_data, val_data, train_val_data, test_data = split_data(
        Re, 1000*A, 1000*B, tau, a_FOM)

    symbols = 'add,sub,mul,sin,cos,constant,variable'

    output_path = sys.argv[1]
    sr_rom_operon(train_data, val_data, train_val_data,
                  test_data, symbols, output_path)

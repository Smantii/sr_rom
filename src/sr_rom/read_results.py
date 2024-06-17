import numpy as np
import os
from sympy import simplify, trigsimp, MatrixSymbol
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from sr_rom.data.data import process_data, split_data


def compute_statistics(r, path, models, scores, idx_test, err_l2_path, err_h10_path, simplify_models=False):
    os.chdir(path)
    train_r_2 = np.zeros(r**2 + r**3, dtype=np.float64)
    test_r_2 = np.zeros(r**2 + r**3, dtype=np.float64)
    A_model_flatten = np.zeros(r**2, dtype=object)
    B_model_flatten = np.zeros(r**3, dtype=object)

    with open(scores) as scores_file:
        scores_file = scores_file.readlines()

    # with open(models) as models_file:
    #    models_file = models_file.readlines()

    # load r^2 errors
    for i, line in enumerate(scores_file[1:]):
        line = line.split(" ")
        train_r_2[i] = line[1]
        test_r_2[i] = line[2]

    # compute r^2 stats
    mean_train_r_2 = np.mean(train_r_2)
    std_train_r_2 = np.std(train_r_2)
    mean_test_r_2 = np.mean(test_r_2)
    std_test_r_2 = np.std(test_r_2)
    # print(f"R^2 training: {mean_train_r_2} +/- {std_train_r_2}")
    # print(f"R^2 test: {mean_test_r_2} +/- {std_test_r_2}")

    err_l2 = np.loadtxt(err_l2_path, delimiter=",", skiprows=1)[idx_test, 1]
    err_h10 = np.loadtxt(err_h10_path, delimiter=",", skiprows=1)[idx_test, 1]

    mean_test_l2 = np.mean(err_l2)
    std_test_l2 = np.std(err_l2)
    mean_test_h10 = np.mean(err_h10)
    std_test_h10 = np.std(err_h10)

    results = [f"$ {round(mean_test_r_2,3)} \pm {round(std_test_r_2,3)}$",
               f"$ {round(mean_test_l2,3)} \pm {round(std_test_l2,3)}$",
               f"$ {round(mean_test_h10,3)} \pm {round(std_test_h10,3)}$"]

    '''
    bins = np.linspace(-1, 1, 100)

    plt.hist(train_r_2, bins, alpha=0.5, label=r'Training $R^2$')
    plt.hist(test_r_2, bins, alpha=0.5, label=r'Test $R^2$')
    plt.legend(loc='upper right')
    plt.savefig("hist.png", dpi=300)

    plt.clf()
    plt.bar(np.arange(25), train_r_2[:25], label=r'Training $R^2$')
    plt.bar(np.arange(25), test_r_2[:25], label=r'Test $R^2$')
    plt.xlabel("Component number for A")
    plt.ylabel(r"$R^2$")
    plt.legend(loc='lower left')
    plt.savefig("A_bar_plot.png", dpi=300)

    plt.clf()
    plt.bar(np.arange(125), train_r_2[25:], label=r'Training $R^2$')
    plt.bar(np.arange(125), test_r_2[25:], label=r'Test $R^2$')
    plt.xlabel("Component number for B")
    plt.ylabel(r"$R^2$")
    plt.legend(loc='lower left')
    plt.savefig("B_bar_plot.png", dpi=300)
    '''

    return results

    if simplify_models:
        open('simplified_models.txt', 'w').close()
        # simplify expressions
        for i, line in enumerate(models_file):
            line = line.split("=")
            fun_name = line[0]
            curr_model = line[1]
            simp_model = simplify(curr_model)
            with open("simplified_models.txt", "a") as text_file:
                text_file.write(fun_name + " = " + str(simp_model) + "\n")

    with open('simplified_models.txt') as models_file:
        # transform simplified expression in simpy models and then save them
        # in A or B resp.
        models_file = models_file.readlines()
        for i, line in enumerate(models_file):
            line = line.split("=")
            curr_model = parse_expr(line[1])
            if i <= r**2-1:
                A_model_flatten[i] = curr_model
            else:
                B_model_flatten[i - 25] = curr_model

    # reshape A and B
    A_model = A_model_flatten.reshape((r, r))
    B_model = B_model_flatten.reshape((r, r, r))

    # define a_FOM
    a_FOM = np.array(MatrixSymbol('a', 2001, r))
    print("Computing symbolic tau...")
    tau = (A_model @ a_FOM.T).T + np.einsum("lj, ijk, lk->li", a_FOM, B_model, a_FOM)
    np.save("tau.npy", tau)
    print("Done!")


if __name__ == "__main__":
    # load and process data
    Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    r2_scores = np.zeros((3, 5), dtype=object)
    l2_scores = np.zeros((3, 5), dtype=object)
    h10_scores = np.zeros((3, 5), dtype=object)

    path = os.path.join(dir_path, "nn_results/")
    results_dir = np.sort([name for name in os.listdir(path)])
    # iterate over directory with different test size
    for j, res_test in enumerate(results_dir):
        res_path = os.path.join(path, res_test)
        w_dir = np.sort([name for name in os.listdir(
            res_path) if name.replace(".out", "") == name])
        #    # iterate over directory with different window size
        for i, res_w in enumerate(w_dir):
            _, _, _, test_data = split_data(
                Re, A, B, tau, a_FOM, test_size=0.2 + 0.1*j)

            idx_test = test_data.y["idx"]

            res_w_path = os.path.join(res_path, res_w)
            models = os.path.join(res_w_path, "models.txt")
            scores = os.path.join(res_w_path, "scores.txt")
            err_l2_path = os.path.join(
                res_w_path, "vmsrom_nn_l2_error_w" + str(3 + 2*i) + "_tp" + str(20 + 10*j) + ".csv")
            err_h10_path = os.path.join(
                res_w_path, "vmsrom_nn_h10_error_w" + str(3 + 2*i) + "_tp" + str(20 + 10*j) + ".csv")

            results = compute_statistics(
                5, res_w_path, models, scores, idx_test, err_l2_path, err_h10_path)

            r2_scores[i, j] = results[0]
            l2_scores[i, j] = results[1]
            h10_scores[i, j] = results[2]
    print(r2_scores)
    print("--------------------")
    print(l2_scores)
    print("--------------------")
    print(h10_scores)
    # tau = np.load(os.path.join(path, "tau.npy"), allow_pickle=True)
    # print("Simplifying...")
    # print(trigsimp(tau[0, 0]))
    # print("Done!")

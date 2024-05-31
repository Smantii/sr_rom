import numpy as np
import os
from sympy import simplify, trigsimp, MatrixSymbol
from sympy.parsing.sympy_parser import parse_expr


def compute_statistics(r, path, models, scores, simplify_models=False):
    os.chdir(path)
    train_r_2 = np.zeros(r**2 + r**3, dtype=np.float64)
    test_r_2 = np.zeros(r**2 + r**3, dtype=np.float64)
    A_model_flatten = np.zeros(r**2, dtype=object)
    B_model_flatten = np.zeros(r**3, dtype=object)

    with open(scores) as scores_file:
        scores_file = scores_file.readlines()

    with open(models) as models_file:
        models_file = models_file.readlines()

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
    print(f"R^2 training: {mean_train_r_2} +/- {std_train_r_2}")
    print(f"R^2 test: {mean_test_r_2} +/- {std_test_r_2}")

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

    dir_path = os.path.dirname(os.path.realpath(__file__))

    path = os.path.join(dir_path, "results_w_3_n_2/")
    models = os.path.join(path, "models.txt")
    scores = os.path.join(path, "scores.txt")
    # tau = compute_statistics(5, path, models, scores)
    tau = np.load(os.path.join(path, "tau.npy"), allow_pickle=True)
    print("Simplifying...")
    print(trigsimp(tau[0, 0]))
    print("Done!")

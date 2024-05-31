import numpy as np
import os


def compute_statistics(scores):
    train_r_2 = np.zeros(150, dtype=np.float64)
    test_r_2 = np.zeros(150, dtype=np.float64)

    with open(scores) as f:
        f = f.readlines()

    for i, line in enumerate(f[1:]):
        line = line.split(" ")
        train_r_2[i] = line[1]
        test_r_2[i] = line[2]

    mean_train_r_2 = np.mean(train_r_2)
    std_train_r_2 = np.std(train_r_2)
    mean_test_r_2 = np.mean(test_r_2)
    std_test_r_2 = np.std(test_r_2)
    print(f"R^2 training: {mean_train_r_2} +/- {std_train_r_2}")
    print(f"R^2 test: {mean_test_r_2} +/- {std_test_r_2}")
    return mean_train_r_2, std_train_r_2, mean_test_r_2, std_test_r_2


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    scores = os.path.join(dir_path, "results_w_7_n_2/scores.txt")
    compute_statistics(scores)

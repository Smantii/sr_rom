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

    print(idx_test)

    # err_l2 = np.loadtxt(err_l2_path, delimiter=",", skiprows=1)[idx_test, 1]
    # err_h10 = np.loadtxt(err_h10_path, delimiter=",", skiprows=1)[idx_test, 1]

    # print(err_l2)

    # mean_test_l2 = np.mean(err_l2)
    # std_test_l2 = np.std(err_l2)
    # mean_test_h10 = np.mean(err_h10)
    # std_test_h10 = np.std(err_h10)

    results = [f"$ {round(mean_test_r_2,3)} \pm {round(std_test_r_2,3)}$",
               # f"$ {round(mean_test_l2,3)} \pm {round(std_test_l2,3)}$",
               # f"$ {round(mean_test_h10,3)} \pm {round(std_test_h10,3)}$"
               ]

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


def simplify_(simplify_models, A_model_flatten, B_model_flatten, r):
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


def plot_errors(Re, idx_tests, l2_error_tau, l2_error_nn, l2_error_sr, l2_error_lr, dd_vms_rom_error, grom):
    fig, axis = plt.subplots(nrows=4, ncols=1, figsize=(6.5, 8))
    letters = ["A", "B", "C", "D"]
    fontsize = 10
    plt.rcParams['font.size'] = fontsize
    for i, idx_test in enumerate(idx_tests):
        # axis[i].plot(Re, dd_vms_rom_error, c="#e41a1c",
        #             marker='o', label="DD VMS-ROM", ms=2.5, clip_on=False)
        # axis[i].plot(Re, l2_error_tau[:, i, 0], c='#377eb8',
        #             marker='o', label="VMS-ROM interp", ms=2.5, clip_on=False)
        axis[i].plot(Re, l2_error_nn[:, i, 0], c='#4daf4a',
                     marker='o', label="NN-ROM", ms=2.5)
        axis[i].plot(Re, l2_error_sr[:, i, 0], c='#984ea3',
                     marker='o', label="SR-ROM", ms=2.5)
        axis[i].plot(Re, l2_error_lr[:, i, 0], c='#ff7f00',
                     marker='o', label="LR-ROM", ms=2.5)
        # axis[i].plot(Re, grom, marker='o', label="GROM", ms=2.5)
        axis[i].scatter(Re[idx_test], 0.045*np.ones_like(Re[idx_test]),
                        marker=".", c="#e41a1c", clip_on=False)
        axis[i].text(-0.15, 1, letters[i], transform=axis[i].transAxes, weight="bold")
        axis[i].set_ylim(bottom=0)
        axis[i].grid(True)
        axis[i].set_xlabel("Re")
        axis[i].set_ylabel(r"$\epsilon_{L^2}$")
        axis[i].set_xticks([200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400])
        axis[i].set_xlim(200, 400)
        axis[i].set_ylim(bottom=0.045)
        handles, labels = axis[i].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', ncol=3)
    # plt.show()
    plt.savefig("plot_w_3_n_2_11_r_2.pdf", dpi=300)


if __name__ == "__main__":
    # load and process data
    t_sample = 200
    Re, A, B, tau, a_FOM, X, X_sampled, residual = process_data(
        5, "2dcyl/Re200_400", t_sample)
    simplify_models = False

    dir_path = os.path.dirname(os.path.realpath(__file__))
    r2_scores = np.zeros((3, 5), dtype=object)
    l2_error_tau = np.zeros((80, 4, 3), dtype=np.float64)
    l2_error_nn = np.zeros((80, 4, 3), dtype=np.float64)
    l2_error_sr = np.zeros((80, 4, 3), dtype=np.float64)
    l2_error_lr = np.zeros((80, 4, 3), dtype=np.float64)

    # li_tau_path = os.path.join(dir_path, "results_extrapolation/li_tau")
    lr_path = os.path.join(dir_path, "results_extrapolation/lr")
    nn_path = os.path.join(dir_path, "results_extrapolation/nn")
    sr_path = os.path.join(dir_path, "results_extrapolation/sr")
    dd_vms_rom_error = np.loadtxt(
        dir_path + "/results_extrapolation/vmsrom_l2_error.csv", delimiter=",", skiprows=1)[:, 1]
    grom = np.load(dir_path + "/results_extrapolation/grom.npy")
    results_dir = np.sort([name for name in os.listdir(lr_path)])
    idx_tests = []
    # iterate over directory with different test size
    for j, res_test in enumerate(results_dir):
        # res_path_li_tau = os.path.join(li_tau_path, res_test)
        res_path_nn = os.path.join(nn_path, res_test)
        res_path_sr = os.path.join(sr_path, res_test)
        res_path_lr = os.path.join(lr_path, res_test)
        w_dir = np.sort([name for name in os.listdir(
            res_path_lr) if name.replace(".out", "") == name])
        _, _, _, test_data = split_data(
            Re, A, B, tau, a_FOM, X, X_sampled, residual, test_size=0.4 + 0.15*j, shuffle_test=False)
        idx_test = test_data.y["idx"]
        idx_tests.append(idx_test)
        #    # iterate over directory with different window size
        for i, res_w in enumerate(w_dir):
            # res_w_path_tau = os.path.join(res_path_li_tau, res_w)
            res_w_path_nn = os.path.join(res_path_nn, res_w)
            res_w_path_sr = os.path.join(res_path_sr, res_w)
            res_w_path_lr = os.path.join(res_path_lr, res_w)
            # l2_error_tau[:, j, i] = np.load(res_w_path_tau + "/l2_rel_error.npy")
            l2_error_nn[:61, j, i] = np.load(res_w_path_nn + "/l2_rel_error.npy")
            l2_error_sr[:61, j, i] = np.load(res_w_path_sr + "/l2_rel_error.npy")
            l2_error_lr[:61, j, i] = np.load(res_w_path_lr + "/l2_rel_error.npy")
            l2_error_nn[61:, j, i] = np.load(res_w_path_nn + "/l2_rel_error_part.npy")
            l2_error_sr[61:, j, i] = np.load(res_w_path_sr + "/l2_rel_error_part.npy")
            l2_error_lr[61:, j, i] = np.load(res_w_path_lr + "/l2_rel_error_part.npy")
            if simplify_models:
                open(res_w_path_sr + '/simplified_models.txt', 'w').close()
                # simplify expressions
                with open(res_w_path_sr + "/models.txt") as models_file:
                    models_file = models_file.readlines()

                for i, line in enumerate(models_file):
                    line = line.split("=")
                    fun_name = line[0]
                    curr_model = line[1]
                    simp_model_int = simplify(curr_model)
                    simp_model = trigsimp(simp_model_int)
                    print(simp_model)
                    with open(res_w_path_sr + "/simplified_models.txt", "a") as text_file:
                        text_file.write(fun_name + " = " + str(simp_model) + "\n")

    plot_errors(Re, idx_tests, l2_error_tau, l2_error_nn,
                l2_error_sr, l2_error_lr, dd_vms_rom_error, grom)

    # mean_dd_vms_rom_error = np.mean(dd_vms_rom_error)
    print(np.mean(dd_vms_rom_error), np.std(dd_vms_rom_error))
    for i, idx_test in enumerate(idx_tests):
        print(np.mean(l2_error_sr[idx_test, i], axis=0),
              np.std(l2_error_sr[idx_test, i], axis=0))

    # tau = np.load(os.path.join(path, "tau.npy"), allow_pickle=True)
    # print("Simplifying...")
    # print(trigsimp(tau[0, 0]))
    # print("Done!")

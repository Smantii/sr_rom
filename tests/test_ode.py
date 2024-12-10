import numpy as np
from sr_rom.time_extrapolation.fit_models import solve_ode


def test_ode():
    # path to the NekROM operators
    sdir = f"/home/smanti/SR-ROM/src/sr_rom/data/2dcyl/Re/Re400"
    # path to results
    time_extrapolation_dir = "/home/smanti/SR-ROM/src/sr_rom/results_time_extrapolation/"
    results_dir = f"{time_extrapolation_dir}2dcyl/r_2/"
    lr_directory = results_dir + "lr/Re400/"
    sr_directory = results_dir + "sr/Re400/"
    nn_directory = results_dir + "nn/Re400/"

    # load actual ucoefs
    u_lr_true = np.load(f"{lr_directory}ucoefs_lr.npy")
    u_sr_true = np.load(f"{sr_directory}ucoefs_sr_0.npy")
    u_nn_true = np.load(f"{nn_directory}ucoefs_nn_0.npy")

    _, _, _, u_lr, _ = solve_ode([400, 500], sdir, lr_directory, "LR", 0,
                                 100, 0.01, 1, 2, 0)
    _, _, _, u_sr, _ = solve_ode([400, 500], sdir, sr_directory, "SR", 0,
                                 100, 0.01, 1, 2, 0)
    _, _, _, u_nn, _ = solve_ode([400, 500], sdir, nn_directory, "NN", 0,
                                 100, 0.01, 1, 2, 0)

    assert np.allclose(u_lr, u_lr_true)
    assert np.allclose(u_sr, u_sr_true)
    assert np.allclose(u_nn, u_nn_true)


if __name__ == "__main__":
    test_ode()

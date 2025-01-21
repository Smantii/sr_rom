import numpy as np
from sr_rom.vmsrom_solver import solve_ode


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

    # load actual energies
    ene_lr_true = np.load(f"{lr_directory}energy_lr.npy")
    ene_sr_true = np.load(f"{sr_directory}energy_sr_0.npy")
    ene_nn_true = np.load(f"{nn_directory}energy_nn_0.npy")

    u_lr, ene_lr = solve_ode(400, "LR", sdir, lr_directory,  100, 0.01, 1, 2, None)
    u_sr, ene_sr = solve_ode(400, "SR", sdir, sr_directory, 100, 0.01, 1, 2, None)
    u_nn, ene_nn = solve_ode(400, "NN", sdir, nn_directory, 100, 0.01, 1, 2, None)

    assert np.allclose(u_lr, u_lr_true)
    assert np.allclose(u_sr, u_sr_true)
    assert np.allclose(u_nn, u_nn_true)

    assert np.allclose(ene_lr, ene_lr_true)
    assert np.allclose(ene_sr, ene_sr_true)
    assert np.allclose(ene_nn, ene_nn_true)


if __name__ == "__main__":
    test_ode()

import pytest
import numpy as np
import pandas as pd
from sr_rom.code.utils import (
    load_rom_ops,
    get_rom_ops_r_dim,
    load_coefficients_training_data,
    load_initial_condition
)
from sr_rom.code.rom_online_solver_wclosure import rom_online_solver_wclosure


def test_ddvmsrom():
    mu = 1. / 200
    T_final = 20
    dt = 0.001
    nsteps = int(T_final / dt)
    iostep = int(nsteps / 2000)
    r = 5

    sdir = "./Re200/T20/ops"

    a0_full, b0_full, cu_full, mb = load_rom_ops(sdir)

    # ROM operators in R dimensional space
    au0_, bu0_, au_, bu_, cu_, cutmp_, cutmp1_ = get_rom_ops_r_dim(
        a0_full, b0_full, cu_full, mb)

    # ROM operators in r dimensional space
    au0, bu0, au, bu, cu, cutmp, cutmp1 = get_rom_ops_r_dim(
        a0_full, b0_full, cu_full, r)
    u0 = load_initial_condition(sdir, r)

    # Coefficients of the projected snapshots onto mb dimensional space
    uk, ns = load_coefficients_training_data(sdir, mb)

    path_to_model = "./tests/data/results_w_3_n_2/"

    # idx_Re is not needed in this case
    idx_Re = None
    istep, ucoef = rom_online_solver_wclosure(
        a0_full, b0_full, au0, bu0, au, bu, cu, cutmp, u0, uk, nsteps, iostep, mu, idx_Re, dt, r, "SR", path_to_model)

    ufinal = ucoef[-1, :]

    # Expected values to compare with ufinal
    expected_values = np.array([1.000000000000000e+00, 6.649945007946781e+00,  -9.143727275280920e-02,  -2.547955829231180e-01, 6.890082050178152e-02, 2.862299245236624e-01])

    # Compare ufinal with expected values within a tolerance
    np.testing.assert_almost_equal(ufinal, expected_values, decimal=10)

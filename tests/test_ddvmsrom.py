import pytest
import numpy as np
import pandas as pd
from src.sr_rom.code.utils import (
    find_snapshot_rank, 
    load_rom_ops, 
    get_rom_ops_r_dim, 
    load_coefficients_training_data, 
    load_initial_condition
)
from src.sr_rom.code.rom_online_solver_wclosure import rom_online_solver_wclosure

def test_ddvmsrom():
    mu = 1. / 200
    T_final = 20
    dt = 0.001
    nsteps = int(T_final / dt)
    iostep = int(nsteps / 2000)
    r = 5

    sdir = "tests/data/ops"

    R = find_snapshot_rank(sdir)
    print(f'Snapshot rank is: {R}')
    
    a0_full, b0_full, cu_full, mb = load_rom_ops(sdir)
    print("done loading ... ")
    
    # ROM operators in R dimensional space
    au0_, bu0_, au_, bu_, cu_, cutmp_, cutmp1_ = get_rom_ops_r_dim(a0_full, b0_full, cu_full, mb)
    
    rows = []
    # ROM operators in r dimensional space
    au0, bu0, au, bu, cu, cutmp, cutmp1 = get_rom_ops_r_dim(a0_full, b0_full, cu_full, r)
    u0 = load_initial_condition(sdir, r)
    
    # Coefficients of the projected snapshots onto mb dimensional space 
    uk, ns = load_coefficients_training_data(sdir, mb)
    
    u = np.zeros((r+1, 3))
    u[:, 0] = u0
    
    path_to_model = "src/sr_rom/results_extrapolation/sr/results_20/results_w_3_n_2/"
    istep, ucoef = rom_online_solver_wclosure(a0_full, b0_full, au0, bu0, au, bu, cu, cutmp, u0, uk, nsteps, iostep, mu, dt, r, "SR", path_to_model)

    ufinal = ucoef[-1, :]

    # Expected values to compare with ufinal
    expected_values = np.array([1.000000000000000e+00, 6.648509519624800e+00, -1.491776533618715e-01, -1.730509015385063e-01, 1.069563447406831e-01, 2.867211912376920e-01])

    # Compare ufinal with expected values within a tolerance
    np.testing.assert_almost_equal(ufinal, expected_values, decimal=10)
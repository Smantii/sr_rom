import numpy as np
import pandas as pd
import os
import argparse
from utils import find_snapshot_rank, load_rom_ops, get_rom_ops_r_dim, load_coefficients_training_data, load_initial_condition
from rom_online_solver_wclosure import rom_online_solver_wclosure


def main(Re):

    mu = 1. / Re  # 1/Re

    T_final = 20
    dt = 0.001
    nsteps = int(T_final / dt)
    iostep = int(nsteps / 2000)

    sdir = "/home/smanti/SR-ROM/src/sr_rom/data/2dcyl/Re200_300/Re" + str(Re)

    R = find_snapshot_rank(sdir)
    print(f'Snapshot rank is: {R}')

    a0_full, b0_full, cu_full, mb = load_rom_ops(sdir)
    print("done loading ... ")

    # ROM operators in R dimensional space
    au0_, bu0_, au_, bu_, cu_, cutmp_, cutmp1_ = get_rom_ops_r_dim(
        a0_full, b0_full, cu_full, mb)

    for r in [5]:
        rows = []
        # ROM operators in r dimensional space
        au0, bu0, au, bu, cu, cutmp, cutmp1 = get_rom_ops_r_dim(
            a0_full, b0_full, cu_full, r)
        u0 = load_initial_condition(sdir, r)

        # Coefficients of the projected snapshots onto mb dimensional space
        uk, ns = load_coefficients_training_data(sdir, mb)

        u = np.zeros((r+1, 3))
        u[:, 0] = u0

        # Create directory if it does not exist
        dir_name = f"nsteps{nsteps}_dt{dt}_N{r}_w_closure"
        os.makedirs(dir_name, exist_ok=True)

        istep, err_l2_avg, err_h10_avg = rom_online_solver_wclosure(
            a0_full, b0_full, au0, bu0, au, bu, cu, cutmp, u0, uk, nsteps, iostep, mu, dt, r, dir_name, ns)

        print(f"Re: {Re}, istep: {istep}, l2: {err_l2_avg}, h10: {err_h10_avg}")
        rows.append({'step': istep, 'l2': err_l2_avg, 'h10': err_h10_avg})

        avg_err_list = pd.DataFrame(rows)

        table_name = f'avg_err_dt{dt}_nsteps{nsteps}_mu{mu}_N{r}_wclosure.csv'
        avg_err_list.to_csv(table_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Reynolds number for ROM solver.")
    parser.add_argument('Re', type=int, help='Reynolds number to process')
    args = parser.parse_args()
    main(args.Re)

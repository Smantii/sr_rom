import warnings
import jax.numpy as jnp
from sr_rom.data.data import process_data, split_data
from alpine.data import Dataset
from alpine.gp import gpsymbreg as gps
from typing import Callable, Tuple, List
import ray
from deap import gp
from deap.base import Toolbox
import time
import sys
import yaml
from dctkit import config
from jax import jit, grad, Array
import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt
import dctkit as dt
from matplotlib import cm
import os


warnings.filterwarnings('ignore')
config()


num_cpus = 2


def compute_MSE_sol(individual: Callable, indlen: int,
                    Re_data: Dataset) -> Tuple[float, List]:

    i = 0

    Re_array = Re_data.X
    tau_i = Re_data.y['tau'][:, :, i]
    a_FOM = Re_data.y['a_FOM']

    tau_computed = jnp.array(
        list(map(individual, Re_array, a_FOM[:, :, 0], a_FOM[:, :, 1], a_FOM[:, :, 2], a_FOM[:, :, 3], a_FOM[:, :, 4])))

    # time_norm = jnp.sum((tau_i - jnp.mean(tau_i))**2, axis=1)
    time_norm = jnp.sum((tau_i - tau_computed)**2, axis=1)
    total_error = jnp.mean(time_norm)
    # total_error = jnp.mean((tau_i - tau_computed)**2)

    return total_error, tau_computed


def compute_MSE_sol_comp(individual: Callable, indlen: int,
                         Re_data: Dataset) -> Tuple[float, List]:

    i = 0

    Re_array = Re_data.X
    tau_i = Re_data.y['tau'][:, :, i]
    B_i = Re_data.y['B'][:, i, :, :]
    a_FOM = Re_data.y['a_FOM']

    # A_computed = jnp.array(list(map(individual, Re_array)))
    A_computed = jnp.array(Re_data.y['A'][:, 0, :].copy())
    A_computed = A_computed.at[:, 1].set(individual(Re_array))

    A_a_FOM = jnp.einsum("ij,ikj->ik", A_computed, a_FOM)
    a_FOM_T_B_a_FOM = jnp.einsum("lmj, ljk, lmk->lm", a_FOM, B_i, a_FOM)
    tau_computed = A_a_FOM + a_FOM_T_B_a_FOM

    # time_norm = jnp.sum((tau_i - jnp.mean(tau_i))**2, axis=1)
    time_norm = jnp.sum((tau_i - tau_computed)**2, axis=1)
    total_error = jnp.mean(time_norm)
    # total_error = jnp.mean((tau_i - tau_computed)**2)

    return total_error, tau_computed


def eval_MSE_sol(individual: Callable, indlen: int,
                 Re_data: Dataset) -> Tuple[float, List]:

    warnings.filterwarnings('ignore')

    config()

    total_error, A_computed = compute_MSE_sol(individual, indlen, Re_data)

    if jnp.isnan(total_error) or total_error > 1e6:
        total_error = 1e6

    return total_error, A_computed


def eval_MSE_and_tune_constants(tree, toolbox, Re_data: Dataset):
    warnings.filterwarnings("ignore")
    config()
    individual, n_constants = compile_individual_with_consts(tree, toolbox)

    def eval_err(consts):
        def ind_with_consts(x1, x2, x3, x4, x5, x6): return individual(
            x1, x2, x3, x4, x5, x6, consts)
        total_error, _ = compute_MSE_sol(ind_with_consts, 0, Re_data)
        return total_error

    objective = jit(eval_err)

    if n_constants > 0:
        obj_grad = jit(grad(eval_err))
        x0 = jnp.ones(n_constants)

        class fitting_problem:
            def fitness(self, x):
                total_err = objective(x)
                return [total_err]

            def gradient(self, x):
                return obj_grad(x)

            def get_bounds(self):
                return (-5.*jnp.ones(n_constants), 5.*jnp.ones(n_constants))

        # NLOPT solver
        prb = pg.problem(fitting_problem())
        algo = pg.algorithm(pg.nlopt(solver="lbfgs"))
        algo.extract(pg.nlopt).ftol_abs = 1e-4
        algo.extract(pg.nlopt).ftol_rel = 1e-4
        algo.extract(pg.nlopt).maxeval = 10
        pop = pg.population(prb, size=0)
        pop.push_back(x0)
        pop = algo.evolve(pop)
        opt_result = algo.extract(pg.nlopt).get_last_opt_result()
        if (opt_result == 1) or (opt_result == 3) or (opt_result == 4):
            best_fit = pop.champion_f[0]
            best_consts = pop.champion_x
        else:
            best_fit = jnp.nan
            best_consts = []

    else:
        best_fit = objective([])
        best_consts = []

    if jnp.isinf(best_fit) or jnp.isnan(best_fit):
        best_fit = 1e6

    return best_fit, best_consts


@ray.remote(num_cpus=num_cpus)
def eval_MSE(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
             Re_data: Dataset, penalty: float) -> float:
    objvals = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(individual, toolbox)

        def callable_with_consts(x1, x2, x3, x4, x5, x6): return callable(
            x1, x2, x3, x4, x5, x6, individual.consts)
        objvals[i], _ = eval_MSE_sol(callable_with_consts, indlen, Re_data)
    return objvals


@ray.remote(num_cpus=num_cpus)
def predict(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
            Re_data: Dataset, penalty: float) -> List:

    best_sols = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(individual, toolbox)

        def callable_with_consts(x1, x2, x3, x4, x5, x6): return callable(
            x1, x2, x3, x4, x5, x6, individual.consts)
        _, best_sols[i] = eval_MSE_sol(callable_with_consts, indlen, Re_data)

    return best_sols


@ray.remote(num_cpus=num_cpus)
def fitness(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
            Re_data: Dataset, penalty: float) -> Tuple[float, ]:

    attributes = []*len(individuals_batch)

    for _, individual in enumerate(individuals_batch):
        MSE, consts = eval_MSE_and_tune_constants(individual, toolbox, Re_data)

        # callable = toolbox.compile(expr=individual)
        # MSE, _ = eval_MSE_sol(callable, indlen, Re_data)

        # add penalty on length of the tree to promote simpler solutions
        fitness = (MSE + penalty["reg_param"]*len(individual),)
        # each value MUST be a tuple
        attributes.append({'consts': consts, 'fitness': fitness})

    return attributes


def compile_individual_with_consts(tree, toolbox):
    const_idx = 0
    tree_clone = toolbox.clone(tree)
    for i, node in enumerate(tree_clone):
        if isinstance(node, gp.Terminal) and node.name[0:3] != "ARG":
            if node.name == "a":
                new_node_name = "a[" + str(const_idx) + "]"
                tree_clone[i] = gp.Terminal(new_node_name, True, float)
                const_idx += 1

    individual = toolbox.compile(expr=tree_clone, extra_args=["a"])
    return individual, const_idx


def assign_consts(individuals, attributes):
    for ind, attr in zip(individuals, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]


def sr_rom(config_file_data, train_data, val_data, train_val_data, test_data, output_path):
    # pset = gp.PrimitiveSetTyped("MAIN", [float], Array)
    # pset = gp.PrimitiveSetTyped("MAIN", [float], float)
    pset = gp.PrimitiveSetTyped("MAIN", [float] + [Array]*5, Array)

    # rename arguments of the tree function
    pset.renameArguments(ARG0="k")
    pset.renameArguments(ARG1="a_1")
    pset.renameArguments(ARG2="a_2")
    pset.renameArguments(ARG3="a_3")
    pset.renameArguments(ARG4="a_4")
    pset.renameArguments(ARG5="a_5")

    # add constants
    pset.addTerminal(object, float, "a")

    # add base for R^5
    # for i in range(5):
    #    e_i = jnp.zeros(5, dtype=dt.float_dtype)
    #    e_i = e_i.at[i].set(1.)
    #    pset.addTerminal(e_i, Array, "e_" + str(i))

    penalty = config_file_data["gp"]['penalty']

    # define extra common arguments of fitness and predict functions
    common_params = {'penalty': penalty}

    # set seed if needed
    seed = ["sc_mul(Mul(Exp(Add(Sub(a_4, a_4), Add(a_1, a_2))), Exp(Square(Sin(a_3)))), MulF(CosF(LogF(Div(SquareF(a), AddF(a, a)))), Div(a, k)))"]

    gpsr = gps.GPSymbolicRegressor(
        pset=pset, fitness=fitness.remote,
        error_metric=eval_MSE.remote, predict_func=predict.remote,
        feature_extractors=[len], print_log=True,
        common_data=common_params, config_file_data=config_file_data,
        save_best_individual=True, save_train_fit_history=True,
        plot_best_individual_tree=False, callback_func=assign_consts,
        output_path=output_path, batch_size=1, seed=seed)

    start = time.perf_counter()
    # NOTE implement plot funcs and test error
    if config_file_data['gp']['validate']:
        gpsr.fit(train_data, val_data)
    else:
        gpsr.fit(train_val_data)

    print("Best MSE on the test set: ", gpsr.score(test_data))

    print("Best constants = ", gpsr.best.consts)

    # PLOTS
    tau_train_val = gpsr.predict(train_val_data)
    tau_test = gpsr.predict(test_data)

    # from sklearn.metrics import r2_score
    # print(r2_score(test_data.y['tau'][:, :, 0], tau_test))

    tau = np.vstack((train_val_data.y['tau'], test_data.y['tau']))[:, :, 0]
    tau_computed = np.vstack((tau_train_val, tau_test))

    re = np.concatenate((train_val_data.X, test_data.X))
    t = np.arange(2001)

    re_mesh, t_mesh = np.meshgrid(re, t)

    _, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={
        "projection": "3d"},  figsize=(20, 10))

    # Plot the surface.
    surf = ax[0].plot_surface(re_mesh, t_mesh, tau.T, cmap=cm.coolwarm,
                              linewidth=0, antialiased=False)
    surf_comp = ax[1].plot_surface(re_mesh, t_mesh, tau_computed.T, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
    ax[0].set_xlabel(r"$Re$")
    ax[0].set_ylabel(r"time index")
    ax[1].set_xlabel(r"$Re$")
    ax[1].set_ylabel(r"time index")
    ax[0].set_title(r"VMS-ROM Closure term")
    ax[1].set_title(r"Approximated Closure term")
    plt.colorbar(surf, shrink=0.5)
    plt.colorbar(surf_comp, shrink=0.5)
    os.chdir(output_path)

    plt.savefig("data_vs_sol.png", dpi=300)

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")


if __name__ == "__main__":
    n_args = len(sys.argv)
    param_file = sys.argv[1]
    with open(param_file) as config_file:
        config_file_data = yaml.safe_load(config_file)

    # load data
    # from sr_rom.data.data import generate_toy_data
    # k_array, A_B_list = generate_toy_data(5)
    Re, A, B, tau, a_FOM = process_data(5, "2dcyl/Re200_300")
    train_data, val_data, train_val_data, test_data = split_data(Re, A, B, tau, a_FOM)

    if n_args >= 3:
        output_path = sys.argv[2]
    else:
        output_path = "."
    sr_rom(config_file_data, train_data, val_data,
           train_val_data, test_data, output_path)

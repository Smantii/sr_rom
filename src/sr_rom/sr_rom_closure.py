import warnings
import jax.numpy as jnp
from sr_rom.data.data import process_data, split_data, smooth_data
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
from jax import jit, grad
import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os


warnings.filterwarnings('ignore')
config()


num_cpus = 2


def compute_MSE_sol(individual: Callable, Re_data: Dataset, tau_comp_idx: int, time_step: int) -> Tuple[float, List]:

    tau_true = Re_data.y["tau"][:, ::time_step, tau_comp_idx]
    tau_computed_sampled = individual(Re_data.y["X_sampled"][:, 1],
                                      Re_data.y["X_sampled"][:, 2],
                                      Re_data.y["X_sampled"][:, 3],
                                      Re_data.y["X_sampled"][:, 4],
                                      Re_data.y["X_sampled"][:, 5])
    tau_computed = individual(Re_data.y["X"][:, 1],
                              Re_data.y["X"][:, 2],
                              Re_data.y["X"][:, 3],
                              Re_data.y["X"][:, 4],
                              Re_data.y["X"][:, 5])

    tau_computed_reshaped = tau_computed_sampled.reshape(tau_true.shape, order="F")
    total_error_tau = jnp.mean(
        (tau_true - tau_computed_reshaped)**2)
    residual_error = jnp.mean(
        (Re_data.y["residual"][:, :, tau_comp_idx] - tau_computed_reshaped[:, :-1])**2)

    return total_error_tau + residual_error, tau_computed


def eval_MSE_sol(individual: Callable, Re_data: Dataset, tau_comp_idx: int, time_step: int) -> Tuple[float, List]:

    warnings.filterwarnings('ignore')

    config()

    total_error, A_computed = compute_MSE_sol(
        individual, Re_data, tau_comp_idx, time_step)

    if jnp.isnan(total_error) or total_error > 1e6:
        total_error = 1e6

    return total_error, A_computed


def eval_MSE_and_tune_constants(tree, toolbox, Re_data: Dataset, tau_comp_idx: int, time_step: int):
    warnings.filterwarnings("ignore")
    config()
    individual, n_constants = compile_individual_with_consts(tree, toolbox)

    def eval_err(consts):
        def ind_with_consts(a_1, a_2, a_3, a_4, a_5): return individual(
            a_1, a_2, a_3, a_4, a_5, consts)
        total_error, _ = compute_MSE_sol(
            ind_with_consts, Re_data, tau_comp_idx, time_step)
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
def eval_MSE(individuals_batch: list[gp.PrimitiveSet], toolbox: Toolbox,
             Re_data: Dataset, tau_comp_idx: int, time_step: int, penalty: float) -> float:
    objvals = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(individual, toolbox)

        def callable_with_consts(a_1, a_2, a_3, a_4, a_5):
            return callable(a_1, a_2, a_3, a_4, a_5, individual.consts)
        objvals[i], _ = eval_MSE_sol(
            callable_with_consts, Re_data, tau_comp_idx, time_step)
    return objvals


@ray.remote(num_cpus=num_cpus)
def predict(individuals_batch: list[gp.PrimitiveSet], toolbox: Toolbox,
            Re_data: Dataset, tau_comp_idx: int, time_step: int, penalty: float) -> List:

    best_sols = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(individual, toolbox)

        def callable_with_consts(a_1, a_2, a_3, a_4, a_5): return callable(
            a_1, a_2, a_3, a_4, a_5, individual.consts)
        _, best_sols[i] = eval_MSE_sol(
            callable_with_consts, Re_data, tau_comp_idx, time_step)

    return best_sols


@ray.remote(num_cpus=num_cpus)
def fitness(individuals_batch: list[gp.PrimitiveSet], toolbox: Toolbox,
            Re_data: Dataset, tau_comp_idx: int, time_step: int, penalty: float) -> Tuple[float, ]:

    attributes = []*len(individuals_batch)

    for _, individual in enumerate(individuals_batch):
        if str(individual).count("a_") == 0:
            # for constant individual we set the MSE equal to 1e6
            MSE, consts = 1e6, []
        else:
            MSE, consts = eval_MSE_and_tune_constants(
                individual, toolbox, Re_data, tau_comp_idx, time_step)

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


def sr_rom(config_file_data, train_data, val_data, train_val_data, test_data, time_step, output_path):
    tau_comp_idx = 0
    pset = gp.PrimitiveSetTyped("MAIN", [float]*5, float)

    # rename arguments of the tree function
    pset.renameArguments(ARG0="a_1")
    pset.renameArguments(ARG1="a_2")
    pset.renameArguments(ARG2="a_3")
    pset.renameArguments(ARG3="a_4")
    pset.renameArguments(ARG4="a_5")

    # add constants
    pset.addTerminal(object, float, "a")

    penalty = config_file_data["gp"]['penalty']

    # define extra common arguments of fitness and predict functions
    common_params = {'penalty': penalty,
                     'tau_comp_idx': tau_comp_idx, 'time_step': time_step}

    # set seed if needed
    seed = None

    gpsr = gps.GPSymbolicRegressor(
        pset=pset, fitness=fitness.remote,
        error_metric=eval_MSE.remote, predict_func=predict.remote,
        print_log=True, common_data=common_params, config_file_data=config_file_data,
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

    # ----- PLOTS -----
    os.chdir(output_path)
    # extract relevant quantities and init matrices
    num_re_train_val = int(len(train_val_data.y["X"][:, tau_comp_idx])/2001)
    num_re_test = int(len(test_data.y["X"][:, tau_comp_idx])/2001)
    num_re = num_re_train_val + num_re_test
    num_t = 2001
    idx_train_val = np.arange(num_re_train_val)
    # NOTE: we test extrapolation, so test set is at the end of reynolds interval
    idx_test = np.arange(num_re_train_val, num_re)
    tau = np.zeros((num_re, num_t))
    tau_computed = np.zeros_like(tau)
    re = np.zeros(num_re)

    # fill with right values at right indices
    tau[idx_train_val] = train_val_data.y["tau"][:, :, tau_comp_idx]
    tau[idx_test] = test_data.y["tau"][:, :, tau_comp_idx]
    tau_computed[idx_train_val] = gpsr.predict(
        train_val_data).reshape((num_re_train_val, 2001), order="F")
    tau_computed[idx_test] = gpsr.predict(
        test_data).reshape((num_re_test, 2001), order="F")
    # to get reynolds without repetitions we should take the first n points of X[:,0], where
    # n = number of different reynolds in the dataset
    re[idx_train_val] = train_val_data.y["X"][:len(idx_train_val), 0]
    re[idx_test] = test_data.y["X"][:len(idx_test), 0]

    t = np.arange(num_t)

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

    plt.savefig("data_vs_sol.png", dpi=300)

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")


if __name__ == "__main__":
    n_args = len(sys.argv)
    param_file = sys.argv[1]
    with open(param_file) as config_file:
        config_file_data = yaml.safe_load(config_file)

    # load data
    time_step = 1
    Re, A, B, tau, a_FOM, X, X_sampled, residual = process_data(
        5, "2dcyl/Re200_300", time_step)
    A_conv, B_conv, tau_conv = smooth_data(A, B, tau, w=3, num_smoothing=2, r=5)
    train_data, val_data, train_val_data, test_data = split_data(
        Re, A_conv, B_conv, tau_conv, a_FOM, X, X_sampled, residual, 0.6, False)

    if n_args >= 3:
        output_path = sys.argv[2]
    else:
        output_path = "."
    sr_rom(config_file_data, train_data, val_data,
           train_val_data, test_data, time_step, output_path)

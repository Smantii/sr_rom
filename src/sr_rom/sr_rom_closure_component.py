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
from jax import jit, grad
import pygmo as pg
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
config()


num_cpus = 2


def eval_MSE_sol(individual: Callable, indlen: int,
                 Re_data: Dataset) -> Tuple[float, List]:

    warnings.filterwarnings('ignore')

    config()

    k_array = Re_data.X
    A_0_0 = Re_data.y['A'][:, 0, 0]

    component_computed = individual(k_array)
    total_error = jnp.mean((component_computed - A_0_0)**2)

    if jnp.isnan(total_error) or total_error > 1e6:
        total_error = 1e6
    return total_error, component_computed


def eval_MSE_and_tune_constants(tree, toolbox, Re_data: Dataset):
    warnings.filterwarnings("ignore")
    config()
    individual, n_constants = compile_individual_with_consts(tree, toolbox)

    def eval_err(consts):
        k_array = Re_data.X
        A_0_0 = Re_data.y['A'][:, 0, 0]

        component_computed = individual(k_array, consts)
        total_error = jnp.mean((component_computed - A_0_0)**2)
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
        algo.extract(pg.nlopt).maxeval = 1000
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
        best_fit = eval_err([])
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
        def callable_with_consts(x): return callable(x, individual.consts)
        objvals[i], _ = eval_MSE_sol(callable_with_consts, indlen, Re_data)
    return objvals


@ray.remote(num_cpus=num_cpus)
def predict(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
            Re_data: Dataset, penalty: float) -> List:

    best_sols = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(individual, toolbox)
        def callable_with_consts(x): return callable(x, individual.consts)
        _, best_sols[i] = eval_MSE_sol(callable_with_consts, indlen, Re_data)

    return best_sols


@ray.remote(num_cpus=num_cpus)
def fitness(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
            Re_data: Dataset, penalty: float) -> Tuple[float, ]:

    attributes = []*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
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


def sr_rom(config_file_data, train_data, val_data, test_data, output_path):
    # for i in range(5):
    #    for j in range(5):
    # i = 0
    # j = 0

    pset = gp.PrimitiveSetTyped("MAIN", [float], float)

    # rename arguments of the tree function
    pset.renameArguments(ARG0="k")

    # add constants
    pset.addTerminal(object, float, "a")

    penalty = config_file_data["gp"]['penalty']

    # define extra common arguments of fitness and predict functions
    common_params = {'penalty': penalty}

    # set seed if needed
    seed = None

    gpsr = gps.GPSymbolicRegressor(
        pset=pset, fitness=fitness.remote,
        error_metric=eval_MSE.remote, predict_func=predict.remote,
        feature_extractors=[len], print_log=True,
        common_data=common_params, config_file_data=config_file_data,
        save_best_individual=True, save_train_fit_history=True,
        plot_best_individual_tree=False, callback_func=assign_consts,
        output_path=output_path, batch_size=1, seed=seed)

    start = time.perf_counter()
    if config_file_data['gp']['validate']:
        gpsr.fit(train_data, val_data)
    else:
        gpsr.fit(train_val_data)

    # recover the solution associated to the best individual among all the populations
    # comp_best = gpsr.predict(train_data)

    # compute test error
    print("Best MSE on the test set: ", gpsr.score(test_data))

    print("Best constants = ", gpsr.best.consts)

    # train_computed = gpsr.predict(train_data_i_j)
    # val_computed = gpsr.predict(val_data_i_j)
    train_val_computed = gpsr.predict(train_val_data)
    test_computed = gpsr.predict(test_data)

    # print(train_computed)
    # print(train_A_i_j)
    # print("------------------")
    # print(val_computed)
    # print(val_A_i_j)

    print(train_val_computed)
    print(train_val_data.y['A'][:, 0, 0])

    print("------------------")

    print(test_computed)
    print(test_data.y['A'][:, 0, 0])

    print("Best constants = ", gpsr.best.consts)

    k_data = np.concatenate((train_val_data.X, test_data.X))

    A_i_j_computed = np.concatenate((train_val_computed, test_computed))

    # NOTE:only for plot
    k_sample = np.linspace(min(k_data), max(k_data), 1001)
    data = Dataset("Re_data", k_sample, {'A': np.zeros((1001, 5, 5))})
    A_i_j_computed = gpsr.predict(data)

    plt.scatter(train_val_data.X, train_val_data.y['A'][:, 0, 0],
                c="#b2df8a", marker=".", label="Training data")
    plt.scatter(test_data.X, test_data.y['A'][:, 0, 0],
                c="#b2df8a", marker="*", label="Test data")
    plt.plot(k_sample, A_i_j_computed, c="#1f78b4", label="Best solution")
    # plt.plot(k_ord, A_i_j_computed_ord, c="#1f78b4", label="Best solution")
    plt.xlabel(r"$Re$")
    plt.ylabel(r"$A_{ij}$")
    plt.legend(loc="lower right")

    import os
    os.chdir(output_path)

    plt.savefig("data_vs_sol.pdf", dpi=300)

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
    train_data, val_data, train_val_data, test_data = split_data(
        Re, 1000*A, 1000*B, tau, a_FOM)

    if n_args >= 3:
        output_path = sys.argv[2]
    else:
        output_path = "."
    sr_rom(config_file_data, train_data, val_data, test_data, output_path)

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


config()


num_cpus = 2


def eval_MSE_sol(individual: Callable, indlen: int,
                 k_component: Dataset) -> Tuple[float, List]:

    warnings.filterwarnings('ignore')

    k_array = k_component.X
    component_computed = individual(k_array)
    total_error = jnp.mean((component_computed - k_component.y)**2)

    if jnp.isnan(total_error) or total_error > 1e6:
        total_error = 1e6
    return total_error, component_computed


def eval_MSE_and_tune_constants(tree, toolbox, k_component: Dataset):
    warnings.filterwarnings("ignore")
    config()
    individual, n_constants = compile_individual_with_consts(tree, toolbox)

    def eval_err(consts):
        k_array = k_component.X
        component_computed = individual(k_array, consts)
        total_error = jnp.mean((component_computed - k_component.y)**2)
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

        # PYGMO SOLVER
        prb = pg.problem(fitting_problem())
        algo = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))
        pop = pg.population(prb, size=1)
        pop.push_back(x0)
        pop = algo.evolve(pop)
        best_fit = pop.champion_f[0]
        best_consts = pop.champion_x
    else:
        best_fit = eval_err([])
        best_consts = []

    if jnp.isinf(best_fit) or jnp.isnan(best_fit):
        best_fit = 1e6

    return best_fit, best_consts


@ray.remote(num_cpus=num_cpus)
def eval_MSE(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
             k_component: Dataset, penalty: float) -> float:
    objvals = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(individual, toolbox)
        def callable_with_consts(x): return callable(x, individual.consts)
        objvals[i], _ = eval_MSE_sol(callable_with_consts, indlen, k_component)
    return objvals


@ray.remote(num_cpus=num_cpus)
def predict(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
            k_component: Dataset, penalty: float) -> List:

    best_sols = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(individual, toolbox)
        def callable_with_consts(x): return callable(x, individual.consts)
        _, best_sols[i] = eval_MSE_sol(callable_with_consts, indlen, k_component)

    return best_sols


@ray.remote(num_cpus=num_cpus)
def fitness(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
            k_component: Dataset, penalty: float) -> Tuple[float, ]:

    attributes = []*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        MSE, consts = eval_MSE_and_tune_constants(individual, toolbox, k_component)

        # callable = toolbox.compile(expr=individual)
        # MSE, _ = eval_MSE_sol(callable, indlen, k_component)

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


def sr_rom(config_file_data, train_data, val_data, test_data):
    best_ind_str = []
    ts_scores = jnp.zeros((5, 5), dtype=jnp.float64)

    # for i in range(5):
    #    for j in range(5):
    i = 0
    j = 0

    train_A_i_j = [A_B['A'][i, j]for A_B in train_data.y]
    val_A_i_j = [A_B['A'][i, j]for A_B in val_data.y]
    test_A_i_j = [A_B['A'][i, j]for A_B in test_data.y]
    train_val_A_i_j = train_A_i_j + val_A_i_j
    train_data_i_j = Dataset("k_component", jnp.array(
        train_data.X), jnp.array(train_A_i_j))
    val_data_i_j = Dataset("k_component", jnp.array(val_data.X), jnp.array(val_A_i_j))
    test_data_i_j = Dataset("k_component", jnp.array(
        test_data.X), jnp.array(test_A_i_j))
    train_val_data_i_j = Dataset(
        "k_component", jnp.array(train_data.X + val_data.X), jnp.array(train_val_A_i_j))

    # print(train_data_i_j.X, val_data_i_j.X, test_data_i_j.X)

    pset = gp.PrimitiveSetTyped("MAIN", [float], float)

    # rename arguments of the tree function
    pset.renameArguments(ARG0="k")

    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(1., float, name="1.")
    pset.addTerminal(2., float, name="2.")
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
        save_best_individual=False, save_train_fit_history=False,
        plot_best_individual_tree=False, callback_func=assign_consts,
        output_path="./", batch_size=200, seed=seed)

    start = time.perf_counter()
    if config_file_data['gp']['validate']:
        gpsr.fit(train_data_i_j, val_data_i_j)
    else:
        gpsr.fit(train_val_data_i_j)

    # recover the solution associated to the best individual among all the populations
    # comp_best = gpsr.predict(train_data)

    # compute and save test error
    best_ind_str.append(str(gpsr.best))
    ts_scores = ts_scores.at[i, j].set(gpsr.score(test_data_i_j))

    print("Best MSE on the test set: ", ts_scores[i, j])

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")

    print(gpsr.predict(train_data_i_j))
    print(train_A_i_j)
    print("------------------")
    print(gpsr.predict(val_data_i_j))
    print(val_A_i_j)

    # print(gpsr.predict(train_val_data_i_j))
    # print(train_val_A_i_j)

    print("------------------")

    print(gpsr.predict(test_data_i_j))
    print(test_A_i_j)

    print("Best constants = ", gpsr.best.consts)

    # jnp.savetxt("best_individuals.txt", jnp.array(best_ind_str), fmt="%s")
    # jnp.savetxt("test_scores.txt", ts_scores)


if __name__ == "__main__":

    param_file = sys.argv[1]
    with open(param_file) as config_file:
        config_file_data = yaml.safe_load(config_file)

    # load data
    # from sr_rom.data.data import generate_toy_data
    # k_array, A_B_list = generate_toy_data(5)
    k_array, A_B_list = process_data(5, "2dcyl")
    train_data, val_data, test_data = split_data(k_array, A_B_list)
    sr_rom(config_file_data, train_data, val_data, test_data)

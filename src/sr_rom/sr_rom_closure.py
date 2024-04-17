import numpy as np
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

import warnings

warnings.filterwarnings('ignore')


num_cpus = 2


def eval_MSE_sol(individual: Callable, indlen: int,
                 k_component: Dataset) -> Tuple[float, List]:

    warnings.filterwarnings('ignore')

    k_array = k_component.X
    # component_array = np.array(k_component.y)
    component_computed = individual(k_array)
    total_error = np.mean((component_computed - k_component.y)**2)

    if np.isnan(total_error) or total_error > 1e6:
        total_error = 1e6
    return total_error, component_computed


@ray.remote(num_cpus=num_cpus)
def eval_MSE(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
             k_component: Dataset, penalty: float) -> float:
    objvals = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable = toolbox.compile(expr=individual)
        objvals[i], _ = eval_MSE_sol(callable, indlen, k_component)
    return objvals


@ray.remote(num_cpus=num_cpus)
def predict(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
            k_component: Dataset, penalty: float) -> List:
    best_sols = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable = toolbox.compile(expr=individual)
        _, best_sols[i] = eval_MSE_sol(callable, indlen, k_component)

    return best_sols


@ray.remote(num_cpus=num_cpus)
def fitness(individuals_batch: list[gp.PrimitiveSet], indlen: int, toolbox: Toolbox,
            k_component: Dataset, penalty: float) -> Tuple[float, ]:

    objvals = [None]*len(individuals_batch)

    for i, individual in enumerate(individuals_batch):
        callable = toolbox.compile(expr=individual)
        MSE, _ = eval_MSE_sol(callable, indlen, k_component)

        # add penalty on length of the tree to promote simpler solutions
        fitness = MSE + penalty["reg_param"]*len(individual)
        # each value MUST be a tuple
        objvals[i] = (fitness,)

    return objvals


def sr_rom(config_file_data, train_data, val_data, test_data):
    best_ind_str = []
    ts_scores = np.zeros((5, 5), dtype=np.float64)

    # for i in range(5):
    #    for j in range(5):
    i = 0
    j = 0

    train_A_i_j = [A_B['A'][i, j]for A_B in train_data.y]
    val_A_i_j = [A_B['A'][i, j]for A_B in val_data.y]
    test_A_i_j = [A_B['A'][i, j]for A_B in test_data.y]
    train_val_A_i_j = train_A_i_j + val_A_i_j
    train_data_i_j = Dataset("k_component", train_data.X, np.array(train_A_i_j))
    val_data_i_j = Dataset("k_component", val_data.X, np.array(val_A_i_j))
    test_data_i_j = Dataset("k_component", test_data.X, np.array(test_A_i_j))
    train_val_data_i_j = Dataset(
        "k_component", train_data.X + val_data.X, np.array(train_val_A_i_j))

    # print(train_data_i_j.X, val_data_i_j.X, test_data_i_j.X)

    pset = gp.PrimitiveSetTyped("MAIN", [float], float)

    # rename arguments of the tree function
    pset.renameArguments(ARG0="k")

    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(1., float, name="1.")
    pset.addTerminal(2., float, name="2.")

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
        plot_best_individual_tree=False,
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
    ts_scores[i, j] = gpsr.score(test_data_i_j)

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

    # np.savetxt("best_individuals.txt", np.array(best_ind_str), fmt="%s")
    # np.savetxt("test_scores.txt", ts_scores)


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

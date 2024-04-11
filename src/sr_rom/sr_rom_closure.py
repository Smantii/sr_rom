import numpy as np
from sr_rom.data.data import generate_toy_data, split_data
from alpine.data import Dataset
from alpine.gp import gpsymbreg as gps
from typing import Callable, Tuple, List
import ray
from deap import gp
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
    component_list = k_component.y
    total_error = 0.
    component_computed_list = []
    for i, k in enumerate(k_array):
        component_computed = individual(k)
        total_error += 1/len(k_array)*(component_computed - component_list[i])**2
        component_computed_list.append(component_computed)

    if np.isnan(total_error) or total_error > 1e6:
        total_error = 1e6
    return total_error, component_computed_list


@ray.remote(num_cpus=num_cpus)
def eval_MSE(individual: Callable, indlen: int, k_component: Dataset,
             penalty: float) -> float:
    MSE, _ = eval_MSE_sol(individual, indlen, k_component)
    return MSE


@ray.remote(num_cpus=num_cpus)
def predict(individual: Callable, indlen: int, k_component: Dataset,
            penalty: float) -> List:
    _, pred = eval_MSE_sol(individual, indlen, k_component)
    return pred


@ray.remote(num_cpus=num_cpus)
def fitness(individual: Callable, indlen: int, k_component: Dataset,
            penalty: float) -> Tuple[float, ]:

    MSE, _ = eval_MSE_sol(individual, indlen, k_component)

    # add penalty on length of the tree to promote simpler solutions
    fitness = MSE + penalty["reg_param"]*indlen

    # return value MUST be a tuple
    return fitness,


def sr_rom(config_file_data, train_data, val_data, test_data):
    i = 0
    j = 0
    train_A_i_j = [A_B['A'][i, j]for A_B in train_data.y]
    val_A_i_j = [A_B['A'][i, j]for A_B in val_data.y]
    test_A_i_j = [A_B['A'][i, j]for A_B in test_data.y]
    train_data_i_j = Dataset("k_component", train_data.X, train_A_i_j)
    val_data_i_j = Dataset("k_component", val_data.X, val_A_i_j)
    test_data_i_j = Dataset("k_component", test_data.X, test_A_i_j)

    pset = gp.PrimitiveSetTyped("MAIN", [float], float)

    # rename arguments of the tree function
    pset.renameArguments(ARG0="k")

    penalty = config_file_data["gp"]['penalty']

    # define extra common arguments of fitness and predict functions
    common_params = {'penalty': penalty}

    gpsr = gps.GPSymbolicRegressor(
        pset=pset, fitness=fitness.remote,
        error_metric=eval_MSE.remote, predict_func=predict.remote,
        feature_extractors=[len], print_log=True,
        common_data=common_params, config_file_data=config_file_data,
        save_best_individual=True, save_train_fit_history=True,
        plot_best_individual_tree=False,
        output_path="./")

    start = time.perf_counter()
    gpsr.fit(train_data_i_j, val_data_i_j)

    # recover the solution associated to the best individual among all the populations
    # comp_best = gpsr.predict(train_data)

    # compute test error
    # print(f"Test Error: {eval_MSE()}")

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")


if __name__ == "__main__":

    param_file = sys.argv[1]
    with open(param_file) as config_file:
        config_file_data = yaml.safe_load(config_file)

    # load data
    k_array, A_B_list = generate_toy_data(3)
    train_data, val_data, test_data = split_data(k_array, A_B_list)
    sr_rom(config_file_data, train_data, val_data, test_data)

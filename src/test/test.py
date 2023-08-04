import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Add parent directory to the module search path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from mealpy.evolutionary_based import GA
from mealpy.swarm_based import SSA

from model import ROA


def knapsack(model, epoch=100, pop_size=50):
    """
    Solves the 0/1 Knapsack Problem using a given optimization algorithm.

    Args:
        model (obj): an optimization algorithm
        epoch (int): number of iterations, default = 100
        pop_size (int): population size, default = 50

    Return:
        obj: A handler to the optimization model.

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [+] References:
        https://developers.google.com/optimization/bin/knapsack
    """

    # Values of items (e.g., profits, revenues, etc.)
    VALUES = np.array([
        360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
        78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
        87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
        312
    ])

    # Weights of items (e.g., weights in a knapsack problem)
    WEIGHTS = np.array([
        7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
        42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
        3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13
    ])

    # Capacity constraint (e.g., maximum capacity of a knapsack)
    CAPACITY = 850

    # 50 items >> 50 dimensions
    # each item has two state: 0 (not put in the bag), 1 (put in the bag)
    # so lower bound is 0 and upper bound is 1.99 because
    # int(0 -> 0.99) = 0
    # int(1 -> 1.99) = 1
    LB = [0] * 50
    UB = [1.99] * 50

    def fitness_function(solution):
        """
        Calculates the fitness value for the given solution.

        Args:
            solution (np.array): Binary array representing the selected items.

        Returns:
            float: Fitness value of the solution.
        """
        def punish_function(value):
            """
            Handles constraint optimization problem by penalizing violations.

            Args:
                current_weight (float): Total weight of selected items.

            Returns:
                float: Penalty value (0 if the constraint is satisfied).
            """
            return 0 if value <= CAPACITY else value

        # Convert float to integer to make a binary decision (0 or 1)
        solution_int = solution.astype(int)
        current_capacity = np.sum(solution_int * WEIGHTS)
        temp = np.sum(solution_int * VALUES) - \
            punish_function(current_capacity)

        return temp

    # Define the problem dictionary for the optimize
    problem_dict = {
        "fit_func": fitness_function,
        "lb": LB,
        "ub": UB,
        "minmax": "max",
    }

    # Build the optimizer model
    optimizer_model = model(epoch=epoch, pop_size=pop_size)

    # Optimize the model by solve method
    best_position, best_fitness = optimizer_model.solve(problem_dict)

    # TODO: log the epoch results to file
    return optimizer_model


def egg_crate(model, epoch, pop_size):
    """
    a mathematicall function with global_minimum = 0

    Args:
        model (obj): an optimization algorithm module
        epoch (int): number of iterations, default = 100
        pop_size (int): population size, default = 50

    Return:
        obj: A handler to the optimization model.
    """
    def fitness_function(solution):
        return np.sum(solution**2 + 25*np.sin(solution)**2)

    # Define the problem dictionary for the optimize
    problem_dict = {
        "fit_func": fitness_function,
        "lb": [-5, ] * 3,
        "ub": [5, ] * 3,
        'minmax': 'min'
    }

    # Build the optimizer model
    optimizer_model = model(epoch=epoch, pop_size=pop_size)

    # Optimize the model by solve method
    best_position, best_fitness = optimizer_model.solve(problem_dict)

    return optimizer_model


def shubert(model, epoch, pop_size):
    """
    A mathematical function with global_minimum = -186.73.

    Args:
        model (module): an optimization algorithm module
        epoch (int): number of iterations, default = 100
        pop_size (int): population size, default = 50

    Return:
        a runned model
    """
    def fitness_function(solution):
        sum_part_1 = 0
        for i in range(1, 6):
            sum_part_1 += i*np.cos((i+1)*solution[0] + i)

        sum_part_2 = 0
        for i in range(1, 6):
            sum_part_2 += i*np.cos((i+1)*solution[1] + i)

        return sum_part_1*sum_part_2

    # Define the problem dictionary for the optimizer
    problem_dict = {
        "fit_func": fitness_function,  # objective function
        "lb": [-10, ] * 2,  # Lower bounds for X and Y
        "ub": [10, ] * 2,   # Upper bounds for X and Y
        'minmax': 'min'   # Minimization or Maximization
    }

    # TODO: Log the epoch results to a file

    # Build the optimizer model
    optimizer_model = model(epoch=epoch, pop_size=pop_size)

    # Optimize the model using the solve method
    best_position, best_fitness = optimizer_model.solve(problem_dict)

    return optimizer_model


class TestEvaluation:
    @staticmethod
    def time_report(time):
        """
        Convert the given time to a formatted string with appropriate units for reporting.

        Args:
            time (float): Time value in seconds.

        Returns:
            str: A formatted string representing the time in the format "X h Y min Z sec".
        """
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        seconds = round(time % 60, 3)

        time_string = ""
        if hours > 0:
            time_string += f"{hours} h "
        if minutes > 0:
            time_string += f"{minutes} min "
        if seconds > 0 or not time_string:
            time_string += f"{seconds} sec"

        return time_string

    def __init__(self, models):
        """
        The class will compare different optimizer algorithm to monitor the performance

        Args:
            models (dic): a dictionary of optimizer models:
                - keys : names
                - values : runned models
        """

        self.models = models

    def CompareModels(self, title):
        """
        Compare some models together and report fitness value plot, ouputs/

        Args:
            title (str): name of problem
        """

        # get the models and their's name from dictionary
        models = list(self.models.values())
        names = list(self.models.keys())

        # Store fitness values and runtimes of each model
        fitness_values_list = []
        runtimes = []
        for model in models:
            fitness_values_list.append(model.history.list_global_best_fit)
            runtimes.append(sum(model.history.list_epoch_time))
            time_string = self.time_report(runtimes[-1])
            # TODO: Use Logging module
            logger.info(
                f'{names[models.index(model)]} --> runtime = {time_string}, Best fitness value = {models[models.index(model)].solution[1][0]}')

        epoch = len(models[0].history.list_epoch_time)
        epoch_list = [i for i in range(epoch)]

        # plot global best ftiness values for each model and save in src/test/outputs/
        for i in range(len(models)):
            plt.plot(epoch_list, fitness_values_list[i], label=f'{names[i]}')

        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.legend()
        plt.savefig(f'src/test/outputs/{title}-fitness_values.png')
        plt.cla()


if __name__ == '__main__':
    # Optimize Knapsack problem with 3 optimizer algorithm
    roa_knp = knapsack(ROA.BaseROA, epoch=100, pop_size=50)
    ssa_knp = knapsack(SSA.BaseSSA, epoch=100, pop_size=50)
    ga_knp = knapsack(GA.BaseGA, epoch=100, pop_size=50)


    # Optimize Eggcreate benchmark with 3 optimizer algorithm
    roa_ec = egg_crate(ROA.BaseROA, epoch=100, pop_size=50)
    ga_ec = egg_crate(GA.BaseGA, epoch=100, pop_size=50)
    ssa_ec = egg_crate(SSA.BaseSSA, epoch=100, pop_size=50)


    # run some algorithms for Shubert equation
    roa_sh = shubert(ROA.BaseROA, epoch=100, pop_size=50)
    ga_sh = shubert(GA.BaseGA, epoch=100, pop_size=50)
    ssa_sh = shubert(SSA.BaseSSA, epoch=100, pop_size=50)


    # define a dictionary of optimizer models
    models = {
        'ROA': roa_knp,  # Rain Optimization Algorithm
        'GA': ga_knp,   # Genetic Algorithm
        'SSA': ssa_knp  # Sparrow Search Algorithm
    }

    # TODO: Log the result
    logger.info("Knapsack problem : ")
    test = TestEvaluation(models=models)
    test.CompareModels('Knapsack Problem')

    # model dictionary
    models = {
        'ROA': roa_ec,
        'GA': ga_ec,
        'SSA': ssa_ec,
    }

    logger.info("Egg Crate problem : ")
    test = TestEvaluation(models=models)
    test.CompareModels('Eggcreate Problem')

    # model dictionary
    models = {
        'ROA': roa_sh,
        'GA': ga_sh,
        'SSA': ssa_sh,
    }

    logger.info("Shubert problem : ")
    test = TestEvaluation(models=models)
    test.CompareModels('Shubert Problem')

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from mealpy.swarm_based import SSA
from mealpy.evolutionary_based import GA
from model import ROA


def Knapsack(model, epoch=100, pop_size=50):
    """
    Args:
        model (module): an optimization algorithm module
        epoch (int): number of iterations, default = 100
        pop_size (int): population size, default = 50

    Return:
        a runned model

    References:
        https://developers.google.com/optimization/bin/knapsack
    """
    VALUES = np.array([
        360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
        78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
        87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
        312
    ])
    WEIGHTS = np.array([
        7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
        42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
        3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13
    ])
    CAPACITY = 850

    ## 50 dimensions since we have 50 items
    ## each item has two state: 0 (not put in the bag), 1 (put in the bag)
    ## so lower bound is 0 and upper bound is 1.99 because
    ## int(0 -> 0.99) = 0
    ## int(1 -> 1.99) = 1
    LB = [0] * 50
    UB = [1.99] * 50

    def fitness_function(solution):
        def punish_function(value):
            """
            Using this function to handling constraint optimization problem
            """
            return 0 if value <= CAPACITY else value

        solution_int = solution.astype(int)                 # Convert float to integer here
        current_capacity = np.sum(solution_int * WEIGHTS)
        temp = np.sum(solution_int * VALUES) - punish_function(current_capacity)
        return temp


    problem_dict = {
        "fit_func": fitness_function,
        "lb": LB,
        "ub": UB,
        "minmax": "max",
    }
    # TODO: log the epoch results to file

    ## Run the algorithm
    model = model(epoch=epoch, pop_size=pop_size)
    best_position, best_fitness = model.solve(problem_dict)
    return model

def Eggcreate(model, epoch, pop_size):
    """
    a mathematicall function with global_minimum = 0

    Args:
        model (module): an optimization algorithm module
        epoch (int): number of iterations, default = 100
        pop_size (int): population size, default = 50

    Return:
        a runned model
    """
    def fitness_function(solution):
        return np.sum(solution**2 + 25*np.sin(solution)**2)

    problem_dict = {
        "fit_func": fitness_function, # objective function
        "lb": [-5, ] * 3,  # Lower bounds for X and Y
        "ub": [5, ] * 3,   # Upper bounds for X and Y
        'minmax': 'min'   # Minimization or Maximization
        }
        # TODO: log the epoch results to file


    model = model(epoch=epoch, pop_size=pop_size)
    best_position, best_fitness = model.solve(problem_dict)
    return model

def Shubert(model, epoch, pop_size):
    """
    a mathematicall function with global_minimum = -186.73

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

    problem_dict = {
        "fit_func": fitness_function, # objective function
        "lb": [-10, ] * 2,  # Lower bounds for X and Y
        "ub": [10, ] * 2,   # Upper bounds for X and Y
        'minmax': 'min'   # Minimization or Maximization
        }
        # TODO: log the epoch results to file

    model = model(epoch=epoch, pop_size=pop_size)
    best_position, best_fitness = model.solve(problem_dict)
    return model

class TestEvaluation:
    def __init__(self, models):
        """
            Initialize the evaluation class
        
            Args:
                models (dictionary): a dictionary of runned models:
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

        # store fitness values and runtime of each model
        fitness_values_list = []
        runtimes = []
        for i in range(len(models)):
            fitness_values_list.append(models[i].history.list_global_best_fit)
            runtimes.append(sum(models[i].history.list_epoch_time))
            print(f'{names[i]} --> runtime = {round(runtimes[-1], 2)} seconds, Best fitness value = {models[i].solution[1][0]}')

        epoch = len(models[0].history.list_epoch_time)
        epoch_list = [i for i in range(epoch)]

        # plot global best ftiness values for each model and save in src/test/outputs/
        for i in range(len(models)):
            plt.plot(epoch_list, fitness_values_list[i], label=f'{names[i]}')
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.legend()
        plt.savefig(f'outputs/{title}-fitness_values.png')
        plt.cla()


if __name__ == '__main__':
    # run some algorithms for Knapsack problem
    roa = Knapsack(ROA.ROA, epoch=100, pop_size=50)
    ssa = Knapsack(SSA.BaseSSA, epoch=100, pop_size=50)
    ga = Knapsack(GA.BaseGA, epoch=100, pop_size=50)


    # Compare runned models together 
    models = {
        'ROA': roa,
        'GA': ga,
        'SSA': ssa
    }

    print("Knapsack problem : ")
    test = TestEvaluation(models=models)
    test.CompareModels('Knapsack Problem')


    # run some algorithms for Eggcreate benchmark
    roa = Eggcreate(ROA.ROA, epoch=100, pop_size=50)
    ga = Eggcreate(GA.BaseGA, epoch=100, pop_size=50)
    ssa = Eggcreate(SSA.BaseSSA, epoch=100, pop_size=50)

    # Compare runned models together 
    models = {
        'ROA': roa,
        'GA': ga,
        'SSA': ssa,
    }

    print("Eggcreate problem : ")
    test = TestEvaluation(models=models)
    test.CompareModels('Eggcreate Problem')

    # run some algorithms for Shubert equation
    roa = Shubert(ROA.ROA, epoch=100, pop_size=50)
    ga = Shubert(GA.BaseGA, epoch=100, pop_size=50)
    ssa = Shubert(SSA.BaseSSA, epoch=100, pop_size=50)

    # Compare runned models together 
    models = {
        'ROA': roa,
        'GA': ga,
        'SSA': ssa,
    }

    print("Shubert problem : ")
    test = TestEvaluation(models=models)
    test.CompareModels('Shubert Problem')
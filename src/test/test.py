import numpy as np
import matplotlib.pyplot as plt


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
        'log_to': 'file',
        'log_file': f'outputs/KnapsackProblem-{model.__name__}-epoch={epoch}-popsize={pop_size}.log',
        "minmax": "max",
    }

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
        'log_to': 'file',
        'log_file': f'outputs/EggcreateProblem-{model.__name__}-epoch={epoch}-popsize={pop_size}.log',
        'minmax': 'min'   # Minimization or Maximization
        }


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
        'log_to': 'file',
        'log_file': f'outputs/ShubertProblem-{model.__name__}-epoch={epoch}-popsize={pop_size}.log',
        'minmax': 'min'   # Minimization or Maximization
        }

    model = model(epoch=epoch, pop_size=pop_size)
    best_position, best_fitness = model.solve(problem_dict)
    return model
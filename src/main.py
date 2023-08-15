from utils import write_solution, npv_calculator, run_simulator
from model import ROA

def obj_func(solution):
    """
    Objective function

    Args:
        solution (list): a solution that generated by algorithm

    Retrun:
        npv (float): npv value
    """
    # write solution to INCLUDE files
    keywords = ['WELSPECS', 'COMPDAT']
    write_solution(solution, keywords, num_inj=0, num_prod=6, n_params=4, is_green=True, is_include=True)
    # run simulator
    run_simulator()

    # calculate npv by reading .RSM
    npv = npv_calculator(npv_constants=npv_constants)
    return npv

if __name__ == '__main__':
    # number of injection, production and total wells
    num_inj = 0
    num_prod = 6
    num_wells = num_inj + num_prod

    # get npv constants from external file
    npv_constants = {}
    with open('src/npv_constants.txt', 'r') as constants:
        lines = constants.readlines()
        for line in lines:
            key, value = line.split('=')
            value = float(value)
            npv_constants[key] = value

    # PUNQS3-real DATA
    problem_dict = {
        'fit_func': obj_func, 
        "lb": [1, 1, 1, 1] * num_wells, # Min. [loc_i, loc_j, perf_k1, perf_k2]
        'ub': [19, 28, 5, 5] * num_wells, # Max. [loc_i, loc_j, perf_k1, perf_k2]
        'minmax': 'max', 
    }

    roa = ROA.BaseROA(epoch=10, pop_size=20)
    best_position, best_fitness = roa.solve(problem_dict)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")
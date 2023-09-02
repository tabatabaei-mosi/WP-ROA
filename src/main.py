from utils import decode_solution, write_solution, npv_calculator, run_simulator
from constraints import logical_constraint, physical_penalty
from npv_constants import constants
from optimizer import ROA
from loguru import logger


def obj_func(solution):
    """
    The objective function for optimizer. The fitness value objtained by this function may minimuze or maximuze.

    Args:
        solution (np.array, list): a solution that generated by algorithm

    Retrun:
        Fitness value (float): NPV value
    """

    # Decode solution
    locs_inj, perfs_inj, locs_prod, perfs_prod = decode_solution(
                                                        solution=solution,
                                                        num_inj=num_inj,
                                                        num_prod=num_prod,
                                                        n_params=n_params
                                                    )
    
    # Check for logical constraints
    logical_flag = logical_constraint(
                    locs_inj=locs_inj,
                    perfs_inj=perfs_inj,
                    locs_prod=locs_prod,
                    perfs_prod=perfs_prod,
                )
    
    # Check for physical constraints
    physical_flag, num_faults =  physical_penalty(
                                            model_name=model_name,
                                            locs_inj=locs_inj,
                                            perfs_inj=perfs_inj,
                                            locs_prod=locs_prod,
                                            perfs_prod=perfs_prod,
                                            gridsize=gridsize, 
                                            targets=['null_block', 'min_space', 'border'],
                                            well_space=well_space,
                                            null_space=null_space
                                        )
    
    # If a logical constraint violated -> NPV = 0
    if logical_flag or physical_flag:
        return 0
    
    else:
        # write solution to INCLUDE files
        write_solution(
                locs_inj=locs_inj,
                perfs_inj=perfs_inj,
                locs_prod=locs_prod,
                perfs_prod=perfs_prod,
                keywords=keywords, 
                is_green=True, is_include=True
            )
        
        # run simulator
        run_simulator()

        # calculate NPV by reading .RSM file (resuled from simulation)
        NPV = npv_calculator(
                model_name=model_name, 
                npv_constants=npv_constants
            )
        
        # Punish the algorithm based on the number of faults and penalty_coeff
        if num_faults >= 1:
            punishment_frac = penalty_coeff * num_faults

            # if punishment fraction is larger than 1 -> NPV = 0
            if punishment_frac >= 1:
                return 0
        
            # Reduced NPV by penalty coeff.
            else:       
                NPV -= (punishment_frac * NPV)
        
        return NPV



# number of injections, productions and number of optimization paramaeters
num_inj = 0
num_prod = 6
n_params = 4
epoch = 10
pop_size = 20

# specify working keywords
keywords = ['WELSPECS', 'COMPDAT']

# Enter the model name (.DATA name)
model_name = 'PUNQS3'

# GRID_SIZE -- PUNQS3
gridsize = [19, 28, 5]

num_wells = num_inj + num_prod
npv_constants = constants

# penalty coefficient
penalty_coeff = 0.5

# Minumm well spacing
# Minimum distance to null blocks
well_space, null_space = 2, 2


# PUNQS3 DATA
problem_dict = {
    'fit_func': obj_func, 
    "lb": [1, 1, 1, 1] * num_wells,         # lower boundary to [loc_i, loc_j, perf_k1, perf_k2]
    'ub': [19, 28, 5, 5] * num_wells,       # upper boundary to [loc_i, loc_j, perf_k1, perf_k2]
    'minmax': 'max', 
}


roa = ROA.BaseROA(epoch=epoch, pop_size=pop_size)
best_position, best_fitness = roa.solve(problem_dict)

logger.info(f"Solution: {best_position}, Fitness: {best_fitness}")

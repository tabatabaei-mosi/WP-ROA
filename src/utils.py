import subprocess
import functools
import pandas as pd
import numpy as np
from pathlib import Path
from mealpy.tuner import Tuner

abs_to_src = Path(__file__).resolve().parent

def path_check(path):
    """
    Check if a directory exists at the given path and create it if it doesn't exist.

    Args:
        path (str or Path): The path to the directory that needs to be checked and created if absent.
    
    Returns:
        None
    """
    # Convert the input path to a Path object
    path_object = Path(path)

    # Check if the directory already exists
    if not path_object.exists():

        # Create the directory since it doesn't exist
        path_object.mkdir(parents=True, exist_ok=True)


def split_solution(solution, num_inj=0, n_params=4):
    """
    split the solution to two different part: The optimized parameters related to (1) injection and (2) production wells.

    Args:
        solution (np.array, list): The original solution generated by optimizer
        num_inj (int): number of injection wells, default = 0
        n_parmas (int): number of decision variable for each well, default = 4

    Return:
        param_inj (np.array, list): including injection wells parameters
        param_prod (np.array, list): including production wells parameters
    """
    # split the whole solution to 2 parts, injections and productions
    param_inj = solution[: num_inj * n_params]
    param_prod = solution[num_inj * n_params: ]
    
    return param_inj, param_prod


def decode_solution(
    solution, 
    num_inj=0, 
    num_prod=6, 
    n_params=4
):
    """
    Decode solutions to 4 parts: locations of injections, perforations of injection, locations of productions and perforations of productions

    Args:
        solution (np.array, list): The original solution that generated by optimizer, ex. : [pi, pj, pk1, pk2, ...., ii, ij, ik1, ik2, ....]
        num_inj (int): number of injection wells, default = 0
        num_prod (int): number of production wells, default = 6
        n_parmas (int): number of decision variable for each well, default = 4

    Return:
        locs_inj (np.array, list): including locations of injections, ex. : [[i1, j1], [i2, j2], ...]
        perfs_inj (np.array, list): including perforations of injections
        locs_prod (np.array, list): including locations of productions
        perfs_prod (np.array, list): including perforations of productions
    """
    # get injections and production pramateres from split_solution
    param_inj, param_prod = split_solution(solution, num_inj, n_params)

    # store final locations and perforations of injection
    locs_inj = []
    perfs_inj = []

    # the variables to slice the list to location and perforation
    start = 0
    slice_loc = 2
    slice_perf = 4

    # Seperate the loc and perfs of inj wells 
    for _ in range(num_inj):
        locs_inj.append(param_inj[start: start + slice_loc])
        perfs_inj.append(param_inj[start + slice_loc: start + slice_perf])
        start += n_params

    # store final locations and perforations of productions
    locs_prod = []
    perfs_prod = []

    start = 0
    
    # Seperate the loc and perfs of prod wells 
    for _ in range(num_prod):
        locs_prod.append(param_prod[start: start + slice_loc])
        perfs_prod.append(param_prod[start + slice_loc: start + slice_perf])
        start += n_params

    # Convert locs_inj and perfs_inj lists to numpy arrays with integer data type
    locs_inj, perfs_inj = np.array(locs_inj, dtype=int), np.array(perfs_inj, dtype=int)

    # Convert locs_prod and perfs_prod lists to numpy arrays with integer data type
    locs_prod, perfs_prod = np.array(locs_prod, dtype=int), np.array(perfs_prod, dtype=int)

    return locs_inj, perfs_inj, locs_prod, perfs_prod


def write_solution(
    locs_inj, perfs_inj,
    locs_prod, perfs_prod, 
    keywords, 
    is_green=True, is_include=True, is_copy=False
):
    """
    This function will write get the raw solution from optimizer and then split, decode and finnally write down different part
    of optimized solution in relevent files where later will include in .DATA file. 

    Args:
        locs_inj (np.array, list): including locations of injections, ex. : [[i1, j1], [i2, j2], ...]
        perfs_inj (np.array, list): including perforations of injections
        locs_prod (np.array, list): including locations of productions
        perfs_prod (np.array, list): including perforations of productions
        keywords (list): including the target keywords of .DATA file (Possible Values: WELSPECS, COMPDAT, WCONPROD, WCONINJE)
        is_green (bool): True, if the reservoir is green (no infill wells); False if the reservoir already has some infill wells.
        is_include (bool): True, if ".DATA" will include sections (e.g., WELSPECS); False if the info will write on ".DATA" itself. 

    Return:
        None
    """
    # a function for write the WELSPECS keyword
    def write_welspecs(file, locs_inj, locs_prod):
        """
        Write the WELSPECS keyword section to a file.

        Args:
            file (file): An open file object to write the keyword section to.
            locs_inj (np.array, list): all the well locations for injection wells
            locs_prod (np.array list): all the well locations for production wells
        """
        # Write the WELSPECS keyword
        file.write(f'{keyword}\n')

        # PUNQS3 injection and production well properties
        # keys : well_names, values: RDBHP values
        wellspec_dic= {
            'PRO-1': 2362.2,
            'PRO-4': 2373.0,
            'PRO-5': 2381.7,
            'PRO-11': 2386.0,
            'PRO-12': 2380.5,
            'PRO-15': 2381.0,
        }

        # Index for well names and RDBHP
        idx_counter = 0

        # Write injection wells
        for well_name, RDBHP in wellspec_dic.items():
            if "INJ" not in well_name:
                continue

            i = locs_inj[idx_counter][0]
            j = locs_inj[idx_counter][1]
            # Injection WELSPECS template
            template = [f'\'{well_name}\'', '\'G1\'', str(int(i)), str(int(j)), str(RDBHP),
                         '\'WATER\'', '1*', '\'STD\'', '3*', '\'SEG\'', '  /']
            line = '  '.join(template)
            idx_counter += 1
            file.write(f'{line}\n')
     
        # Reset idx_counter to zero
        idx_counter = 0

        # Write production wells
        for well_name, RDBHP in wellspec_dic.items():
            if "PRO" not in well_name:
                continue

            i = locs_prod[idx_counter][0]
            j = locs_prod[idx_counter][1]
            # Production template
            template = [f'\'{well_name}\'', '\'G1\'', str(int(i)), str(int(j)), str(RDBHP),
                         '\'OIL\'', '1*', '\'STD\'', '3*', '\'SEG\'', '  /']
            line = '  '.join(template)
            idx_counter += 1
            file.write(f'{line}\n')

        # End of the WELSPECS section
        file.write('/')

    # a function for write the COMPDAT keyword
    def write_compdat(file, perfs_inj, perfs_prod):
        """
        Write the COMPDAT keyword section to a file.

        Args:
            file (file): An open file object to write the keyword section to.
            perfs_inj (np.array, list): all the well locations for injection wells
            perfs_prod (np.array, list): all the well locations for production wells
        """
        # Write the COMPDAT keyword
        file.write(f'{keyword}\n')

        # PUNQS3 properties
        name_idxs = [1, 4, 5, 11, 12, 15]
        idx_counter = 0

        # write injection wells
        for k1, k2 in perfs_inj:
            well_name = f'INJ-{name_idxs[idx_counter]}'
            # injection tmplate
            template = [f'\'{well_name}\'', '2*', str(int(k1)), str(int(k2)), 
                        '\'OPEN\'', '2*', str(0.15), '1*', str(5.0),   '  /']
            line = '  '.join(template)
            idx_counter += 1
            file.write(f'{line}\n')

        # write production wells
        for k1, k2 in perfs_prod:
            well_name = f'PRO-{name_idxs[idx_counter]}'
            # production template
            template = [f'\'{well_name}\'', '2*', str(int(k1)), str(int(k2)), 
                        '\'OPEN\'', '2*', str(0.15), '1*', str(5.0),   '  /']
            line = '  '.join(template)
            idx_counter += 1
            file.write(f'{line}\n')

        # End of the COMPDAT section
        file.write('/')

    if is_copy:
        write_path = f'{abs_to_src}/log_dir/INCLUDE'
    
    else:
        write_path = f'{abs_to_src}/model/INCLUDE'
    
    path_check(write_path)

    for keyword in keywords:
        if is_include:
            file_name = keyword.lower()
            file_path = f'{write_path}/{file_name}.inc'
        # TODO: write else and find keyword in PUNQS3 DATA

        if keyword == 'WELSPECS':
            if is_include:
                with open(file_path, 'w+') as welspecs:
                    write_welspecs(welspecs, locs_inj, locs_prod)
            # TODO: write else and change WELSPECS in PUNQS3 DATA

        if keyword == 'COMPDAT':
            if is_include:
                with open(file_path, 'w+') as compdat:
                    write_compdat(compdat, perfs_inj, perfs_prod)
            # TODO: write else and change COMPDAT in PUNQS3 DATA

def npv_calculator(
    model_name, 
    npv_constants, 
    SM3_to_STB=6.289810770432105
):
    """
    calculate NPV by parsing RSM file (resulted from a simulation run)

    The NPV formula is given by:
    \[
    NPV = \\sum_{t=1}^{T} \\frac{ (Q_o * r_o) - (Q_w * r_wp) + (Q_g * r_gp) - OPEX }{(1 + d)^t} - CAPEX
    \]
    
    where:
    - t: time (annular)
    - $Q_o$: Total Oil Field Flow Rate (production)
    - $Q_w$: Totla Water Field Flow Rate (production)
    - $Q_g$: TOtal Gas Field Flow Rate (production)
    
    constants to define (dict):
        ro: oil price ($/bbl)
        rgp: gas price ($/bbl)
        rwp: water production cost ($/bbl)
        d: annual discount rate (0<d<1)
        opex: operational expenditure ($)
        capex: capital expenditure ($)

    Args:
        model_name (str): name of .DATA model (without .DATA)
        npv_constants (dict): The constants to use in npv formula
        SM3_to_STB (float): unit converter, to convert SM# to STB, default = 6.289810770432105

    Return:
        npv (float): npv value
    """

    # Enter the RSM file path
    file_name = f'{model_name}.RSM'
    file_path = f'{abs_to_src}/model/{file_name}'

    # Read the RSM file and create DataFrame
    df = pd.read_fwf(file_path, header=1)

    # columns that can have specific units
    columns = ['FOPT', 'FGPT', 'FWPT']

    start_idx = 1

    # Define index of units row in df
    units_idx = 1
    # Get unit of each columns
    units = {}
    for col in columns:
        if '*' in str(df[col][units_idx]):
            start_idx = 2
            #  [1:] to convert '*10**3' to '10**3'
            units[col] = eval(df[col][units_idx][1:])
        else:
            units[col] = 1

    cum_FOPT, cum_FGPT, cum_FWPT = [], [], []

    # Initialize counter to convert until a year in NPV formula
    year_counter = 1
    npv = 0

    # Iterate over the data starting from the specified index
    for i in range(start_idx, len(df['TIME'])):        
        # check if reach the end of the year
        if float(df['TIME'][i]) >= year_counter * 365:
            # Accumulate yearly production values
            cum_FOPT.append(float(df['FOPT'][i]) * units['FOPT'])
            cum_FGPT.append(float(df['FGPT'][i]) * units['FGPT'])
            cum_FWPT.append(float(df['FWPT'][i]) * units['FWPT'])

            year_counter += 1

    cum_FOPT.append(float(df['FOPT'].iloc[-1]) * units['FOPT'])
    cum_FGPT.append(float(df['FGPT'].iloc[-1]) * units['FGPT'])
    cum_FWPT.append(float(df['FWPT'].iloc[-1]) * units['FWPT'])
            
    for i in range(len(cum_FOPT)):
        # Check if it is first year
        if i == 0:
            FOPT_year = cum_FOPT[0]
            FGPT_year = cum_FGPT[0]
            FWPT_year = cum_FWPT[0]
        
        # Obtaining Net Production for the year.
        else:
            FOPT_year = max(0, cum_FOPT[i] - cum_FOPT[i-1])
            FGPT_year = max(0, cum_FGPT[i] - cum_FGPT[i-1])
            FWPT_year = max(0, cum_FWPT[i] - cum_FWPT[i-1])

        # Calculate the Net Present Value (NPV) for the current year
        npv += (
            (FOPT_year * npv_constants['ro'] * SM3_to_STB) + 
            (FGPT_year * npv_constants['rgp'] * SM3_to_STB) - 
            (FWPT_year * npv_constants['rwp'] * SM3_to_STB) - 
            npv_constants['opex']*FOPT_year*SM3_to_STB
        )/((1 + npv_constants['d'])**(i+1))

    # Return npv after subtracing capex from it
    return (npv - npv_constants['capex']) / 10**6


def count_calls(obj_func):
    """
    A decorator that counts the number of function evaluation

    Args:
        obj_func (function): Objective function

    Returns:
        callable: The decorated function.

    Attributes:
        call_count (int): The number of times the decorated function has been called.
    """
    @functools.wraps(obj_func)
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        result = obj_func(*args, **kwargs)
        return result
    
    wrapper.call_count = 0
    return wrapper


@count_calls
def run_simulator():
    """
    Call and subprocess the .bat which will run the Eclipse for reservoir simulation

    Args:
        None
    
    Return:
        None
    """
    # Path to directory where include .bat and .DATA files
    working_dir = f'{abs_to_src}/model'

    # open a file to log the simulator report
    with open(f'{abs_to_src}/log_dir/bat_results.txt', 'a') as batch_outputs:
        subprocess.call([rf"{working_dir}/$MatEcl.bat"], stdout=batch_outputs, cwd=working_dir)


def tuning(
    optimizer,
    problem_dict,
    params_grid,
    mode='single',
    n_trials=2,
):
    """
    Perform hyperparameter tuning using a specified optimizer.

    Args:
    - optimizer (mealpy.Optimizer): The instance of a mealpy.Optimizer.
    - problem_dict (dict): A dictionary representing the optimization problem.
    - params_grid (dict): A dictionary containing the hyperparameter grid to search.
    - mode (str, optional): The tuning mode, either 'single', 'swarm', 'thread', 'process'. Default is 'single'.
    - n_trials (int, optional): The Number of trials on the problem. Default is 2.

    Returns:
    None

    """
    # Create a tuner object with the specified optimizer and hyperparameter grid.
    tuner = Tuner(
            algorithm=optimizer,
            param_grid=params_grid
        )

    # Execute the tuning process.
    tuner.execute(
            problem=problem_dict,
            n_trials=n_trials,
            mode=mode,
    )
    
    # Export the tuning results to a CSV file in 'src/tuning' directory.
    tuner.export_results(save_path=f'{abs_to_src}/tuning', file_name='tuning.csv')


def amend_position(position, lb, ub):
    return np.clip(position, lb, ub)
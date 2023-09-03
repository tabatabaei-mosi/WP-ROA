import subprocess
import shutil
from pathlib import Path
from datetime import datetime

from utils import path_check, write_solution, decode_solution
from constraints import physical_penalty

# Path of root directory (absolute to src)
abs_to_src = Path(__file__).resolve().parent

def bat_summary():
    """
    Summarize information from a log file and generate a summary report based on specific keywords.
    
    This function reads a log file containing information about simulation runs, searches for
    keywords ('Errors', 'Warnings', 'Problems'), and generates a summary report containing
    counts and details about these keywords.

    Returns:
        None
    """
    # List of keywords to search for in the result file
    keywords = ['Errors', 'Warnings', 'Problems']

    # Define the directory for log
    # ... Create the directory if it doesn't exist
    log_dir = f'{abs_to_src}/log_dir'
    path_check(log_dir)

    # Loop through keywords
    for keyword in keywords:
        # Initialize vaiables for count run and keyword_value > 0
        run_count = 0
        keyword_count = 0

        # Determine the write mode based on the keyword
        if keyword == 'Errors':
            write_mode = 'w+'
        
        else:
            write_mode = 'a'

        # Open result files for reading and and another file for writing
        with open(f'{log_dir}/bat_results.txt', 'r') as bat_result, open(f'{log_dir}/bat_summary.txt', write_mode) as bat_sum:
            # Write a seperator and keyword header
            bat_sum.write(f'--------------------------------------------------------------------\n')
            bat_sum.write(f'{keyword}\n')

            # Read each line in the result file
            for line in bat_result.readlines():
                # Check if the keyword is present in the line
                if keyword in line:
                    # Increment the total count of occurrences
                    run_count += 1

                    # Split the line to extract the keyword value
                    keyword_list = line.split()
                    keyword_value = int(keyword_list[1])

                    # Increment the count of non-zero occurrences
                    if keyword_value != 0:
                        keyword_count += 1
                        bat_sum.write(f'Call numebr {run_count} --> {keyword} : {keyword_value}\n')

            # Write the total count of the keyword occurrences and seperator
            bat_sum.write(f'Total {keyword} : {keyword_count}\n')
            bat_sum.write('--------------------------------------------------------------------\n')

            # Write the total number of simulation calls in last line of file
            if keyword == 'Problems':
                bat_sum.write(f'Total number of simulation calls : {run_count}\n')


def write_best(model_name,
               optimizer,
               best_solution, 
               best_fitness,
               num_inj=0, 
               num_prod=6, 
               n_params=4,
               gridsize=(19, 28, 5),
               keywords=['WELSPECS', 'COMPDAT']
    ):
    """
    Write optimization results and perform simulation with the best solution.

    This function takes the best solution obtained from an optimization process and performs
    several tasks related to result reporting and reservoir simulation.

    1- The function decodes the best solution to obtain well locations and perforations.
    2- Information about the optimization process, well parameters, and constraints is recorded.
    3- The function runs a simulation with the best solution and logs the output.
    4- It also copies simulation files to the log directory for reference.

    Args:
        model_name (str): Name of the reservoir simulation model
        optimizer (mealpy.Optimizer): An instance of the mealpy optimizer class.
        best_solution (np.array, list): The best solution obtained from the optimization process.
        best_fitness (float): The fitness value of the best solution.
        num_inj (int): The number of injection wells. Default: 0.
        num_prod (int): The number of production wells. Default: 6.
        n_params (int): The number of optimization parameters. Default is 4.
        gridsize (list, tuple): Grid size information as [X, Y, Z]. Default: [19, 28, 5].
        keywords (list, tuple): Keywords for solution writing. Default: ['WELSPECS', 'COMPDAT'].

    Returns:
        None
    """
    
    # Decode the best solution to obtain well locations and perforations
    locs_inj, perfs_inj, locs_prod, perfs_prod = decode_solution(
                                                    solution=best_solution,
                                                    num_inj=num_inj,
                                                    num_prod=num_prod,
                                                    n_params=n_params
                                                )
    
    
    # Define the directory for log files
    log_dir = f'{abs_to_src}/log_dir'

    # Create the log directory if it doesn't exist
    path_check(log_dir)

    # Open a text file in write mode ('w+')
    with open(f'{log_dir}/best_result.txt', 'w+') as best_file:

        # # Write a header indicating the purpose of this file and seperator
        best_file.write('Final solution (best solution) of WP optimization process\n')
        best_file.write(100*'-'+'\n')

        # Write the best fitness value in terms of NPV (Net Present Value).
        best_file.write(f'Best Objective value -->  NPV = {best_fitness/10**9} Bilion $\n\n')

        # Get Name and Hyperparameters of optimizer
        optimizer_name = optimizer.get_name()
        params = optimizer.get_parameters()
        params_line = ''

        # Write the name of optimizer and its hyperparameters 
        best_file.write(f'The {optimizer_name} parameters : \n')
        for name, value in params.items():
            params_line += f'{name} = {value}, '

        # Write hyperparameters line on another seperator
        best_file.write(f'{params_line[:-2]}\n')
        best_file.write(100*'-'+'\n')

        # Loop through production wells
        for i in range(num_prod):
            well_name = f'PRO-{i+1}'
            loc_i, loc_j = locs_prod[i][0], locs_prod[i][1]
            k1, k2 = perfs_prod[i][0], perfs_prod[i][1]

            # Write each production well line parameters
            best_file.write(
                f'> {well_name} -->  i = {loc_i}, j = {loc_j}, perf_start = {k1}, perf_end = {k2}\n')
            
        # Loop through injection wells
        for i in range(num_inj):
            well_name = f'INJ-{i+1}'
            loc_i, loc_j = locs_inj[i][0], locs_inj[i][1]
            k1, k2 = perfs_inj[i][0], perfs_inj[i][1]

            # Write each injection well line parameters
            best_file.write(
                f'> {well_name} -->  i = {loc_i}, j = {loc_j}, perf_start = {k1}, perf_end = {k2}\n')
            
        # Write another seperator again
        best_file.write(100*'-'+'\n')

    # Write physical constraints messages to best_result file
    physical_penalty(
                    model_name=model_name,
                    locs_inj=locs_inj,
                    perfs_inj=perfs_inj,
                    locs_prod=locs_prod,
                    perfs_prod=perfs_prod,
                    gridsize=gridsize,
                    final_solution=True
            )

    # Write simulation files for the final simulation run
    write_solution(
                locs_inj=locs_inj,
                perfs_inj=perfs_inj,
                locs_prod=locs_prod,
                perfs_prod=perfs_prod,
                keywords=keywords,
                is_green=True, is_include=True, is_copy=False
            )
        
    # Run simulation again with the best solution
    with open(f'{log_dir}/best_result.txt', 'a') as best_file:

        working_dir = f'{abs_to_src}/model'
        subprocess.call([rf"{working_dir}/$MatEcl.bat"], stdout=best_file, cwd=working_dir)

    # Copy simulatiom files to log directory
    write_solution(
            locs_inj=locs_inj,
            perfs_inj=perfs_inj,
            locs_prod=locs_prod,
            perfs_prod=perfs_prod,
            keywords=keywords,
            is_green=True, is_include=True, is_copy=True
        )


def save_charts(
        optimizer, 
        targets=['global_best_fitness', 'exploration_exploitation']
    ):
    """
    Save charts related to the optimization process.

    This function saves charts based on specified targets using the optimizer's history.

    Args:
        optimizer: The optimizer object for which to save charts.
        targets (list, optional): A list of targets to save charts for, each corresponding to a specific chart type.
            Defaults to ['global_best_fitness', 'exploration_exploitation'].

    Note:
        - The function dynamically generates method names based on the targets to call the corresponding
          chart-saving methods from the optimizer's history.
        - Charts are saved in the 'src/log_dir/Charts' directory with filenames corresponding to the targets.

    Returns:
        None
    """

    # Define the directory where charts will be saved
    chart_path = f'src/log_dir/Charts'
    path_check(chart_path)

    # Iterate through each target for chart saving
    for target in targets:
        
        # Generate the name of the method to call for saving chart based on target
        # ex. : optimizer.history.save_global_best_fitness_chart, target = 'global_best_fitness'
        method = f'save_{target}_chart'

        # Use getattr to dynamically access the method from the optimizer's history
        method_to_call = getattr(optimizer.history, method)

        try:
            # Check if the method is callable (i.e., exists and can be called)
            if callable(method_to_call):
                # Call the method with a specified filename to save the chart.
                method_to_call(filename=f'{chart_path}/{target}')

        except:
            pass
            
        # TODO: Handle any exceptions that may occur while calling the method.


def copy_to_history(optimizer):
    """
    Copy the log files of an optimization run process to the run history directory.
    
    Args:
        optimizer (mealpy.Optimizer): An instance of the mealpy optimizer class.
        
    Returns:
        None
        
    Notes:
        - The source log files are expected to be located in '{abs_to_src}/log_dir'.
        - The destination directory will be '{abs_to_src}/run_history/' followed by
          the optimizer's name and configuration parameters.
        - If the destination directory already exists, the copying process will not be
          performed.
    """
    
    # Define the source path where log files are located
    source_path = f'{abs_to_src}/log_dir'
    # Create the source path if not exists.
    path_check(source_path)
    
    # Get the optimizers hyperparameters
    params = optimizer.get_parameters()

    # Create a destination name for the copied directory based on optimizers name and parameters
    dest_name = f'{optimizer.get_name()}'

    for name, value in params.items():
        dest_name += f', {name}={value}'

    # Get the current date and time in the specified format
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Append the current time to the destination name
    dest_name += f', time={time_now}'

    # Define the destination path where the log files will be copied.
    dest_path = f'{abs_to_src}/run_history/{dest_name}'

    # Copy the entire source directory to the destination.
    shutil.copytree(source_path, dest_path)
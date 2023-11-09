import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from constraints import physical_penalty
from utils import decode_solution, path_check, write_solution

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
    # --- Create the directory if it doesn't exist
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

def write_best(model_name,
               optimizer,
               best_solution, 
               best_fitness,
               sim_call,
               num_inj=0, 
               num_prod=6, 
               n_params=4,
               gridsize=(19, 28, 5),
               keywords=['WELSPECS', 'COMPDAT'],
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
        best_file.write(f'Best Objective value -->  $ NPV = {best_fitness:.3f} B\n\n')

        # Get Name and Hyperparameters of optimizer
        optimizer_name = optimizer.get_name()
        params = optimizer.get_parameters()
        params_line = ''

        best_file.write(100*'-'+'\n')
        # Write the name of optimizer and its hyperparameters 
        best_file.write(f'The {optimizer_name} parameters : \n')
        for name, value in params.items():
            if name == 'pop_size':
                params_line += f'{name} = {value}'
                params_line += '\n'
                continue

            params_line += f'{name} = {value}, '


        # Calculate the runtime of optimization process
        runtime = sum(optimizer.history.list_epoch_time)
        time_string = time_report(runtime)

        # Get number of function evaluation 
        nfe = optimizer.problem.fit_func.call_count

        # Write hyperparameters line, runtime, nfe and another seperator
        best_file.write(f'{params_line[:-2]}\n')
        best_file.write(f'runtime = {time_string}\n')
        best_file.write(f'nfe = {nfe},  Number simulation calls = {sim_call}\n')
        best_file.write(100*'-'+'\n')

        # Write header for well parameters
        best_file.write('Parameters of each well : \n')
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
                method_to_call(filename=f'{chart_path}/{target}', verbose=False)

        except:
            pass
            
        # TODO: Handle any exceptions that may occur while calling the method.


def save_gbf(optimizer):
    """
    Save the global best fitness values and corresponding epochs to an Excel file.

    This function takes an optimizer object as input, extracts the global best fitness
    values and their corresponding epochs, and saves them to an Excel file.

    Args:
        optimizer (mealpy.Optimizer): An instance of the mealpy optimizer class.

    Returns:
        None
    """
    # Extract the global best fitness values and their corresponding epochs
    gbf_list = optimizer.history.list_global_best_fit
    epoch_list = [i for i in range(1, len(gbf_list) + 1)]

    # Create a DataFrame to store the data
    df = pd.DataFrame({'Epoch': epoch_list, 'GBF': gbf_list})

    # Define the directory where the Excel file will be saved
    gbf_dir = f'{abs_to_src}/log_dir/GBF'
    path_check(gbf_dir)

    # Save the DataFrame to an Excel file (gbf.xlsx) without including the index
    df.to_excel(f'{gbf_dir}/gbf.xlsx', index=False)


def simulation_info():
    """
    Extract and save simulation information from log files.

    This function reads a log file, searches for specific keywords (ERROR, WARNING, PROBLEM),
    and extracts related messages. It then saves this information to a new file.

    Returns:
        None
    """
    # Define the directory where log files are located
    log_dir = f'{abs_to_src}/log_dir'

    # List of keywords to search for in the log file
    keywords = ['ERROR', 'WARNING', 'PROBLEM']

    # Dictionary to store messages corresponding to each keyword
    messages = {
        'ERROR': [],
        'WARNING': [],
        'PROBLEM': []
    }

    # Open the log file for reading
    with open(f'{log_dir}/best_result.txt', 'r') as best_file:
        lines = best_file.readlines()

        # Loop through each keyword
        for keyword in keywords:
            for idx, line in enumerate(lines):
                message = ''
                # Check if the keyword is found in the line
                if f'--{keyword}' in line:
                    # Extract related messages until the next '@' symbol
                    for i in range(idx, idx+100):
                        if '@' in lines[i]:
                            message += f'{lines[i]}'
                        else:
                            break
                    
                    # Append the message to the corresponding keyword's list
                    messages[keyword].append(f'{message}\n')
        
        # Open a new file for saving simulation information
        with open(f'{log_dir}/simulation_info.txt', 'w+') as sim_info:
            # Write a header and a seperator
            sim_info.write('The final simulation run info\n')
            sim_info.write(70*'-' + '\n')
            # Loop through keywords
            for keyword in keywords:
                # Write the count of messages for each keyword
                sim_info.write(f'{keyword} count : {len(messages[keyword])}\n\n')

                # Write the messages for the keyword
                for message in messages[keyword]:
                    sim_info.write(message)

                # write a seperator line
                sim_info.write(70*'-' + '\n')
                

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


def track_solution(solution, nfe, sim_call, n_params=4):
    """
    Track the solution and log it to a file.

    This function takes a solution, the number of objective function calls, the number of
    simulation calls, and an optional number of decision variable. It logs the solution
    and relevant information to a text file.

    Args:
        solution (np.array): The solution to be tracked.
        nfe (int): Number of objective function calls.
        sim_call (int): Number of simulation calls.
        n_params (int, optional): Number of parameters per well. Defaults to 4.

    Returns:
        None
    """

    # Define the directory where log files are located
    log_dir = f'{abs_to_src}/log_dir'

    # Calculate the new shape for reshaping the solution
    new_shape = (int(len(solution)/n_params), n_params)

    # Reshape the solution as per the new shape and convert to integers
    reshaped_solution = solution.astype(int).reshape(new_shape)

    # Open the log file for appending
    with open(f'{log_dir}/track_solutions.txt', 'a') as track:
        # Write a header with objective function call and simulation call information
        track.write(f'Obj_func call {nfe} (simulation call : {sim_call}): \n')

        # Write a header for the well parameters
        track.write(f'  wellname  loc_i   loc_j  perf_k1  perf_k2\n')

        # Iterate over the reshaped solution and log each well's parameters
        for idx, well in enumerate(reshaped_solution):
            track.write(f'  Well {idx+1} :   ')
            row_str = '       '.join(map(str, well))
            track.write(row_str + '\n')

        # Write a separator line to distinguish between different logs
        track.write(50 * '-' + '\n')


def track_npv(sim_call, npv):
    """
    Track the Net Present Value (NPV) and log it to a file.

    This function takes a solution, the number of simulation calls, and the NPV value.
    It logs the NPV value to a text file along with relevant information.

    Args:
        sim_call (int): Number of simulation calls.
        npv (float): The Net Present Value (NPV) in billions of dollars.

    Returns:
        None
    """
    # Define the directory where log files are located
    log_dir = f'{abs_to_src}/log_dir'

    # Open the log file for appending
    with open(f'{log_dir}/track_npv.txt', 'a') as track:
        # Write a line containing the simulation call number and NPV value
        track.write(f'-Simulation call {sim_call} : NPV($ B) = {npv:.3f}\n')
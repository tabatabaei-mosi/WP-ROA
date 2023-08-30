from pathlib import Path
from utils import path_check

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
        # Initia
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
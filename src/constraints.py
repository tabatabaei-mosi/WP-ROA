import numpy as np

from pathlib import Path

# Path of root directory (absolute to src)
abs_to_src = Path(__file__).resolve().parent

def logical_constraint(
    locs_inj, perfs_inj,
    locs_prod, perfs_prod,
):
    """
    Check the logical constraints of well locations and perforations.
    If any logical constrain is violated, the NPV is equal to Zero as punishment for algorithm.
    
    This function verifies two logical constraints:
    1. Duplicate Well Locations: Ensures that the same well location is not repeated.
    2. Perforation Order: Checks if perforation 'start' is greater than 'end'.

    Args:
        locs_inj (np.array, list): Array of injection well locations, e.g., [[i1, j1], [i2, j2], ...].
        perfs_inj (np.array, list): Array of injection well perforations.
        locs_prod (np.array, list): Array of production well locations.
        perfs_prod (np.array, list): Array of production well perforations.

    Returns:
        bool: True if any logical constraint is violated, False if no problem.
        """
    # If there're injection wells, merge location and perforation arrays between production and injection.
    if locs_inj:
        locs = np.concatenate([locs_inj, locs_prod])
        perfs = np.concatenate([perfs_inj, perfs_prod])
        
    else:
        locs = locs_prod
        perfs = perfs_prod
    
    # Loop through wells to count the number of repeatation of a location
    # ... create a list from locs, to use buitl-in method of list, count().
    locs = [list(item) for item in locs]
    for loc in locs:
        loc_count = locs.count(list(loc))
        # a SINGLE repeatation, will terminated in NPV=0 (True, violated) 
        if loc_count > 1:
            return True

    # Loop through perforations
    for perf in perfs:
        # Check if 'start' greater than 'end'
        if perf[0] > perf[1]:
            return True   
    
    # No logical constraint violations found
    return False


def read_grdecl(
        model_name='PUNQS3', 
        gridsize=(19, 28, 5), 
        target='ACTCELL'
    ):
    """
    Read .GRDECL file and extract information about the target keyword (Default: ACTCELL)

    Args:
        model_name (str): Name of the model (without extension). Default: 'PUNQS3'
        gridsize (list or tuple): Grid demnsions, [x, y, z]. Default (PUNQS3): [19, 28, 5]
        target (str): Keyword to search for in the GRDECL file. Default (PUNQS3): 'ACTCELL'

    Retruns:
        target_value (np.array): An array contains values of target keywords. In default, activity status of each block will be generated.
                            0 represents Not Active, 1 represents Active.
    """
    # Constant variables for file name and file path to the .GRDECL file
    file_name = f'{model_name}.GRDECL'
    file_path = f'{abs_to_src}/model/{file_name}'

    # Open .GRDECL file
    with open(file_path, 'r') as grd:
        # get file lines
        lines = grd.readlines()

        # Find the line index where the target keyword is located
        for idx, line in enumerate(lines):
            if target in line:
                line_actcell = idx
                break
        
        # Caclculate the start and end line indices of keyword's values
        start_line = line_actcell + 1
        end_line = int(start_line + (gridsize[0]*gridsize[1]*gridsize[2])/6) + 1

        # Initialize indices to keep track of the current position in the 3D grid
        idx_x = 0
        idx_y = 0
        idx_z = 0

        # Create a numpy array to store ativity status of blocks, (0: NotActive, 1: Active)
        actcell = np.zeros((gridsize[0], gridsize[1], gridsize[2]))

        # Iterate through keyword value lines
        for i in range(start_line, end_line):
            # Split line individual value (each line has 6 value)
            line = lines[i].split()

            for actnum in line:
                # Reset x index and increment y index if we have processed a full row
                if idx_x == gridsize[0]:
                    idx_x = 0
                    idx_y += 1

                # Reset y index and increment z index if we have processed a full column
                if idx_y == gridsize[1]:
                    idx_y = 0
                    idx_z += 1

                # Break the loop if we have processed all layers
                if idx_z == gridsize[2]:
                    break
                
                # Store the activity status in the numpy array
                actcell[idx_x, idx_y, idx_z] = int(actnum)
                idx_x += 1

            # Break if each end of values
            if idx_z == gridsize[2]:
                break

        return actcell
    

def physical_penalty(
        model_name,
        locs_inj, perfs_inj,
        locs_prod, perfs_prod, 
        gridsize=(19, 28, 5), 
        targets=['null_block', 'min_space', 'border'],
        well_space=2,
        null_space=2,
        final_solution=False
    ):
    """
    Calculate physical penalties based on specified criteria for a reservoir simulation model. This function will check
    some target constrains, expected: ['null_block', 'min_space', 'border'].
    minimum space of wells and from null blocks to not be less than a threshold and additionally, to not well be drilled in boarders or in a null block. 

    Args:
        model_name (str): Name of the reservoir simulation model
        locs_inj (np.array, list): Array of injection well locations.
        perfs_inj (np.array, list): Array of injection well perforations .
        locs_prod (np.array, list): Array of production well locations.
        perfs_prod (np.array, list): Array of production well perforations.
        gridsize (list or tuple): Grid dimensions [x, y, z]. Default (PUNQS3): [19, 28, 5]
        targets (list): List of target constraints to check. Default: ['null_block', 'min_space', 'border']
        well_space (int): Minimum well spacing constraint. Default: 2
        null_space (int): Minimum distance to nulk blocks, Default: 2

    Returns:
        tuple: A tuple containing a boolean indicating constraint violation (True if violated), and
               the cumulative number of physical penalty faults.
    """
    # Initialize counters for different types of faults (counter of times a physical constrain will violate)
    null_fault = 0
    min_fault = 0
    border_fault = 0

    # Define a list to store constraints messages
    messages = []

    # Combine injection and production well locations and perforations
    if locs_inj:
        locs = np.concatenate([locs_inj, locs_prod])
        perfs = np.concatenate([perfs_inj, perfs_prod])
        
    else:
        locs = locs_prod
        perfs = perfs_prod

    # Check for border constraint
    if 'border' in targets:
        for i in range(len(locs)):
            # Check if the well is at the border of the grid
            if (locs[i][0] in [1, gridsize[0]]) or (locs[i][1] in [1, gridsize[1]]):
                # append message to messages
                messages.append(
                    f'!! Border: Well number {i+1} is located on the border of the reservoir\n')
                border_fault += 1

    # Check for minimum well spacing constraint
    if 'min_space' in targets:
        # Loop through all wells
        for i in range(len(locs)):
            well1_start = [locs[i][0], locs[i][1], perfs[i][0]]
            well1_end = [locs[i][0], locs[i][1], perfs[i][1]]
            
            # Loop through remaining wells
            for j in range(i+1, len(locs)):
                well2_start = [locs[j][0], locs[j][1], perfs[j][0]]   
                well2_end = [locs[j][0], locs[j][1], perfs[j][1]]   

                # Calculate distances between start points and end points of 2 wells
                dist_start = np.sqrt((well1_start[0] - well2_start[0])**2 + 
                                     (well1_start[1] - well2_start[1])**2 +
                                     (well1_start[2] - well2_start[2])**2)
                
                dist_end = np.sqrt((well1_end[0] - well2_end[0])**2 + 
                                   (well1_end[1] - well2_end[1])**2 +
                                   (well1_end[2] - well2_end[2])**2)
                
                # Check if minimum spacing constraint is violated
                if dist_start <= well_space or dist_end <= well_space:
                    # append message to messages
                    messages.append(
                        f'!! Well space : Distance between well {i+1} and well {j+1} is less than the minimum well spacing\n')
                    min_fault += 1

    # Check for null blocks constraint
    if 'null_block' in targets:
        # Read grid status and pass to actcell array using read_grdecl
        actcell = read_grdecl(
                    model_name=model_name, 
                    gridsize=gridsize, 
                    target='ACTCELL'
                )
        
        # Loop through wells to check null block constraint
        for i in range(len(locs)):
            loc_i = locs[i][0] - 1
            loc_j = locs[i][1] - 1
            perf_start = perfs[i][0] - 1
            perf_end = perfs[i][1] - 1

            # Check if any perforation in null block, Return True -> NPV = 0
            if any(actcell[loc_i, loc_j, perf_start: perf_end + 1] == 0):
                return True, 0
            
            # Define start and end point of well
            well_start = [loc_i, loc_j, perf_start]
            well_end = [loc_i, loc_j, perf_end]

            flag_null = False
            # Checking a specific radius of null blocks based on null_space.
            for x in range(loc_i - null_space, loc_i + null_space + 1):
                # Checking the legal range of x
                if x < 0 or x >= gridsize[0]:
                    continue
                    
                for y in range(loc_j - null_space, loc_j + null_space + 1):
                    # Checking the legal range of y
                    if y < 0 or y >= gridsize[1]:
                        continue
                        
                    for z in range(perf_start - null_space, perf_end + null_space + 1):
                        # Checking the legal range of x
                        if z < 0 or z >= gridsize[2]:
                            continue
                        status = actcell[x, y, z]
                    
                        if status == 0:
                            # Calculate distances to determine null block violations
                            dist_start = np.sqrt((well_start[0] - x)**2 + 
                                                 (well_start[1] - y)**2 + 
                                                 (well_start[2] - z)**2)

                            dist_end = np.sqrt((well_end[0] - x)**2 + 
                                               (well_end[1] - y)**2 + 
                                               (well_end[2] - z)**2)
                            
                            # Check if null space constraint is violated
                            if dist_start <= null_space or dist_end <= null_space:
                                # append message to messages
                                messages.append(
                                    f'!! Null blocks : The distance between well number {i+1} and null block ({x, y, z}) is less than the minimum allowable distance\n')
                                flag_null = True
                                null_fault += 1
                                break
                    if flag_null:
                        break
                if flag_null:
                    break
        
    # Calculate total faults by summing up the individual errors
    faults = null_fault + min_fault + border_fault

    # Check for final_solution: if True write in best_result.txt
    if final_solution:

        # best_result file path
        best_log = f'{abs_to_src}/log_dir/best_result.txt'

        # Open file to add constraints messages to it
        with open(best_log, 'a') as best_file:

            # Write seperator and constraints header
            best_file.write(100*'-'+'\n')
            best_file.write('Physical Constraints: \n\n')

            # Check if any fault is equal to zero, write a No problem message
            if border_fault == 0:
                best_file.write('! Border: No well on border\n\n')

            if min_fault == 0:
                best_file.write('! Well space : Not problem in Well space\n\n')

            if null_fault == 0:
                best_file.write('! Null blocks : No well is near null blocks\n\n')

            # Loop through messages to write each message in file
            for message in messages:
                best_file.write(message)

            # Write a final seperator
            best_file.write(100*'-'+'\n')


    # Return a tuple with violation status (False if no violation) and total faults
    return False, faults

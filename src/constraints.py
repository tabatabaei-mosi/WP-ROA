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
    # Merge location and perforation arrays
    if locs_inj != []:
        locs = np.concatenate([locs_inj, locs_prod])
        perfs = np.concatenate([perfs_inj, perfs_prod])
        
    else:
        locs = locs_prod
        perfs = perfs_prod

    # Loop through wells to caculate count of each well
    locs = [list(item) for item in locs]
    for loc in locs:
        loc_count = locs.count(list(loc))
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
        gridsize=[19, 28, 5], 
        target='ACTCELL'
    ):
    """
    Read .GRDECL file and extract ActiveCells information.

    Args:
        model_name (str): Name of the model (without extension). Default: 'PUNQS3'
        gridsize (list or tuple): Grid demnsions, [x, y, z]. Default (PUNQS3): [19, 28, 5]
        target (str): Keyword to search for in the file. Default (PUNQS3): 'ACTCELL'

    Retruns:
        actcell (np.array): Array containinh activity status of each block.
                            0 represents Not Active, 1 represents Active
    """
    # Constant file name and file path for the .GRDECL file
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
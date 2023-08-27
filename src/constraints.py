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
def split_solution(solution, num_inj=0, n_params=4):
    """
    split the solution to injection wells and production wells

    Args:
        solution (list): a solution that generated by algorithm
        num_inj (int): number of injection wells, default = 0
        n_parmas (int): number of each well parameters, default = 4

    Return:
        param_inj (list): including injection wells parameters
        param_prod (list): including production wells parameters
    """
    # split the whole solution list to 2 parts, injections and productions
    param_inj = solution[:num_inj*n_params]
    param_prod = solution[num_inj*n_params:]
    return param_inj, param_prod


def decode_solution(solution, num_inj=0, num_prod=1, n_params=4):
    """
    decode solution to 4 parts: locations of injections, perforations of injection, locations of productions and perforations of productions

    Args:
        solution (list): a solution that generated by algorithm
        num_inj (int): number of injection wells, default = 0
        num_prod (int): number of production wells, default = 1
        n_parmas (int): number of each well parameters, default = 4

    Return:
        locs_inj (list): including locations of injections, ex. : [[i1, j1], [i2, j2]]
        perfs_inj (list): including perforations of injections
        locs_prod (list): including locations of productions
        perfs_prod (list): including perforations of productions
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

    # a for loop to append each injection well params
    for i in range(num_inj):
        locs_inj.append(param_inj[start:start+slice_loc])
        perfs_inj.append(param_inj[start+slice_loc:start+slice_perf])
        start += n_params

    # store final locations and perforations of productions
    locs_prod = []
    perfs_prod = []

    start = 0
    # a for loop to append each production well params
    for i in range(num_prod):
        locs_prod.append(param_prod[start:start+slice_loc])
        perfs_prod.append(param_prod[start+slice_loc:start+slice_perf])
        start += n_params

    return locs_inj, perfs_inj, locs_prod, 

def write_solution(solution, keywords, num_inj=0, num_prod=1, n_params=4, is_green=True, is_include=True):
    """
    a function to write the solution on the DATA file to simulating

    Args:
        solution (list): a solution that generated by algorithm
        keywords (list): including names of keywords
        num_inj (int): number of injection wells, default = 0
        num_prod (int): number of production wells, default = 1
        n_parmas (int): number of each well parameters, default = 4
        is_green (bool): True, if reservoir is clear and Fulse, if reservoir has wells 
        is_include (bool): True, if we use INCLUDE in DATA file and Fulse, if we write the wells params in original DATA file

    Return:
        None
    """
    # a function for write the WELSPECS keyword
    def write_welspecs(file, locs_inj, locs_prod):
        """
        a function to write the WELSPECS keyword

        Args:
            file (file): an open file to write
            locs_inj (list): including locations of injections
            locs_prod (list): including locations of productions
        """
        file.write(f'{keyword}\n')

        # PUNQS3-real properties
        name_idxs = [1, 4, 5, 11, 12, 15]
        RDBHP = [2362.2, 2373.0, 2381.7, 2386.0, 2380.5, 2381.0]

        # index for well names and RDBHP
        idx_counter = 0

        # write injection wells
        for i, j in locs_inj:
            well_name = f'INJ-{name_idxs[idx_counter]}'
            idx_counter += 1
            # injection welspecs template
            template = [f'\'{well_name}\'', '\'G1\'', str(int(i)), str(int(j)), str(RDBHP[idx_counter]),
                         '\'WATER\'', '1*', '\'STD\'', '3*', '\'SEG\'', '  /']
            line = '  '.join(template)
            idx_counter += 1
            file.write(f'{line}\n')

        # write production wells
        for i, j in locs_prod:
            well_name = f'PRO-{name_idxs[idx_counter]}'
            # production template
            template = [f'\'{well_name}\'', '\'G1\'', str(int(i)), str(int(j)), str(RDBHP[idx_counter]),
                         '\'OIL\'', '1*', '\'STD\'', '3*', '\'SEG\'', '  /']
            line = '  '.join(template)
            idx_counter += 1
            file.write(f'{line}\n')
        file.write('/')

    # a function for write the COMPDAT keyword
    def write_compdat(file, perfs_inj, perfs_prod):
        """
        a function to write the COMPDAT keyword

        Args:
            file (file): an open file to write
            perfs_inj (list): including perforations of injections
            perfs_prod (list): including perforations of productions
        """
        file.write(f'{keyword}\n')
        name_idxs = [1, 4, 5, 11, 12, 15]
        idx_counter = 0

        # write injection wells
        for k1, k2 in perfs_inj:
            if k1 > k2:
                k1, k2 = k2, k1
            well_name = f'INJ-{name_idxs[idx_counter]}'
            # injection tmplate
            template = [f'\'{well_name}\'', '2*', str(int(k1)), str(int(k2)), 
                        '\'OPEN\'', '2*', str(0.15), '1*', str(5.0),   '  /']
            line = '  '.join(template)
            idx_counter += 1
            file.write(f'{line}\n')

        # write production wells
        for k1, k2 in perfs_prod:
            if k1 > k2:
                k1, k2 = k2, k1
            well_name = f'PRO-{name_idxs[idx_counter]}'
            # production template
            template = [f'\'{well_name}\'', '2*', str(int(k1)), str(int(k2)), 
                        '\'OPEN\'', '2*', str(0.15), '1*', str(5.0),   '  /']
            line = '  '.join(template)
            idx_counter += 1
            file.write(f'{line}\n')

        file.write('/')

    # get locations and perforarions of wells fron decode_solution
    locs_inj, perfs_inj, locs_prod, perfs_prod = decode_solution(solution, num_inj=num_inj, 
                                                                 num_prod=num_prod, n_params=n_params)
    for keyword in keywords:
        if is_include:
            file_name = keyword.lower()
            file_path = f'src/Eclipse/PUNQS3-real/INCLUDE/{file_name}.inc'
        # TODO: write else and read keyword from PUNQS3-real DATA

        if keyword == 'WELSPECS':
            if is_include:
                with open(file_path, 'w+') as welspecs:
                    write_welspecs(welspecs, locs_inj, locs_prod)
            # TODO: write else and change PUNQS3-real DATA

        if keyword == 'COMPDAT':
            if is_include:
                with open(file_path, 'w+') as compdat:
                    write_compdat(compdat, perfs_inj, perfs_prod)
            # TODO: write else and change COMPDAT in PUNQS3-real DATA

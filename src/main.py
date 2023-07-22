import numpy as np
from mealpy.optimizer import Optimizer


class ROA(Optimizer):
    """
    The developed version: Rain Optimization Algorithm (ROA)

    References
    ~~~~~~~~~~
    [1] Ali Reza Moazzeni, Ehsan Khamehchi 2020. Rain optimization algorithm (ROA): A new metaheuristic method 
    for drilling optimization solutions, DOI: https://doi.org/10.1016/j.petrol.2020.107512
    """
    def __init__(self, epoch, pop_size,
                init_radius, joint_size, rain_speed, soil_adsorption,
                **kwargs):
        """
        Initialize the Rain Optimization Algorithm (ROA) optimizer.

        Args:
            epoch (int): Maximum number of iterations.
            pop_size (int): Population size.
            init_radius (float): Initial radius of raindrops.
            joint_size (float): Joint size for merging raindrops.
            rain_speed (float): Rain speed for moving raindrops. usually [1E-6, 1E-3]
            soil_adsorption (float): Soil adsorption constant for raindrops absorption.

        Returns:
            None
        """
        super().__init__(**kwargs)

        #Check the validation range for each hyper-parameters
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [10, 10000])
        self.init_radius = self.validator.check_float('init_radius', init_radius)
        self.joint_size = self.validator.check_float('joint_size', joint_size)
        self.rain_speed = self.validator.check_float('rain_speed', rain_speed)
        self.soil_adsorption = self.validator.check_float('soil_adsorption', soil_adsorption)

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True
        # Determine to sort the problem or not in each epoch
        ## if True, the problem always sorted with fitness value increase
        ## if False, the problem is not sorted

    def initialize_variables(self):
        """
        initilize some supporter variables before initilize the population

        """
        # Generate a radius of first population
        self.radius = np.full(self.pop_size, self.init_radius)


    def initialization(self):
        """
        initilize guess and Generate random population

        """
        # Required code of mealpy Optimizer
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)
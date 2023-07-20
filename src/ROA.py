import numpy as np
from mealpy.optimizer import Optimizer

class ROA(Optimizer):
    def __init__(self, epoch, pop_size, init_radius, joint_size, rain_speed, soil_adsorption, **kwargs):
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
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [10, 10000])
        self.init_radius = self.validator.check_float('init_radius', init_radius)
        self.joint_size = self.validator.check_int('joint_size', joint_size)
        self.rain_speed = self.validator.check_float('rain_speed', rain_speed)
        self.soil_adsorption = self.validator.check_float('soil_adsorption', soil_adsorption)

    def initialization(self):
        """
        initilize guess and Generate random population

        """
        # Required code of mealpy Optimizer
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)

        # Generate a position, radius and joint sizes of first population
        self.position = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dimension))
        self.radius = np.full(self.pop_size, self.init_radius)
        self.size = np.full(self.pop_size, self.joint_size)
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
        self.size = np.full(self.pop_size, self.joint_size)


    def initialization(self):
        """
        initilize guess and Generate random population

        """
        # Required code of mealpy Optimizer
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)


    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for i in range(self.pop_size):
            for j in range(self.problem.n_dims):

                # Change each variable Xi to Xi + R and Xi - R and evaluate the new position by the objective function
                ## Change Xi to Xi + R
                new_position_1 = np.copy(self.pop[i][self.ID_POS])
                new_position_1[j] += self.radius[i]
                new_cost_1 = self.problem.fit_func(new_position_1)

                ## Change Xi to Xi - R
                new_position_2 = np.copy(self.pop[i][self.ID_POS])
                new_position_2[j] -= self.radius[i]
                new_cost_2 = self.problem.fit_func(new_position_2)

                # if the new cost is smaller than the previous cost, accept a new position for pop[i]
                if new_cost_1 < self.pop[i][self.ID_TAR][self.ID_FIT]:
                    self.pop[i][self.ID_POS] = new_position_1
                    self.pop[i][self.ID_TAR][self.ID_FIT] = new_cost_1

                if new_cost_2 < self.pop[i][self.ID_TAR][self.ID_FIT]:
                    self.pop[i][self.ID_POS] = new_position_2
                    self.pop[i][self.ID_TAR][self.ID_FIT] = new_cost_2

            
            while True:
                # move the droplet at the same direction with the same velocity
                new_position = self.pop[i][self.ID_POS] + np.random.uniform(-self.size[i], self.size[i], self.problem.n_dims)
                new_position = np.clip(new_position, self.problem.lb, self.problem.ub)
                new_cost = self.problem.fit_func(new_position)

                if new_cost < self.pop[i][self.ID_TAR][self.ID_FIT]:
                    self.pop[i][self.ID_POS] = new_position
                    self.pop[i][self.ID_TAR][self.ID_FIT] = new_cost
                
                else:
                    break
                
                # reduce size of droplet depending on the soil adsorption properties
                self.size[i] *= self.soil_adsorption
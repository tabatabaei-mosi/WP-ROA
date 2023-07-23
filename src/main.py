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
                init_radius, joint_size, rain_speed, soil_adsorption, r_min,
                **kwargs):
        """
        Initialize the Rain Optimization Algorithm (ROA) optimizer.

        Args:
            epoch (int): Maximum number of iterations. 
            pop_size (int): Population size.
            init_radius (float): Initial radius of raindrops.
            joint_size (float): Joint size for merging raindrops.
            rain_speed (float): Rain speed for moving raindrops.
            soil_adsorption (float): Soil adsorption constant for raindrops absorption.
            r_min (float): Minimum possible radius for droplet.

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
        self.r_min = self.validator.check_float('r_min', r_min)

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

                # compare new cost with current cost
                if new_cost < self.pop[i][self.ID_TAR][self.ID_FIT]:
                    self.pop[i][self.ID_POS] = new_position
                    self.pop[i][self.ID_TAR][self.ID_FIT] = new_cost
                    
                else:
                    break
                
                # reduce size of droplet depending on the soil adsorption properties
                self.size[i] *= self.soil_adsorption

            # join near droplets to each other and change the size of new droplets
            for j in range(self.problem.n_dims):
                if i == j:
                    continue
                r1 = self.radius[i]
                r2 = self.radius[j]
                dist = np.linalg.norm(self.pop[i][self.ID_POS] - self.pop[j][self.ID_POS])

                if dist <= (r1 + r2):
                    R = (r1**self.problem.n_dims + r2**self.problem.n_dims)**(1/self.problem.n_dims)

                    if r1 > r2:
                        self.pop[j][self.ID_POS] = self.pop[i][self.ID_POS]
                        self.pop[j][self.ID_TAR][self.ID_FIT] = self.pop[i][self.ID_TAR][self.ID_FIT]
                        self.radius[j] = R
                        self.size[j] = self.joint_size

                    else:
                        self.pop[i][self.ID_POS] = self.pop[j][self.ID_POS]
                        self.pop[i][self.ID_TAR][self.ID_FIT] = self.pop[j][self.ID_TAR][self.ID_FIT]
                        self.radius[i] = R
                        self.size[i] = self.joint_size

        # define the cost array from self.pop for use in Omit weak droplets
        cost = []
        for i in range(self.pop_size):
            cost.append(np.array(self.pop)[:, self.ID_TAR][i][self.ID_FIT])
        cost = np.array(cost)

        # Omit weak droplets depending on soil adsorption
        weak_indices = np.where((self.soil_adsorption * self.radius**self.problem.n_dims)**(1/self.problem.n_dims) < self.r_min)[0]
        self.pop = np.delete(self.pop, weak_indices, axis=0)
        self.radius = np.delete(self.radius, weak_indices)
        self.size = np.delete(self.size, weak_indices)
        self.pop_size -= len(weak_indices)
        cost = np.delete(cost, weak_indices)

        sorted_indices = np.argsort(cost)
        self.radius = self.radius[sorted_indices]
        self.size = self.size[sorted_indices]

        # Generate new droplets depending on rain speed
        num_new_droplets = max(1, int(self.rain_speed * self.pop_size))
        new_droplets = self.create_population(num_new_droplets)
        self.pop = np.vstack((self.pop, new_droplets))
        self.radius = np.hstack((self.radius, np.full(num_new_droplets, self.init_radius)))
        self.size = np.hstack((self.size, np.full(num_new_droplets, self.joint_size)))
        self.pop_size += num_new_droplets

# test algorithm
if __name__ == '__main__':
    # define objective function
    def fitness(solution):
        return np.sum(solution**2)

    problem_dict = {
        "fit_func": fitness, # objective function 
        "lb": [-5, ] * 5,  # Lower bounds for X and Y
        "ub": [5, ] * 5,   # Upper bounds for X and Y
        "n_dims": 5,  # number of variables (dimension)
    }

    epoch = 50    # number of iterations
    pop_size = 100    # population size
    init_radius = 0.01       # initial radius of droplets
    joint_size = 1   # joint size
    rain_speed = 0.03    # speed of rain
    soil_adsorption =  1    # rate of soil adsorption
    r_min = 0.001   # minimmum radius of droplets

    roa = ROA(epoch, pop_size, init_radius, joint_size, rain_speed, soil_adsorption, r_min)
    best_pop, best_fitness = roa.solve(problem_dict)
    print(f"Solution: {best_pop}, Fitness: {best_fitness}")
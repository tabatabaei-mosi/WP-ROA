import numpy as np
from mealpy.optimizer import Optimizer


class BaseROA(Optimizer):
    """
    The modifiedd version: Base Rain Optimization Algorithm (ROA)

    References
    ~~~~~~~~~~
    [1] Ali Reza Moazzeni, Ehsan Khamehchi 2020. Rain optimization algorithm (ROA): 
    A new metaheuristic method for drilling optimization solutions.
    DOI: https://doi.org/10.1016/j.petrol.2020.107512
    """
    def __init__(self, 
                epoch=100, pop_size=50,
                init_radius=1, joint_size=1, rain_speed=2, soil_adsorption=2,
                **kwargs
                ):
        """
        Initialize the Rain Optimization Algorithm (ROA) optimizer.

        Args:
            epoch (int): Maximum number of iterations. 
            pop_size (int): Population size.
            init_radius (float): Initial radius of raindrops. default = 0.01
            joint_size (float): Joint size for merging raindrops. default = 1
            rain_speed (float): Rain speed for moving raindrops. [0, 1]
            soil_adsorption (float): Soil adsorption constant for raindrops absorption (percentage). [0, 100]

        Returns:
            None
        """
        super().__init__(**kwargs)

        # Check the validation range for each hyper-parameter
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int('pop_size', pop_size, [10, 10000])
        self.init_radius = self.validator.check_float('init_radius', init_radius)
        self.joint_size = self.validator.check_float('joint_size', joint_size)
        self.rain_speed = self.validator.check_float('rain_speed', rain_speed)
        self.soil_adsorption = self.validator.check_float('soil_adsorption', soil_adsorption)

        self.set_parameters(["epoch", "pop_size", "init_radius", "joint_size", "rain_speed", 'soil_adsorption'])

        # Determine to sort the problem or not in each epoch
        ## if True, the problem always sorted with fitness value increase
        ## if False, the problem is not sorted
        self.sort_flag = True

    def initialize_variables(self):
        """
        Initilize some supporting variables before initilizing the population.

        """
        # Generate the radius of first population
        self.radius = np.full(self.pop_size, self.init_radius)
        self.size = np.full(self.pop_size, self.joint_size)

    def initialization(self):
        """
        Initialize guess and Generate random population.

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
        if epoch > 0:
            # Define sorted indices for sorting the population (considering minmax setting)
            if self.problem.minmax == 'min':
                sorted_indices = np.argsort(
                    [self.pop[i][self.ID_TAR][self.ID_FIT] for i in range(self.pop_size)])
            else:
                sorted_indices = np.argsort(
                    [-self.pop[i][self.ID_TAR][self.ID_FIT] for i in range(self.pop_size)])        

            # Sort the population based on fitness
            self.radius = self.radius[sorted_indices]
            self.size = self.size[sorted_indices]

            # Reduce the population size based on rain speed
            self.pop_size = max(1, int(self.pop_size - self.rain_speed))

            # Update radius, size, and population size
            self.radius = self.radius[:self.pop_size]
            self.size = self.size[:self.pop_size]
            self.pop = self.pop[:self.pop_size]


            # Generate new droplets depending on rain speed
            num_new_droplets = max(1, int(self.rain_speed))
            new_droplets = self.create_population(num_new_droplets)
            self.pop = np.vstack((self.pop, new_droplets))
            self.radius = np.hstack((self.radius, np.full(num_new_droplets, self.init_radius)))
            self.size = np.hstack((self.size, np.full(num_new_droplets, self.joint_size)))
            self.pop_size += num_new_droplets

        # List to store the indices of droplets to be removed
        to_delete = []
        for i in range(self.pop_size):
            for j in range(self.problem.n_dims):

                # Change each variable Xi to Xi + R and Xi - R and evaluate the new position by the objective function
                # Change Xi to Xi + R
                new_position_1 = np.copy(self.pop[i][self.ID_POS])
                new_position_1[j] += self.radius[i]
                new_position_1 = self.problem.amend_position(new_position_1, self.problem.lb, self.problem.ub)
                
                # Change Xi to Xi - R
                new_position_2 = np.copy(self.pop[i][self.ID_POS])
                new_position_2[j] -= self.radius[i]
                new_position_2 = self.problem.amend_position(new_position_2, self.problem.lb, self.problem.ub)

                # Evaluate the new targets
                new_target_1 = self.get_target_wrapper(new_position_1)
                new_target_2 = self.get_target_wrapper(new_position_2)

                # Update the population if the new cost is better
                self.pop[i] = self.get_better_solution(self.pop[i], [new_position_1, new_target_1])
                self.pop[i] = self.get_better_solution(self.pop[i], [new_position_2, new_target_2])


            while True:
                # Move the droplet in the same direction with the same velocity
                new_position = self.pop[i][self.ID_POS] + \
                    np.random.uniform(-self.size[i], 
                                      self.size[i], self.problem.n_dims)
                new_position = self.problem.amend_position(
                    new_position, self.problem.lb, self.problem.ub)
                
                # Calculate the new cost of new position
                new_target = self.get_target_wrapper(new_position)
                new_pop = [new_position, new_target]

                # Compare the new agent with current agent by fitness
                if self.compare_agent(new_pop, self.pop[i]):
                    self.pop[i] = new_pop
                else:
                    break
                
                # Reduce size of droplet depending on the soil adsorption properties
                R = (self.soil_adsorption * self.radius[i]**self.problem.n_dims)**(1/self.problem.n_dims)
                self.radius[i] = R

                # join near droplets to each other and change the size of new droplets
                for j in range(self.pop_size):
                    if i == j or (i or j in to_delete):
                        continue
                    r1 = self.radius[i]
                    r2 = self.radius[j]
                    dist = np.linalg.norm(
                        self.pop[i][self.ID_POS] - self.pop[j][self.ID_POS])

                    if dist <= (r1 + r2):
                        R = (r1 ** self.problem.n_dims + 
                             r2 ** self.problem.n_dims) ** (1 / self.problem.n_dims)
                        
                        # Compare the agents for find the weak one and delete it
                        if self.compare_agent(self.pop[i], self.pop[j]):
                            # Update the radius of stronger
                            self.radius[i] = R
                            to_delete.append(j)

                        else:
                            # Update the radius of stronger
                            self.radius[j] = R  
                            to_delete.append(i)

        # Remove marked droplets
        self.pop = np.delete(self.pop, to_delete, axis=0)
        self.radius = np.delete(self.radius, to_delete)
        self.size = np.delete(self.size, to_delete)
        self.pop_size = len(self.pop) 



class OriginalROA(Optimizer):
    """
    The developed version: Original Rain Optimization Algorithm (ROA)

    References
    ~~~~~~~~~~
    [1] Ali Reza Moazzeni, Ehsan Khamehchi 2020. Rain optimization algorithm (ROA): A new metaheuristic method 
    for drilling optimization solutions, Appendix A - MATLAB code, DOI: https://doi.org/10.1016/j.petrol.2020.107512
    """

    def __init__(self, epoch=100, pop_size=50,
                 init_radius=0.05, joint_size=1, rain_speed=10, soil_adsorption=10,
                 **kwargs):
        """
        Initialize the Rain Optimization Algorithm (ROA) optimizer.

        Args:
            epoch (int): Maximum number of iterations. 
            pop_size (int): Population size.
            init_radius (float): Initial radius of raindrops. [0, 1]
            joint_size (float): Joint size for merging raindrops. default = 1
            rain_speed (float): Rain speed for moving raindrops. [0, 100]
            soil_adsorption (float): Soil adsorption constant for raindrops absorption. [0, 100]

        Returns:
            None
        """
        super().__init__(**kwargs)

        # Check the validation range for each hyper-parameters
        self.epoch = self.validator.check_int('epoch', epoch, [1, 100000])
        self.pop_size = self.validator.check_int(
            'pop_size', pop_size, [10, 10000])
        self.init_radius = self.validator.check_float(
            'init_radius', init_radius, [0, 1])
        self.joint_size = self.validator.check_float('joint_size', joint_size)
        self.rain_speed = self.validator.check_float('rain_speed', rain_speed, [0, 100])
        self.soil_adsorption = self.validator.check_float(
            'soil_adsorption', soil_adsorption, [0, self.pop_size])
        
        # Set the name of parameters
        self.set_parameters(["epoch", "pop_size", "init_radius", "joint_size", "rain_speed", 'soil_adsorption'])

        # Determine to sort the problem or not in each epoch
        # if True, the problem always sorted with fitness value increase
        # if False, the problem is not sorted
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def initialize_variables(self):
        """
        initilize some supporter variables before initilize the population

        """
        # Generate a radius of first population
        self.radius = np.full(self.pop_size, self.init_radius)
        self.direction = np.full(self.pop_size, self.joint_size)

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
        # Omit waek droplets
        if epoch > 1:
            self.pop_size = int(max(1, self.pop_size - self.soil_adsorption))
            self.pop = self.pop[:self.pop_size]

        # Generate new droplets depending on rain speed
        num_new_droplets = int(max(1, self.rain_speed))
        new_droplets = self.create_population(num_new_droplets)
        self.pop = np.vstack((self.pop, new_droplets))
        self.radius = np.hstack(
            (self.radius, np.full(num_new_droplets, self.init_radius)))
        self.direction = np.hstack(
            (self.direction, np.full(num_new_droplets, self.joint_size)))
        self.pop_size += num_new_droplets

        for i in range(self.pop_size):
            for j in range(self.problem.n_dims):

                # Change each variable Xi to Xi + R and Xi - R and evaluate the new position by the objective function
                # Change Xi to Xi + R
                new_position_1 = np.copy(self.pop[i][self.ID_POS])
                new_position_1[j] += self.radius[i]
                new_position_1 = np.clip(
                    new_position_1, self.problem.lb, self.problem.ub)

                # Change Xi to Xi - R
                new_position_2 = np.copy(self.pop[i][self.ID_POS])
                new_position_2[j] -= self.radius[i]
                new_position_2 = np.clip(
                    new_position_2, self.problem.lb, self.problem.ub)

                new_cost_1 = self.problem.fit_func(new_position_1)
                new_cost_2 = self.problem.fit_func(new_position_2)

                # if the new cost is smaller than the previous cost, accept a new position for pop[i]
                if self.problem.minmax == 'min':
                    if new_cost_1 < new_cost_2:
                        x2 = new_position_1[j]
                        y2 = new_cost_1
                        # self.pop[i][self.ID_POS] = new_position_1
                        # self.pop[i][self.ID_TAR][self.ID_FIT] = new_cost_1

                    else:
                        x2 = new_position_2[j]
                        y2 = new_cost_2
                        # self.pop[i][self.ID_POS] = new_position_2
                        # self.pop[i][self.ID_TAR][self.ID_FIT] = new_cost_2

                elif self.problem.minmax == 'max':
                    if new_cost_1 > new_cost_2:
                        x2 = new_position_1[j]
                        y2 = new_cost_1
                        # self.pop[i][self.ID_POS] = new_position_1
                        # self.pop[i][self.ID_TAR][self.ID_FIT] = new_cost_1

                    else:
                        x2 = new_position_2[j]
                        y2 = new_cost_2
                        # self.pop[i][self.ID_POS] = new_position_2
                        # self.pop[i][self.ID_TAR][self.ID_FIT] = new_cost_2
                self.direction[i] = (abs(
                    y2 - self.pop[i][self.ID_TAR][self.ID_FIT])/abs(x2 - self.pop[i][self.ID_POS][j]))

            dir_size = np.linalg.norm(self.direction[i])
            self.direction[i] /= dir_size
            popm_pos = np.copy(self.pop[i][self.ID_POS])
            popm_pos = self.pop[i][self.ID_POS] - \
                (self.radius[i] * self.direction[i])
            popm_cost = self.problem.fit_func(popm_pos)

            if self.problem.minmax == 'min':
                if popm_cost < self.pop[i][self.ID_TAR][self.ID_FIT]:
                    self.pop[i][self.ID_POS] = np.copy(popm_pos)
                    while popm_cost < self.pop[i][self.ID_TAR][self.ID_FIT]:
                        popm_pos = self.pop[i][self.ID_POS].copy()
                        popm_pos = self.pop[i][self.ID_POS] - \
                            (self.radius[i] * self.direction[i])
                        popm_cost = self.problem.fit_func(popm_pos)
                        if popm_cost < self.pop[i][self.ID_TAR][self.ID_FIT]:
                            self.pop[i][self.ID_POS] = popm_pos
                            self.pop[i][self.ID_TAR][self.ID_FIT] = popm_cost

                elif popm_cost > self.pop[i][self.ID_TAR][self.ID_FIT]:
                    self.radius[i] *= 0.5

                else:
                    self.radius[i] *= 2

            if self.problem.minmax == 'max':
                if popm_cost > self.pop[i][self.ID_TAR][self.ID_FIT]:
                    self.pop[i][self.ID_POS] = np.copy(popm_pos)
                    while popm_cost > self.pop[i][self.ID_TAR][self.ID_FIT]:
                        popm_pos = self.pop[i][self.ID_POS].copy()
                        popm_pos = self.pop[i][self.ID_POS] - \
                            (self.radius[i] * self.direction[i])
                        popm_cost = self.problem.fit_func(popm_pos)
                        if popm_cost > self.pop[i][self.ID_TAR][self.ID_FIT]:
                            self.pop[i][self.ID_POS] = popm_pos
                            self.pop[i][self.ID_TAR][self.ID_FIT] = popm_cost

                elif popm_cost < self.pop[i][self.ID_TAR][self.ID_FIT]:
                    self.radius[i] *= 0.5

                else:
                    self.radius[i] *= 2

            # join near droplets to each other and change the size of new droplets
            for j in range(self.pop_size):
                if i == j:
                    continue
                r1 = np.sqrt(self.problem.n_dims*(self.radius[i]**2))
                r2 = np.sqrt(self.problem.n_dims*(self.radius[j]**2))
                dist = np.linalg.norm(
                    self.pop[i][self.ID_POS] - self.pop[j][self.ID_POS])

                if dist <= (r1 + r2):
                    R = (self.radius[i]**self.problem.n_dims + self.radius[j]
                         ** self.problem.n_dims)**(1/self.problem.n_dims)

                    if self.problem.minmax == 'min':
                        if self.pop[i][self.ID_TAR][self.ID_FIT] < self.pop[j][self.ID_TAR][self.ID_FIT]:
                            self.pop[j][self.ID_POS] = self.pop[i][self.ID_POS]
                            self.pop[j][self.ID_TAR][self.ID_FIT] = self.pop[i][self.ID_TAR][self.ID_FIT]
                            self.radius[j] = R
                            self.direction[j] = self.direction[i]

                        else:
                            self.pop[i][self.ID_POS] = self.pop[j][self.ID_POS]
                            self.pop[i][self.ID_TAR][self.ID_FIT] = self.pop[j][self.ID_TAR][self.ID_FIT]
                            self.radius[i] = R
                            self.direction[i] = self.direction[j]

                    if self.problem.minmax == 'max':
                        if self.pop[i][self.ID_TAR][self.ID_FIT] > self.pop[j][self.ID_TAR][self.ID_FIT]:
                            self.pop[j][self.ID_POS] = self.pop[i][self.ID_POS]
                            self.pop[j][self.ID_TAR][self.ID_FIT] = self.pop[i][self.ID_TAR][self.ID_FIT]
                            self.radius[j] = R
                            self.direction[j] = self.direction[i]

                        else:
                            self.pop[i][self.ID_POS] = self.pop[j][self.ID_POS]
                            self.pop[i][self.ID_TAR][self.ID_FIT] = self.pop[j][self.ID_TAR][self.ID_FIT]
                            self.radius[i] = R
                            self.direction[i] = self.direction[j]


if __name__ == '__main__':
    # define objective function
    def fitness(solution):
        return np.sum(np.abs((solution**2) - (2*solution) - 3) - solution)

    problem_dict = {
        "fit_func": fitness,  # objective function
        "lb": [-1.1, ] * 3,  # Lower bounds for X and Y
        "ub": [4, ] * 3,   # Upper bounds for X and Y
        'minmax': 'min'   # Minimization or Maximization
    }

    epoch = 50    # number of iterations
    pop_size = 100    # population size

    roa = BaseROA(epoch=epoch, pop_size=pop_size)
    best_position, best_fitness = roa.solve(problem_dict)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")
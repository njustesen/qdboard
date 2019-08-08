import numpy as np
import math
import numpy


class Dimension:

    def __init__(self, name, min_value, max_value):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value

    def to_json(self):
        return {
            'name': self.name,
            'min_value': self.min_value,
            'max_value': self.max_value
        }


class Archive:

    def __init__(self, dimensions, cells, solutions):
        self.cells = cells
        self.dimensions = dimensions
        self.solutions = solutions
        self.fitnesses = [solution.fitness for solution in solutions]

    def to_json(self):
        return {
            'cells': [cell.to_json() for cell in list(self.cells.values())],
            #'solutions': [solution.to_json() for solution in self.solutions],
            'dimensions': [dim.to_json() for dim in self.dimensions],
            'fitness_std': np.std(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_mean': np.mean(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_min': np.min(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_max': np.max(self.fitnesses) if len(self.fitnesses) > 0 else None
        }


class Cell:

    def __init__(self, points, solutions):
        self.points = points
        self.solutions = solutions
        self.fitnesses = [solution.fitness for solution in solutions]

    def add_solution(self, solution):
        self.solutions.append(solution)
        self.fitnesses.append(solution.fitness)

    def to_json(self):
        return {
            'points': self.points,
            'solutions': [solution.to_json() for solution in self.solutions],
            'fitness_mean': np.mean(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_std': np.std(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_min': np.min(self.fitnesses) if len(self.fitnesses) > 0 else None,
            'fitness_max': np.max(self.fitnesses) if len(self.fitnesses) > 0 else None
        }


class Solution:

    def __init__(self, genotype, behavior, fitness):
        self.genotype = genotype
        self.behavior = behavior
        self.fitness = fitness

    def to_json(self):
        return {
            'genotype': self.genotype,
            'behavior': self.behavior,
            'fitness': self.fitness
        }


class QDAlgorithm:

    def __init__(self, run_id, config, b_dimensions, problem):
        self.run_id = run_id
        self.config = config
        self.b_dimensions = b_dimensions
        self.b_dims = len(self.b_dimensions)
        self.problem = problem
        self.b_mins = [dimension.min_value for dimension in self.b_dimensions]
        self.b_maxs = [dimension.max_value for dimension in self.b_dimensions]

    def start(self):
        raise NotImplementedError("Must be overridden by sub-class")

    def stop(self):
        raise NotImplementedError("Must be overridden by sub-class")

    def get_archive(self):
        raise NotImplementedError("Must be overridden by sub-class")

    def is_done(self):
        raise NotImplementedError("Must be overridden by sub-class")

    def to_json(self):
        return {
            'run_id': self.run_id,
            'b_dimensions': [dim.to_json() for dim in self.b_dimensions],
            'problem': self.problem.to_json(),
            'is_done': self.is_done(),  # Fix this
            'b_mins': self.b_mins,
            'b_maxs': self.b_maxs
        }


class Problem:

    def __init__(self, name, x_dims, b_dims, min_fit, max_fit, continuous=True, x_min=0, x_max=1):
        self.name = name
        self.x_dims = x_dims
        self.b_dims = b_dims
        self.continuous = continuous
        self.x_min = x_min
        self.x_max = x_max
        self.min_fit = min_fit
        self.max_fit = max_fit

    def evaluate(self, genotype):
        raise NotImplementedError("Must be overridden by sub-class")

    def to_json(self):
        return {
            'name': self.name,
            'x_dims': self.x_dims,
            'b_dims': self.b_dims,
            'continous': self.continuous,
            'x_min': self.x_min,
            'x_max': self.x_max,
            'min_fit': self.min_fit,
            'max_fit': self.max_fit
        }


class Rastrigin(Problem):

    def __init__(self, x_dims, b_dims, min_fit=-100, max_fit=0):
        super().__init__(f"Rastrigin-{x_dims}D-{b_dims}D", x_dims, b_dims, min_fit, max_fit)

    def evaluate(self, genotype):
        fitness = self.__rastrigin(genotype)
        behavior = genotype[:2]
        return Solution(genotype, behavior, fitness)

    def __rastrigin(self, xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f

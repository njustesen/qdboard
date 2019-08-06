import numpy as np


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
        self.solutions = solutions
        self.fitness_mean = np.mean([solution.fitness for solution in self.solutions])
        self.fitness_max = np.max([solution.fitness for solution in self.solutions])
        self.fitness_min = np.min([solution.fitness for solution in self.solutions])

    def to_json(self):
        return {
            'cells': [cell.to_json() for cell in self.cells],
            'solutions': [solution.to_json() for solution in self.solutions],
            'fitness_mean': self.fitness_mean,
            'fitness_min': self.fitness_min,
            'fitness_max': self.fitness_max
        }


class Cell:

    def __init__(self, points, solutions):
        self.points = points
        self.solutions = solutions
        self.fitnesses = [solution.fitness for solution in solutions]

    def to_json(self):
        return {
            'points': self.points,
            'solutions': [solution.to_json() for solution in self.solutions],
            'fitness_mean': np.mean(self.fitnesses),
            'fitness_std': np.std(self.fitnesses),
            'fitness_min': np.min(self.fitnesses),
            'fitness_max': np.max(self.fitnesses)
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

    def __init__(self, run_id, config):
        raise NotImplementedError("Must be overridden by sub-class")

    def start(self):
        raise NotImplementedError("Must be overridden by sub-class")

    def stop(self, config):
        raise NotImplementedError("Must be overridden by sub-class")

    def load(self):
        raise NotImplementedError("Must be overridden by sub-class")

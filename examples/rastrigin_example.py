#!/usr/bin/env python3

import math
import uuid
import qdboard.server as server
from qdboard.model import Problem, Dimension, Solution
from qdboard.algos.map_elites import MapElites
from qdboard.api import add_run


class Rastrigin(Problem):

    def __init__(self, x_dims, b_dims, min_fit=-100, max_fit=0):
        super().__init__(f"Rastrigin-{x_dims}D-{b_dims}D", x_dims, b_dims, min_fit, max_fit)

    def evaluate(self, genotype):
        fitness = self.__rastrigin(genotype)
        behavior = genotype[:2]
        return Solution(uuid.uuid1(), genotype, behavior, phenotype=genotype, fitness=fitness, img=None)

    def __rastrigin(self, xx):
        x = xx * 10.0 - 5.0
        f = 10 * x.shape[0]
        for i in range(0, x.shape[0]):
            f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
        return -f

config = {
    "cvt_samples": 25000,
    "batch_size": 100,
    "random_init": 1000,
    "random_init_batch": 100,
    "sigma_iso": 0.01,
    "sigma_line": 0.2,
    "dump_period": 100,
    "parallel": True,
    "cvt_use_cache": True,
    "archive_path": "/Users/njustesen/git/qdboard/qdboard/map-elites/runs/",
    "centroids_path": "/Users/njustesen/git/qdboard/qdboard/map-elites/centroids/",
    "num_niches": 5000,
    "num_gens": 10000
}

x_dims = 6
b_dims = 2
b_dimensions = [Dimension(str(i+1), 0, 1) for i in range(b_dims)]
problem = Rastrigin(x_dims, b_dims)
algo = MapElites(str(uuid.uuid1()), config, b_dimensions=b_dimensions, problem=problem)
add_run(algo)
algo.start()

# Run server
server.start_server(debug=True, use_reloader=False)

"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains functions to communicate with a game host to manage games.
"""

import pickle
import uuid
from qdboard.model import *
from qdboard.algos.map_elites import MapElites, MapElitesProxy

runs = {}

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
    "archive_path": "/Users/noju/qdboard/map-elites/runs/",
    "centroids_path": "/Users/noju/qdboard/map-elites/centroids/",
    "num_niches": 5000,
    "num_gens": 10000
}

x_dims = 6
b_dims = 2
b_dimensions = [Dimension(str(i+1), 0, 1) for i in range(b_dims)]
problem = Rastrigin(x_dims, b_dims)

algo = MapElitesProxy("1", config, b_dimensions=b_dimensions, problem=problem)
runs["1"] = algo
algo.start()


def create_run(name, config, dimensions, problem):
    run_id = str(uuid.uuid1())
    if name.lower() == "map-elites":
        run = MapElitesProxy(run_id, config, dimensions, problem)
        runs[run_id] = run
    else:
        raise SyntaxError(name + "is not a known algorithm")
    return run


def get_runs():
    return [run for key, run in runs.items()]


def get_archive(run_id):
    if run_id not in runs:
        raise Exception(f"Run not found with id {run_id}")
    return runs[run_id].get_archive

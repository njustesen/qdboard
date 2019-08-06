"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains functions to communicate with a game host to manage games.
"""

import pickle
import uuid
from qdboard.model import Run, QDAlgorithm, MAPElites

runs = {}


def create_run(name, config):
    run_id = str(uuid.uuid1())
    if name.lower() == "map-elites":
        run = MapElites(run_id, config)
    else:
        raise SyntaxError(name + "is not a known algorithm")
    r = Run(run_id, run.config)
    runs.append


def get_runs():
    runs
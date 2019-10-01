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
from qdboard.algos.map_elites import MapElites

# Store runs in memory
runs = {}


def add_run(algorithm):
    runs[algorithm.run_id] = algorithm


def remove_run(run_id):
    if run_id in runs:
        del runs[run_id]


def create_run(algorithm):
    runs[algorithm.run_id] = algorithm


def get_runs():
    return [run for key, run in runs.items()]


def get_run(run_id):
    return runs[run_id]


def get_archive(run_id):
    if run_id not in runs:
        raise Exception(f"Run not found with id {run_id}")
    return runs[run_id].get_archive()

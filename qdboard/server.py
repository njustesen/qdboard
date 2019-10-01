"""
==========================
Author: Niels Justesen
Year: 2018
==========================
Run this script to start a Flask server locally. The server will start a Host, which will manage games.
"""

from flask import Flask, request, render_template
from qdboard import api
import json
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/runs/create', methods=['PUT'])
def create():
    data = json.loads(request.data)
    run = api.create_run(data['name'], data['config'])
    return json.dumps(run.to_json())


@app.route('/runs/', methods=['GET'])
def get_all_runs():
    runs = api.get_runs()
    run_list = [run.to_json() for run in runs]
    return json.dumps(run_list)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


@app.route('/runs/<run_id>/archive', methods=['GET'])
def get_archive(run_id):
    return json.dumps(api.get_archive(run_id).to_json(), cls=NpEncoder)


@app.route('/runs/<run_id>', methods=['GET'])
def get(run_id):
    return json.dumps(api.get_run(run_id).to_json())


def start_server(debug=False, use_reloader=False):
    
    # Change jinja notation to work with angularjs
    jinja_options = app.jinja_options.copy()
    jinja_options.update(dict(
        block_start_string='<%',
        block_end_string='%>',
        variable_start_string='%%',
        variable_end_string='%%',
        comment_start_string='<#',
        comment_end_string='#>'
    ))
    app.jinja_options = jinja_options

    app.config['TEMPLATES_AUTO_RELOAD']=True
    app.run(debug=debug, use_reloader=use_reloader)

import json
import os
from os.path import join
from os.path import expanduser as expnd

cwd = os.path.dirname(os.path.realpath(__file__))

with open(join(cwd, 'config.json')) as json_file:
    config = json.load(json_file)

for key in config.keys():
    config[key] = expnd(config[key])

input_data_dir = None
submissions_dir = None
models_dir = None

vars().update(config)

import numpy as np
import cv2
from config import Config
import os
import argparse
import json
import pandas as pd

def heatmap(map):
    map = (map*255).astype(np.uint8)
    return cv2.applyColorMap(map, cv2.COLORMAP_BONE)


def get_npy_folds(path):
    return list(zip(*np.load(path)))


def get_csv_folds(path, d):
    df = pd.read_csv(path)
    df = df[['id', 'fold']]
    train = [[], [], [], [], []]
    test = [[], [], [], [], []]

    folds = {}
    for i in range(5):
        fold_ids = list(df[df['fold'].isin([i])]['id'])
        folds.update({i: [n for n, l in enumerate(d) if l.split('_')[0] in fold_ids]})

    for k, v in folds.items():
        for i in range(5):
            if i != k:
                train[i].extend(v)
        test[k] = v

    return list(zip(np.array(train), np.array(test)))

def get_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        config['fold'] = None
    with open(os.path.join('..', '..', 'config', 'config.json'), 'r') as f:
        second_config = json.load(f)
    config['dataset_path'] = second_config['input_data_dir']
    config['models_dir'] = second_config['models_dir']
    print(config)
    return Config(**config)

import numpy as np
from collections import defaultdict, OrderedDict
import os
from PIL import Image
import json
import pandas as pd

def generate_folds():
    with open(os.path.join('..', '..', 'config', 'config.json'), 'r') as f:
        config = json.load(f)
    TRAIN_DATA = os.path.join(config['input_data_dir'], 'train_masks')
    train_files = os.listdir(TRAIN_DATA)
    train_ids = {s[:-12] for s in train_files}

    squares_fma = {}
    squares_f04a = {}
    for car_idx in train_ids:
        fma = 0
        for view_idx in range(1, 17, 1):
            im = np.asarray(Image.open(os.path.join(TRAIN_DATA, car_idx + "_" + str(view_idx).zfill(2) + "_mask.gif")))
            if view_idx == 5:
                squares_f04a[car_idx] = np.mean(im)
            fma += np.mean(im)
        squares_fma[car_idx] = fma

    squares_fma = OrderedDict(sorted(squares_fma.items(), key=lambda x: x[1]))
    squares_f04a = OrderedDict(sorted(squares_f04a.items(), key=lambda x: x[1]))
    fma_rows = [(i, square * 100, fold % 5) for fold, (i, square) in enumerate(squares_fma.items())]
    f04a_rows = [(i, square * 100, fold % 5) for fold, (i, square) in enumerate(squares_f04a.items())]

    fma_df = pd.DataFrame(fma_rows, columns=['id', 'square', 'fold'])
    f04a_df = pd.DataFrame(f04a_rows, columns=['id', 'square', 'fold'])

    fma_df.to_csv('../fma.csv', index=False)
    f04a_df.to_csv('../f04a.csv', index=False)

if __name__ == "__main__":
    generate_folds()
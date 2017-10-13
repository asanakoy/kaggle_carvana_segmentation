import cv2
import os
import numpy as np
from multiprocessing import Pool, freeze_support
from functools import partial


def folds_mean(roots, prob_file):
    ims = []
    for fold in range(5):
        for r in roots:
            prob_path = os.path.join(r, 'fold{}_'.format(fold) + prob_file)
            im = cv2.imread(prob_path, cv2.IMREAD_GRAYSCALE)
            ims.append(im)
    mean = (np.mean(ims, axis=0)).astype(np.uint8)
    cv2.imwrite(os.path.join('..', 'results', 'albu27.09', prob_file), mean)


def parallel_mean():
    root = r'../results'
    os.makedirs(os.path.join(root, 'albu27.09'), exist_ok=True)
    roots = [
        os.path.join(root, 'fma_s44'),
        os.path.join(root, 'fma_s88'),
        os.path.join(root, 'fma_noseed'),
        os.path.join(root, 'f04a_clahe'),
        os.path.join(root, 'f04a_s44'),
        os.path.join(root, 'f04a_clahe_rmsprop')
    ]
    prob_files = os.listdir(roots[0])
    unfolded = {f[6:] for f in prob_files if f.startswith('fold')}
    print(len(unfolded))
    f = partial(folds_mean, roots)
    with Pool() as pool:
        pool.map(f, unfolded)

if __name__ == "__main__":
    freeze_support()
    parallel_mean()

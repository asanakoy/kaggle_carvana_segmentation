import argparse
from os.path import join
import pandas as pd
import numpy as np
import glob
from joblib import Parallel
from joblib import delayed
from sklearn.utils import gen_even_slices
from tqdm import tqdm
from PIL import Image

import config


def get_areas_df(masks_dir, car_ids, is_quiet=False):
    areas = np.zeros((len(car_ids), 16))
    for i, car_id in tqdm(enumerate(car_ids),
                          desc='Calc areas',
                          disable=is_quiet,
                          total=len(car_ids)):
        for j in xrange(1, 17):
            p = join(masks_dir, '{}_{:02d}_mask.gif'.format(car_id, j))
            mask = Image.open(p)
            mask = np.asarray(mask.convert('L'))
            assert mask is not None, p
            assert len(mask.shape) == 2
            assert mask.max() == 255
            areas[i, j - 1] = mask.sum() / 255

    df = pd.DataFrame(index=car_ids, data=areas, columns=['view_{:02d}'.format(x) for x in xrange(1, 17)])
    return df


if __name__ == '__main__':
    """
    Calculate car areas on train images. Useful for splitting cars on folds.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--n_jobs', type=int, default=1, metavar='N',
                        help='number of parallel jobs')
    args = parser.parse_args()

    masks_dir = join(config.input_data_dir, 'train_masks')

    paths = glob.glob1(masks_dir, '*_mask.gif')
    print 'Num images:', len(paths)
    car_ids = np.unique(map(lambda x: x[:-len('_01_mask.gif')], paths))
    print 'Num unique images:', len(car_ids)
    assert len(paths) % len(car_ids) == 0

    ret = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(get_areas_df)(masks_dir, car_ids[s], is_quiet=(i > 0))
        for i, s in enumerate(gen_even_slices(len(car_ids), args.n_jobs)))

    df = pd.concat(ret, axis=0)
    df['sum'] = df.values.sum(axis=1)
    output_path = join(config.input_data_dir, 'areas_df.hdf5')
    df.to_hdf(output_path, 'df', mode='w')
    print 'Train car areas were successfully calculated'

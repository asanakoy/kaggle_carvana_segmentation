import argparse
import glob
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from os.path import join
import cv2
from pathlib2 import Path
from joblib import Parallel
from joblib import delayed
from sklearn.utils import gen_even_slices

import config
from data_utils import rle_encode
from data_utils import rle_to_string
from dataset import CARVANA


def load_from_files(test_image_paths, output_dir=None, is_quiet=False):
    all_rles = []
    all_img_filenames = []

    for sample_name in tqdm(test_image_paths, desc='Read files', disable=is_quiet):
        sample_name = Path(sample_name).stem
        probs_img = cv2.imread(str(output_dir.joinpath(sample_name + '.png')),
                               cv2.IMREAD_GRAYSCALE)
        assert probs_img is not None, sample_name
        mask = (probs_img >= 128)
        rle = rle_to_string(rle_encode(mask))
        all_rles.append(rle)
        all_img_filenames.append(sample_name + '.jpg')

    img_idx = map(os.path.basename, all_img_filenames)
    df = pd.DataFrame(index=img_idx, data={'img': img_idx, 'rle_mask': all_rles})
    return df


def average_from_files(test_image_paths, probs_dirs, output_dir, should_save_masks=True, is_quiet=False):
    for dir_path in probs_dirs:
        if not dir_path.exists():
            raise ValueError('{} not found'.format(dir_path))
    if should_save_masks:
        output_dir.mkdir(exist_ok=True)

    all_rles = []
    all_img_filenames = []
    for sample_name in tqdm(test_image_paths, desc='Avg files', disable=is_quiet):
        sample_name = Path(sample_name).stem

        probs = None
        for dir_path in probs_dirs:
            mask_img = cv2.imread(str(dir_path.joinpath(sample_name + '.png')), cv2.IMREAD_GRAYSCALE)
            assert mask_img is not None, sample_name
            mask_img = mask_img.astype(np.float32)

            if probs is None:
                probs = mask_img
            else:
                probs += mask_img
        probs /= len(probs_dirs)  # values from 0 to 255

        # save to disk
        prob_img = np.asarray(np.round(probs), dtype=np.uint8)
        if should_save_masks:
            cv2.imwrite(str(output_dir.joinpath(sample_name + '.png')), prob_img)

        mask = (probs >= 128)
        rle = rle_to_string(rle_encode(mask))
        all_rles.append(rle)
        all_img_filenames.append(sample_name + '.jpg')

    img_idx = map(os.path.basename, all_img_filenames)
    df = pd.DataFrame(index=img_idx, data={'img': img_idx, 'rle_mask': all_rles})
    return df


def create_submission(df, output_path):
    print('Create submission...')
    sample_subm = pd.read_csv(join(config.input_data_dir, 'sample_submission.csv'))
    assert len(df) == len(sample_subm), 'wrong len'
    assert sorted(sample_subm.img.values) == sorted(df.img.values), 'img names differ!'
    df.to_csv('{}.gz'.format(output_path), index=False, compression='gzip')
    print 'Saved'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--n_jobs', type=int, default=1, metavar='N',
                        help='number of parallel jobs')
    parser.add_argument('--load', action='store_true',
                        help='load pregenerated probs from folder?')
    parser.add_argument('--net_name', choices=['scratch', 'vgg11v1'])
    args = parser.parse_args()
    print 'config.submissions_dir', config.submissions_dir

    if args.net_name == 'vgg11v1':
        probs_dirs = list()
        for fold_id in xrange(7):
            dirs = glob.glob(join(config.submissions_dir,
                                  'test_probs_vgg11v1_s1993_im1024_gacc1_aug1_v2fold{}.7_noreg_epoch*'.format(fold_id)))
            epochs = map(lambda x: int(x.rsplit('_epoch', 1)[1]), dirs)
            last_epoch_dir = sorted(zip(epochs, dirs))[-1][1]
            probs_dirs.append(last_epoch_dir)
        print map(lambda x: os.path.basename(x), probs_dirs)
        output_dir = Path(config.submissions_dir) / ('test_vgg11v1_final')

    elif args.net_name == 'scratch':
        probs_dirs = list()
        for fold_id in xrange(7):
            dirs = glob.glob(join(config.submissions_dir,
                                  'test_probs_scratch_s1993_im1024_aug1_fold{}.7_epoch*'.format(fold_id)))
            epochs = map(lambda x: int(x.rsplit('_epoch', 1)[1]), dirs)
            last_epoch_dir = sorted(zip(epochs, dirs))[-1][1]
            probs_dirs.append(last_epoch_dir)
        print map(lambda x: os.path.basename(x), probs_dirs)
        output_dir = Path(config.submissions_dir) / ('test_scratch2')
    else:
        raise ValueError('Unknown net_name {}'.format(args.net_name))

    probs_dirs = map(Path, probs_dirs)
    with open(str(output_dir) + '.txt', mode='w') as f:
        f.write('Following models were averaged:\n')
        for l in probs_dirs:
            f.write(str(l) + '\n')
    test_pathes = CARVANA.get_test_paths(is_hq=True)

    print 'Reading from', map(str, probs_dirs)
    print 'output_dir', output_dir

    if not args.load:
        fd = delayed(average_from_files)
        ret = Parallel(n_jobs=args.n_jobs, verbose=0)(
            fd(test_pathes[s], probs_dirs=probs_dirs, output_dir=output_dir, is_quiet=(i > 0))
            for i, s in enumerate(gen_even_slices(len(test_pathes), args.n_jobs)))
    else:
        fd = delayed(load_from_files)
        ret = Parallel(n_jobs=args.n_jobs, verbose=0)(
            fd(test_pathes[s], output_dir=output_dir, is_quiet=(i > 0))
            for i, s in enumerate(gen_even_slices(len(test_pathes), args.n_jobs)))

    df = pd.concat(ret, axis=0)

    output_path = str(output_dir) + '.csv'
    create_submission(df, str(output_path))


if __name__ == '__main__':
    main()

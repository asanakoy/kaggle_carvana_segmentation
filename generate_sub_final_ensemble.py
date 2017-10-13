import argparse
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
from asanakoy.data_utils import rle_encode
from asanakoy.data_utils import rle_to_string
from asanakoy.dataset import CARVANA


def biggest_contour(im):
    im2, contours, hierarchy = cv2.findContours(np.copy(im), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, maxarea = None, 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > maxarea:
            biggest, maxarea = cnt, a
    return biggest


def check_if_top_is_unreliable(mean_pred, albu_pred):
    unreliable = np.zeros_like(albu_pred)
    rows, cols = unreliable.shape
    unreliable[(albu_pred > 30) & (albu_pred < 210)] = 255
    unreliable = cv2.erode(unreliable, (55, 55), iterations=10)
    unreliable = unreliable[0:rows//2, ...]
    biggest = biggest_contour(unreliable)
    if cv2.contourArea(biggest) > 40000:
        result = (mean_pred > 127).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(biggest)
        x, y, w, h = max(x - 5, 0), y - 5, w + 10, h + 10
        mask = (albu_pred > 7).astype(np.uint8) * 255
        result[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
        return result
    return None


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


def average_from_files(test_image_paths, probs_dirs, output_dir, should_save_masks=True,
                       is_quiet=False):
    for dir_path, w in probs_dirs:
        if not dir_path.exists():
            raise ValueError('{} not found'.format(dir_path))
    output_dir.mkdir(exist_ok=True)

    all_rles = []
    all_img_filenames = []
    for sample_name in tqdm(test_image_paths, desc='Avg files', disable=is_quiet):
        albu_prediction = None
        sample_name = Path(sample_name).stem

        probs = None
        for dir_path, weight in probs_dirs:
            assert 0 <= weight <= 1.0, weight
            mask_img = cv2.imread(str(dir_path.joinpath(sample_name + '.png')),
                                  cv2.IMREAD_GRAYSCALE)
            if 'albu' in dir_path:
                albu_prediction = np.copy(mask_img)
            assert mask_img is not None, sample_name
            mask_img = mask_img.astype(np.float32)

            if probs is None:
                probs = mask_img * weight
            else:
                probs += mask_img * weight
        assert probs.max() <= 256, probs.max()
        probs = np.clip(probs, 0, 255)
        prob_img = np.asarray(np.round(probs), dtype=np.uint8)

        fixed_top = check_if_top_is_unreliable(probs, albu_prediction)
        if fixed_top is not None:
            prob_img = fixed_top

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
    print 'Saved submission file in {}'.format('{}.gz'.format(output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--n_jobs', type=int, default=1, metavar='N',
                        help='number of parallel jobs')
    parser.add_argument('--load', action='store_true',
                        help='load pregenerated probs from folder?')
    parser.add_argument('--no_save', action='store_true',
                        help='not save probs as pngs?')

    args = parser.parse_args()

    probs_dirs = [
        ('test_scratch2', 1.0),
        ('test_vgg11v1_final', 1.0),
        ('albu27.09', 1.0),
        ('ternaus/ternaus_sep27', 1.0),
    ]
    w_sum = sum([x[1] for x in probs_dirs])
    print 'W_sum=', w_sum
    probs_dirs = map(lambda x: (Path(join(config.submissions_dir, x[0])), float(x[1]) / w_sum), probs_dirs)
    print 'Weights:', [x[1] for x in probs_dirs]
    output_dir = Path(config.submissions_dir) / ('ens_scratch2(1)_v1-final(1)_al27(1)_te27(1)')

    with open(str(output_dir) + '.txt', mode='w') as f:
        f.write('Following models were averaged:\n')
        for l, w in probs_dirs:
            f.write(str(l) + '; weight={}\n'.format(w))
            print str(l.stem) + '; weight={}\n'.format(w)
    print '===='
    test_pathes = CARVANA.get_test_paths(is_hq=True)

    print 'Reading from', map(str, probs_dirs)
    print 'output_dir', output_dir

    if not args.load:
        fd = delayed(average_from_files)
        ret = Parallel(n_jobs=args.n_jobs, verbose=0)(
            fd(test_pathes[s], probs_dirs=probs_dirs,
               output_dir=output_dir, is_quiet=(i > 0),
               should_save_masks=not args.no_save)
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

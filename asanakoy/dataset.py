import os
from os.path import isfile, join
from PIL import Image
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from pathlib2 import Path

import config


def get_stratified_by_area_folds(car_ids, n_splits, fold_id, random_state, min_bin_size=7):
    """
    Sample stratified folds accounting on the area of the car masks
    :param car_ids: string ids of the cars (order matters!)
    :param n_splits:
    :param fold_id:
    :param random_state:
    :param min_bin_size: min number of samples in each of the area classes(bins)
    :return:
    """
    df = pd.read_hdf(join(config.input_data_dir, 'areas_df.hdf5'))
    assert len(car_ids) == len(df)
    df = df.loc[car_ids]
    freq, bins = np.histogram(df['sum'], bins=35)
    new_bins = [bins[0]]
    new_bin_hs = []
    pos = 0
    cur_bin_right = 0
    cur_bin_h = 0
    while pos < len(freq):
        cur_bin_right = max(cur_bin_right, bins[pos + 1])
        cur_bin_h += freq[pos]
        if cur_bin_h >= min_bin_size or pos == len(freq) - 1:
            new_bin_hs.append(cur_bin_h)
            cur_bin_h = 0
            new_bins.append(cur_bin_right)
        pos += 1
    if new_bin_hs[-1] < min_bin_size:
        new_bin_hs[-2] += new_bin_hs[-1]
        del new_bin_hs[-1]
        new_bins[-2] = new_bins[-1]
        del new_bins[-1]

    new_bins[0] -= 1
    assert max(new_bins) == max(bins)
    assert freq.sum() == np.sum(new_bin_hs) == len(df)
    print 'Num area bins:', len(new_bin_hs)
    assert len(new_bin_hs) >= min_bin_size, len(new_bin_hs)
    df['area_class'] = pd.cut(df['sum'], new_bins, labels=False)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(kf.split(X=np.arange(len(df)), y=df['area_class'].values))[fold_id]
    car_indices = {'train': folds[0], 'val': folds[1]}
    return car_indices


class CARVANA(Dataset):
    """
        CARVANA dataset that contains car images as .jpg. Each car has 16 images
        taken in different angles and a unique id: id_01.jpg, id_02.jpg, ..., id_16.jpg
        The labels are provided as a .gif image that contains the manually cutout mask
        for each training image
    """

    def __init__(self, root, subset="train", image_size=512,
                 transform=None, is_hq=True, seed=1993, v=1, n_folds=10, fold_id=0, group='all', return_image_id=False):
        """

        :param root: it has to be a path to the folder that contains the dataset folders
        :param train: boolean true if you want the train set false for the test one
        :param transform: transform the images and labels
        """
        assert v in [1, 2], 'Unknown folds version: {}'.format(v)
        assert 0 <= fold_id < n_folds, fold_id
        assert group in range(1, 9) + ['all']
        print 'CARVANA::folds version={}'.format(v)
        self.is_hq = is_hq
        self.group = group
        self.return_image_id = return_image_id

        if group == 'all':
            num_views = 16
        else:
            num_views = 2
        print 'Group: {}; num_views:{}'.format(group, num_views)

        self.root = os.path.abspath(os.path.expanduser(root))
        self.transform = transform
        self.subset = subset
        self.data_path, self.labels_path = [], []
        self.rs = np.random.RandomState(seed)

        if self.subset in ['train', 'val']:
            suff = ''
            if image_size == 512:
                suff = '_{}'.format(image_size)
            images_dir = self.root + '/train' + ('_hq' if is_hq else '') + suff
            print 'Reading images from ', images_dir
            self.data_path = self.get_paths(images_dir, group)
            self.labels_path = self.get_paths(self.root + '/train_masks' + suff, group)
            assert len(self.data_path) / num_views == 5088 / num_views
            assert len(self.data_path) % num_views == 0
            num_cars = len(self.data_path) / num_views

            if v == 1:
                # random k folds
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.rs)
                folds = list(kf.split(np.arange(num_cars)))[fold_id]
                car_indices = {'train': folds[0], 'val': folds[1]}
            else:
                print 'Stratified folds based on mask area...'
                car_ids = np.unique(map(lambda x: os.path.basename(x[:-len('_01.jpg')]), self.data_path))
                assert os.path.basename(self.data_path[128 + 1]) == car_ids[128 / num_views] + '_{:02}.jpg'.format(2)
                car_indices = get_stratified_by_area_folds(car_ids,
                                                           n_splits=n_folds,
                                                           fold_id=fold_id,
                                                           random_state=self.rs)

            # TRAIN_FRAC = 0.9
            # num_train = int(TRAIN_FRAC * (len(self.data_path) / 16)) * 16
            # car_ids = self.rs.permutation(len(self.data_path) / 16)

            indices = dict()
            for split_name in ['train', 'val']:
                img_indices = []
                for car_id in car_indices[split_name]:
                    img_indices.extend(range(car_id * num_views, (car_id + 1) * num_views))
                indices[split_name] = img_indices

            self.data_path = self.data_path[indices[self.subset]]
            self.labels_path = self.labels_path[indices[self.subset]]

            print 'Dataset::{}: fold {}/{}'.format(self.subset, fold_id, n_folds)

        elif self.subset == "test":
            self.data_path = self.get_test_paths(is_hq, group)
            self.labels_path = None
        else:
            raise RuntimeError('Invalid subset ' + self.subset + ', it must be one of:'
                                                                 ' \'train\', \'val\' or \'test\'')

    def reload_all_test_paths(self):
        if self.subset != 'test':
            raise ValueError('Only possible for test!')
        self.data_path = self.get_test_paths(self.is_hq, self.group)

    @staticmethod
    def get_paths(dir_path, group):
        """
        returns all the sorted image paths.
        :param dir_path:
        :return: array with all the paths to the images
        """
        assert group in range(1, 9) + ['all']
        images_dir = [join(dir_path, f) for f in os.listdir(dir_path) if
                      isfile(join(dir_path, f))]
        final_paths = []
        if group in range(1, 9):
            endings = ['_{:02d}.jpg'.format(group), '_{:02d}_mask.gif'.format(group),
                       '_{:02d}.jpg'.format(group + 8), '_{:02d}_mask.gif'.format(group + 8)]
            for path in images_dir:
                for ending in endings:
                    if path.endswith(ending):
                        final_paths.append(path)
                        break
        else:
            final_paths = images_dir
        final_paths.sort()
        return np.asarray(final_paths)

    @staticmethod
    def get_test_paths(is_hq, group='all'):
        return CARVANA.get_paths(join(config.input_data_dir, 'test') +
                                 ('_hq' if is_hq else ''), group)

    def __getitem__(self, index):
        """

        :param index:
        :return: tuple (img, target) with the input data and its label
        """

        # load image and labels
        img = Image.open(self.data_path[index])
        target = Image.open(self.labels_path[index]) if not self.subset == 'test' else None
        # target = np.asarray(target) * 255
        if target is not None and target.mode == 'P':
            # has values from 0 to 1
            target = target.convert('L') # convert to int value from 0 to 255

        # apply transforms to both
        if self.transform is not None:
            if target is not None:
                img, target = self.transform(img, target)
            else:
                img = self.transform(img)

        if target is not None:
            assert target.max() == 1.0, 'Wrong scaling for target mask (max val = {})'.format(target.max())
            target[(target > 0) & (target < 1.0)] = 0
            assert ((target > 0) & (target < 1.0)).sum() == 0
            if not self.return_image_id:
                return img, target
            else:
                return img, target, Path(self.data_path[index]).stem
        else:
            return img, Path(self.data_path[index]).stem

    def __len__(self):
        return len(self.data_path)


class CarvanaPlus(Dataset):
    """
        CARVANA dataset that contains car images as .jpg. Each car has 16 images
        taken in different angles and a unique id: id_01.jpg, id_02.jpg, ..., id_16.jpg
        The labels are provided as a .gif image that contains the manually cutout mask
        for each training image
    """

    def __init__(self, root, subset="train", image_size=512,
                 transform=None, is_hq=True, seed=1993, v=1, n_folds=10, fold_id=0,
                 group='all', return_image_id=False):
        if subset not in ['train']:
            raise ValueError('No test split available')
        self.carvana = CARVANA(root, subset, image_size,
                               transform, is_hq, seed, v, n_folds,
                               fold_id, group, return_image_id)

    def __getitem__(self, index):
        return self.carvana[index]

    def __len__(self):
        return len(self.carvana)


def im_show(img_list):
    """
    It receives a list of images and plots them together
    :param img_list:
    :return:
    """
    to_PIL = transforms.ToPILImage()
    if len(img_list) >= 10:
        raise Exception("len(img_list) must be smaller than 10")

    for idx, img in enumerate(img_list):
        img = np.array(to_PIL(img))
        plt.subplot(100 + 10 * len(img_list) + (idx + 1))
        fig = plt.imshow(img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()


def save_checkpoint(state, is_best, filepath='checkpoint.pth.tar'):
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath,
                        os.path.join(os.path.dirname(filepath), 'model_best.pth.tar'))

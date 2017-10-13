import random
import numpy as np
from torch.utils.data.dataset import Dataset as BaseDataset
import scipy.ndimage

from .crops import ImageCropper

class Dataset(BaseDataset):
    def __init__(self, h5dataset, image_indexes, config, stage='train', transform=None):
        self.cropper = ImageCropper(config.img_rows,
                                    config.img_cols,
                                    config.target_rows,
                                    config.target_cols,
                                    config.train_pad if stage=='train' else config.test_pad,
                                    config.use_crop,
                                    config.use_resize)
        self.dataset = h5dataset
        self.image_indexes = image_indexes if isinstance(image_indexes, list) else image_indexes.tolist()
        self.transform = transform
        self.stage = stage
        self.keys = {'image', 'image_name'}
        self.config = config

    def __getitem__(self, item):
        raise NotImplementedError

    def image_to_float(self, image):
        return np.moveaxis(image / (255. if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32)


class MaskedDataset(Dataset):
    def __init__(self, h5dataset, image_indexes, config, stage, transform=None):
        super(MaskedDataset, self).__init__(h5dataset, image_indexes, config, stage, transform)
        self.keys = {'image', 'mask', 'image_name'}

    def expand_mask(self, mask):
        return np.expand_dims(mask / 255., axis=0).astype(np.float32)

    def distance_transform(self, mask):
        emask = -scipy.ndimage.distance_transform_edt(~mask) + scipy.ndimage.distance_transform_edt(mask)
        emask_img = np.sign(emask) * np.log1p(np.abs(emask))
        return np.expand_dims(emask_img, axis=0).astype(np.float32)



class TrainDataset(MaskedDataset):
    def __init__(self, h5dataset, image_indexes, config, stage='train', transform=None):
        super(TrainDataset, self).__init__(h5dataset, image_indexes, config, stage, transform)

    def __getitem__(self, idx):
        """
        idx seems to be unused
        """
        im_idx = self.image_indexes[idx % len(self.image_indexes)]
        for ix in range(50):
            sx, sy = self.cropper.randomCropCoords()
            name = self.dataset['names'][im_idx]
            if 'alphas' in self.dataset:
                alpha = self.cropper.getImage(self.dataset, 'alphas', im_idx, sx, sy)
                if np.mean(alpha) < 3:
                    continue
            mask = self.cropper.getImage(self.dataset, 'masks', im_idx, sx, sy)
            if 3 < np.mean(mask) < 252:  # sample borders
                break
            if random.random() < .7:  # sample background twice less then target
                break
        im = self.cropper.getImage(self.dataset, 'images', im_idx, sx, sy)
        if self.transform is not None:
            im, mask = self.transform(im, mask)
        return {'image': self.image_to_float(im), 'mask': self.expand_mask(mask), 'image_name': name}

    def __len__(self):
        return len(self.image_indexes) * self.config.epoch_size # epoch size is len images

class SequentialDataset(Dataset):
    def __init__(self, h5dataset, image_indexes, config, stage='train', transform=None):
        super(SequentialDataset, self).__init__(h5dataset, image_indexes, config, stage, transform)
        self.good_tiles = []
        self.init_good_tiles()
        self.keys = {'image', 'image_name', 'sy', 'sx'}

    def init_good_tiles(self):
        self.good_tiles = []
        positions = self.cropper.positions
        for im_idx in self.image_indexes:
            if 'alphas' in self.dataset:
                alpha_generator = self.cropper.sequentialCrops(self.dataset['alphas'][im_idx])
                for idx, alpha in enumerate(alpha_generator):
                    if np.mean(alpha) > 5:
                        self.good_tiles.append((im_idx, *positions[idx]))
            else:
                for pos in positions:
                    self.good_tiles.append((im_idx, *pos))

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        im_idx, sx, sy = self.good_tiles[idx]
        im = self.cropper.getImage(self.dataset, 'images', im_idx, sx, sy)
        if self.transform is not None:
            im = self.transform(im)
        name = self.dataset['names'][im_idx]
        return {'image': self.image_to_float(im), 'startx': sx, 'starty': sy, 'image_name': name}

    def __len__(self):
        return len(self.good_tiles)


class ValDataset(SequentialDataset, MaskedDataset):
    def __init__(self, h5dataset, image_indexes, config, stage='train', transform=None):
        super(ValDataset, self).__init__(h5dataset, image_indexes, config, stage, transform)
        self.keys = {'image', 'mask', 'image_name', 'sx', 'sy'}

    def __getitem__(self, idx):
        res = SequentialDataset.__getitem__(self, idx)
        if res is None:
            return res
        im_idx, sx, sy = self.good_tiles[idx]
        mask = self.cropper.getImage(self.dataset, 'masks', im_idx, sx, sy)
        res['mask'] = self.expand_mask(mask)
        return res


import os

import cv2
import numpy as np
from PIL import Image

from .abstract_dataset import ReadingDataset
from transforms import CLAHE

class CarvanaDataset(ReadingDataset):
    def __init__(self, root, rows, cols, channels=3, image_folder_name='train', apply_clahe=False):
        super(CarvanaDataset, self).__init__(root, rows, cols, channels)
        self.image_folder_name = image_folder_name
        self.im_names = os.listdir(os.path.join(root, self.image_folder_name))
        self.images = {}
        self.masks = {}
        self.clahe = CLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.apply_clahe = apply_clahe

    def read_image(self, fn):
        im = cv2.imread(os.path.join(self.root, self.image_folder_name, fn))
        return self.clahe(im) if self.apply_clahe else im

    def read_mask(self, fn):
        path = os.path.join(self.root, 'train_masks', os.path.splitext(fn)[0] + '_mask.gif')
        mask = np.copy(np.asarray(Image.open(path)))# * 255
        if np.max(mask) < 255:
            mask[mask > 0] = 255
        return mask.astype(np.uint8)

    def finalyze(self, data):
        return self.pad_image(data)

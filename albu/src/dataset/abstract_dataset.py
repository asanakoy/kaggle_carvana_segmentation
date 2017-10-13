import cv2
import numpy as np

class AbstractDataset:
    def __init__(self, rows, cols, channels=3):
        self.rows = rows
        self.cols = cols
        self.channels = channels

    def reflect_border(self, image, b=12):
        return cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_REFLECT)

    def pad_image(self, image):
        channels = image.shape[2] if len(image.shape) > 2 else None
        if image.shape[:2] != (self.rows, self.cols):
            empty_x = np.zeros((self.rows, self.cols, channels), dtype=image.dtype) if channels else np.zeros((self.rows, self.cols), dtype=image.dtype)
            empty_x[0:image.shape[0],0:image.shape[1],...] = image
            image = empty_x
        return image

class ReadingDataset(AbstractDataset):
    def __init__(self, root, rows, cols, channels=3):
        super(ReadingDataset, self).__init__(rows, cols, channels)
        self.root = root
        self.im_names = []
        self.with_alpha = None

    def read_image(self, fn):
        raise NotImplementedError

    def read_mask(self, fn):
        raise NotImplementedError

    def read_alpha(self, fn):
        raise NotImplementedError

    def get_image(self, idx):
        fn = self.im_names[idx]
        data = self.read_image(fn)
        return self.finalyze(data)

    def get_mask(self, idx):
        fn = self.im_names[idx]
        data = self.read_mask(fn)
        return self.finalyze(data)

    def get_alpha(self, idx):
        fn = self.im_names[idx]
        data = self.read_alpha(fn)
        return self.finalyze(data)

    def finalyze(self, data):
        return self.pad_image(self.reflect_border(data))

    def __len__(self):
        return len(self.im_names)

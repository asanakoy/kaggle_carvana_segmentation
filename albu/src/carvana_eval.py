import os

import cv2
import numpy as np
from utils import get_config, get_csv_folds, get_npy_folds
from dataset.h5like_interface import H5LikeFileInterface
from eval import Evaluator, flip

from dataset.carvana_dataset import CarvanaDataset


class FullImageEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_data(self, predicted, model, data, prefix=""):
        names, samples, masks = self.get_data(data)
        for i in range(len(names)):
            self.prev_name = names[i]
            self.full_pred = np.squeeze(predicted[i,...])
            if samples is not None:
                self.full_image = (samples[i,...] * 255).astype(np.uint8)
            if masks is not None:
                self.full_mask = (np.squeeze(masks[i,...]) * 255).astype(np.uint8)
            self.on_image_constructed(prefix)

    def save(self, name, prefix=""):
        cv2.imwrite(os.path.join(self.config.results_dir, 'mask_{}'.format(name)), (self.full_pred * 255).astype(np.uint8))


class CarvanaEval(FullImageEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, name, prefix=""):
        name, ext = os.path.splitext(name)
        cv2.imwrite(os.path.join('..', 'results', self.config.folder, "{}{}.png".format(prefix, name)), (self.full_pred[:1280, :1918] * 255).astype(np.uint8))

def eval_config(config_path):
    test = True
    config = get_config(config_path)

    num_workers = 0 if os.name == 'nt' else 3
    root = config.dataset_path
    image_folder_name = 'train_hq' if not test else 'test_hq'
    c_ds = CarvanaDataset(root, config.img_rows, config.img_cols, image_folder_name=image_folder_name, apply_clahe=config.use_clahe)
    ds = H5LikeFileInterface(c_ds)
    if not test:
        if 'f04a' in config.folder:
            folds = get_csv_folds(os.path.join(root, 'folds_csv.csv'), os.listdir(os.path.join(root, image_folder_name)))
        else:
            folds = get_npy_folds(os.path.join(root, 'folds_train.npy'))
    else:
        folds = [([], list(range(len(c_ds)))) for i in range(5)]

    keval = CarvanaEval(config, ds, folds, test=test, flips=flip.FLIP_LR, num_workers=num_workers, border=0)
    keval.need_dice = True
    skip_folds = [i for i in range(5) if config.fold is not None and i != int(config.fold)]
    print('skipping folds: ', skip_folds)
    keval.predict(skip_folds=skip_folds)


if __name__ == "__main__":
    for config in os.listdir('../configs'):
        eval_config(os.path.join('..', 'configs', config))

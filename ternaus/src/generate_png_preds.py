import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import train
import utils
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_models import UNet11


class PredictionDatasetPure:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = train.load_image(path)
        return utils.img_transform(image), path.stem


def predict(model, from_paths, batch_size: int, to_path):
    loader = DataLoader(
        dataset=PredictionDatasetPure(from_paths),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=True
    )

    for batch_num, (inputs, stems) in enumerate(tqdm(loader, desc='Predict')):
        inputs = utils.variable(inputs, volatile=True)
        outputs = model(inputs)
        mask = (outputs.data.cpu().numpy() * 255).astype(np.uint8)

        for i, image_name in enumerate(stems):
            cv2.imwrite(str(to_path / (stems[i] + '.png')), mask[i, 0, :, 1:-1])


def get_model(model_name):
    model = UNet11()
    model = nn.DataParallel(model, device_ids=[0]).cuda()

    state = torch.load(
        str(utils.MODEL_PATH / model_name / 'best-model_{fold}.pt'.format(fold=fold, model_name=model_name)))

    model.load_state_dict(state['model'])
    model.eval()

    return model


if __name__ == '__main__':
    img_rows, img_cols = 1280, 1918

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--fold', type=int)

    utils.add_args(parser)
    args = parser.parse_args()

    model_path = utils.MODEL_PATH
    data_path = utils.DATA_ROOT

    model_name = 'unet_11'

    pred_path = Path(model_name)
    pred_path.mkdir(exist_ok=True, parents=True)

    fold_path = pred_path / str(args.fold)

    val_path = fold_path / 'val'
    val_path.mkdir(exist_ok=True, parents=True)

    test_path = fold_path / 'test'
    test_path.mkdir(exist_ok=True, parents=True)

    fold = args.fold
    batch_size = 2

    model = get_model(model_name)

    val_images = sorted(list((Path(str(args.fold)) / 'val' / 'images').glob('*.jpg')))
    num_val = len(val_images)
    predict(model, val_images, batch_size, val_path)

    test_images = sorted(list((data_path / 'test_hq').glob('*.jpg')))
    num_test = len(test_images)
    predict(model, test_images, batch_size, test_path)

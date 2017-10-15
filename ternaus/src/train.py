import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.backends.cudnn
import utils
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from unet_models import Loss, UNet11

img_cols, img_rows = 1280, 1920

Size = Tuple[int, int]


class CarvanaDataset(Dataset):
    def __init__(self, root: Path, to_augment=False):
        # TODO This potentially may lead to bug.
        self.image_paths = sorted(root.joinpath('images').glob('*.jpg'))
        self.mask_paths = sorted(root.joinpath('masks').glob('*'))
        self.to_augment = to_augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        mask = load_mask(self.mask_paths[idx])

        if self.to_augment:
            img, mask = augment(img, mask)

        return utils.img_transform(img), torch.from_numpy(np.expand_dims(mask, 0))


def grayscale_aug(img, mask):
    car_pixels = (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * img).astype(np.uint8)

    gray_car = cv2.cvtColor(car_pixels, cv2.COLOR_RGB2GRAY)

    rgb_gray_car = cv2.cvtColor(gray_car, cv2.COLOR_GRAY2RGB)

    rgb_img = img.copy()
    rgb_img[rgb_gray_car > 0] = rgb_gray_car[rgb_gray_car > 0]
    return rgb_img


def augment(img, mask):
    if np.random.random() < 0.5:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)

    if np.random.random() < 0.5:
        if np.random.random() < 0.5:
            img = random_hue_saturation_value(img,
                                              hue_shift_limit=(-50, 50),
                                              sat_shift_limit=(-5, 5),
                                              val_shift_limit=(-15, 15))
        else:
            img = grayscale_aug(img, mask)

    return img.copy(), mask.copy()


def random_hue_saturation_value(image,
                                hue_shift_limit=(-180, 180),
                                sat_shift_limit=(-255, 255),
                                val_shift_limit=(-255, 255)):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
    h = cv2.add(h, hue_shift)
    sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
    s = cv2.add(s, sat_shift)
    val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
    v = cv2.add(v, val_shift)
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def load_image(path: Path):
    img = cv2.imread(str(path))
    img = cv2.copyMakeBorder(img, 0, 0, 1, 1, cv2.BORDER_REFLECT_101)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


def load_mask(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if '.gif' in str(path):
                img = (np.asarray(img) > 0)
            else:
                img = (np.asarray(img) > 255 * 0.5)
            img = cv2.copyMakeBorder(img.astype(np.uint8), 0, 0, 1, 1, cv2.BORDER_REFLECT_101)
            return img.astype(np.float32)


def validation(model: nn.Module, criterion, valid_loader) -> Dict[str, float]:
    model.eval()
    losses = []

    dice = []

    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        dice += [get_dice(targets, (outputs > 0.5).float()).data[0]]

    valid_loss = np.mean(losses)  # type: float

    valid_dice = np.mean(dice)

    print('Valid loss: {:.5f}, dice: {:.5f}'.format(valid_loss, valid_dice))
    metrics = {'valid_loss': valid_loss, 'dice_loss': valid_dice}
    return metrics


def get_dice(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1) + epsilon

    return 2 * (intersection / union).mean()


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--dice-weight', type=float)
    arg('--nll-weights', action='store_true')
    arg('--device-ids', type=str, help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--size', type=str, default='1280x1920', help='Input size, for example 288x384. Must be multiples of 32')
    utils.add_args(parser)
    args = parser.parse_args()

    model_name = 'unet_11'

    args.root = str(utils.MODEL_PATH / model_name)

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    model = UNet11()

    device_ids = list(map(int, args.device_ids.split(',')))
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    loss = Loss()

    def make_loader(ds_root: Path, to_augment=False, shuffle=False):
        return DataLoader(
            dataset=CarvanaDataset(ds_root, to_augment=to_augment),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=True
        )

    train_root = utils.DATA_ROOT / str(args.fold) / 'train'
    valid_root = utils.DATA_ROOT / str(args.fold) / 'val'

    valid_loader = make_loader(valid_root)
    train_loader = make_loader(train_root, to_augment=True, shuffle=True)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation,
        fold=args.fold
    )


if __name__ == '__main__':
    main()

from pathlib import Path
import shutil

import pandas as pd
from tqdm import tqdm
import utils


if __name__ == '__main__':
    global_data_path = utils.DATA_ROOT
    local_data_path = Path('.').absolute()

    local_data_path.mkdir(exist_ok=True)

    train_path = global_data_path / 'train_hq'

    mask_path = global_data_path / 'train_masks'

    train_file_list = train_path.glob('*')

    folds = pd.read_csv('src/folds_csv.csv')

    num_folds = folds['fold'].nunique()

    angles = ['0' + str(x) for x in range(1, 10)] + [str(x) for x in range(10, 17)]

    for fold in range(num_folds):

        (local_data_path / str(fold) / 'train' / 'images').mkdir(exist_ok=True, parents=True)
        (local_data_path / str(fold) / 'train' / 'masks').mkdir(exist_ok=True, parents=True)

        (local_data_path / str(fold) / 'val' / 'images').mkdir(exist_ok=True, parents=True)
        (local_data_path / str(fold) / 'val' / 'masks').mkdir(exist_ok=True, parents=True)

    for i in tqdm(folds.index):
        car_id = folds.loc[i, 'id']
        fold = folds.loc[i, 'fold']

        for angle in angles:
            old_image_path = train_path / (car_id + '_' + angle + '.jpg')

            new_image_path = local_data_path / str(fold) / 'val' / 'images' / (car_id + '_' + angle + '.jpg')
            shutil.copy(str(old_image_path), str(new_image_path))

            old_mask_path = mask_path / (car_id + '_' + angle + '_mask.gif')
            new_mask_path = local_data_path / str(fold) / 'val' / 'masks' / (car_id + '_' + angle + '_mask.gif')
            shutil.copy(str(old_mask_path), str(new_mask_path))

        for t_fold in range(num_folds):
            if t_fold == fold:
                continue

            for angle in angles:
                old_image_path = train_path / (car_id + '_' + angle + '.jpg')

                new_image_path = local_data_path / str(t_fold) / 'train' / 'images' / (car_id + '_' + angle + '.jpg')
                shutil.copy(str(old_image_path), str(new_image_path))

                old_mask_path = mask_path / (car_id + '_' + angle + '_mask.gif')
                new_mask_path = local_data_path / str(t_fold) / 'train' / 'masks' / (car_id + '_' + angle + '_mask.gif')
                shutil.copy(str(old_mask_path), str(new_mask_path))

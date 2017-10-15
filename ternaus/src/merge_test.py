from pathlib import Path

import cv2
import numpy as np
import utils
from joblib import Parallel, delayed


def merge_test(file_name):
    result = np.zeros((num_folds, 1280, 1918))
    for fold in range(num_folds):
        img_path = file_name.parent.parent.parent / str(fold) / 'test' / (file_name.stem + '.png')
        img = cv2.imread(str(img_path), 0)
        result[fold] = img

    img = result.mean(axis=0).astype(np.uint8)

    cv2.imwrite(str(utils.SUBMISSION_PATH / 'ternaus27' / (file_name.stem + '.png')), img)


if __name__ == '__main__':
    num_folds = 5
    model_name = 'unet_11'

    test_images = sorted(list((Path(model_name) / model_name / '0' / 'test').glob('*.png')))

    (utils.SUBMISSION_PATH / model_name / 'test_averaged').mkdir(exist_ok=True, parents=True)

    Parallel(n_jobs=16)(delayed(merge_test)(x) for x in test_images)

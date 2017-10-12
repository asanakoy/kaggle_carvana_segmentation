import cv2
import random
import math
import numpy as np
from PIL import Image
from torchvision import transforms

import config


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, images):
        assert isinstance(images, list)
        if random.random() < 0.5:
            return map(lambda x: x.transpose(Image.FLIP_LEFT_RIGHT), images)
        return images


class HorizontalFlip(object):
    """Horizontally flips the given PIL.Image
    """

    def __call__(self, images):
        assert isinstance(images, list)
        return map(lambda x: x.transpose(Image.FLIP_LEFT_RIGHT), images)
        return images


class Rescale(object):
    """Rescales the input list of PIL.Image's to the given 'size'.
    size: size is the tuple (w, h)
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        if self.size == image.size:
            return image
        else:
            image = image.resize(self.size, self.interpolation)
        return image


class Pad(object):
    """Pad the input list of PIL.Image's to the given 'size' from right and bottom.
    size: size is the tuple (w, h)

    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if self.size == image.size:
            return image
        else:
            if self.size[0] < image.size[0] or self.size[1] < image.size[1]:
                raise ValueError('Cannot pad bigger image: im_size={}'.format(image.size))
            image_mode = image.mode
            image = np.asarray(image)
            w_pad = self.size[0] - image.shape[1]
            h_pad = self.size[1] - image.shape[0]
            if len(image.shape) == 2:
                image = np.pad(image, ((0, h_pad), (0, w_pad)), 'edge')
            else:
                image = np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)), 'edge')
            image = Image.fromarray(image, mode=image_mode)
        return image


def random_shift_scale_rotate(images,
                              shift_limit=(-0.0625, 0.0625),
                              scale_limit=(1 / 1.1, 1.1),
                              rotate_limit=(-7, 7),
                              aspect_limit=(1, 1),
                              borderMode=cv2.BORDER_REFLECT_101,
                              u=0.5):
    """

    :param images: list [image, mask]
    :param shift_limit:
    :param scale_limit:
    :param rotate_limit:
    :param aspect_limit:
    :param borderMode:
    :param u:
    :return:
    """
    assert isinstance(images, list)
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    # add 3rd channel for mask
    images[1] = images[1].reshape(images[1].shape + (1,))

    if random.random() < u:
        height, width, channel = images[0].shape

        angle = random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = random.uniform(scale_limit[0], scale_limit[1])
        aspect = random.uniform(aspect_limit[0], aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = math.cos(angle / 180 * math.pi) * (sx)
        ss = math.sin(angle / 180 * math.pi) * (sy)
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        for i, image in enumerate(images):
            images[i] = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=borderMode)  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    return images


def random_gray(image, u=0.5):
    if random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(image * coef, axis=2)
        image = np.dstack((gray, gray, gray))
    return image


def random_brightness(image, limit=(-0.3, 0.3), u=0.5):
    assert image.max() <= 1.0
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        image = alpha * image
        image = np.clip(image, 0., 1.)
    return image


def random_contrast(image, limit=(-0.3, 0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha * image + gray
        image = np.clip(image, 0., 1.)
    return image


def random_saturation(image, limit=(-0.3, 0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = image * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        image = alpha * image + (1.0 - alpha) * gray
        image = np.clip(image, 0., 1.)
    return image


# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def random_hue(image, hue_limit=(-0.1, 0.1), u=0.5):
    if random.random() < u:
        h = int(random.uniform(hue_limit[0], hue_limit[1]) * 180)
        # print(h)

        image = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255
    return image


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


class TrainTransform:
    def __init__(self, new_size_, aug=None, resize_mask=True, should_pad=False,
                 should_normalize=False):
        if should_pad:
            self.resize = Pad(new_size_)
        else:
            self.resize = Rescale(new_size_)
        self.flip = RandomHorizontalFlip()
        self.to_tensor = transforms.ToTensor()
        self.aug = aug
        self.resize_mask = resize_mask
        # standard normalization for pytorch pretrained networks
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        self.should_normalize = should_normalize
        if self.should_normalize:
            print 'Transformer::Use input normalization'

    def __call__(self, image, target):
        image = self.resize(image)
        if self.resize_mask:
            target = self.resize(target)
        # else:
        #     target = target.copy()

        if self.aug is not None:
            assert isinstance(self.aug, int), self.aug
            assert self.aug in [0, 1, 2], 'Uknown aug={}'.format(self.aug)
            image, target = self.flip([image, target])
            image, target = map(np.array, [image, target])

            if self.aug == 1 or self.aug == 2:
                image, target = \
                    random_shift_scale_rotate([image, target],
                                            shift_limit=(-0.0625, 0.0625),
                                            scale_limit=(0.91, 1.21),
                                            rotate_limit=(-10, 10) if self.aug == 1 else (-7, 7))

                image = image.astype(np.float32) / 255.0
                image = random_brightness(image, limit=(-0.5, 0.5), u=0.5)
                image = random_contrast(image, limit=(-0.5, 0.5), u=0.5)
                image = random_saturation(image, limit=(-0.3, 0.3), u=0.5)
                image = random_gray(image, u=0.25)
                image = np.asarray(image * 255, dtype=np.uint8)
            elif self.aug == 0:
                image, target = \
                    random_shift_scale_rotate([image, target],
                                            shift_limit=(-0.0625, 0.0625),
                                            scale_limit=(0.91, 1.21),
                                            rotate_limit=(-5, 5),
                                            u=0.75)
            target[target > 128] = 255
            target[target < 255] = 0
            target = Image.fromarray(target.squeeze(), mode='L')
            image = Image.fromarray(image)

        image = self.to_tensor(image)
        target = self.to_tensor(target)

        if self.should_normalize:
            image = self.normalize(image)

        return image, target

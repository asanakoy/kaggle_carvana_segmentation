import argparse
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os
import shutil
import pandas as pd
from os.path import join
import cv2
from pathlib2 import Path
from itertools import izip

import config
from vgg_unet import UnetVgg11
from vgg_unet import Vgg11a
import data_utils
from data_utils import rle_encode
from data_utils import rle_to_string
from unet import Unet
from unet import Unet5
from dataset import CARVANA
import glob


def predict(test_loader, model, threshold=0.5, dbg=False, save_probs=False, output_dir=None):
    if save_probs:
        if output_dir is None:
            raise ValueError('Specify an output dir to save probs')
        output_dir.mkdir(exist_ok=True)

    TRAIN = False
    model.train(TRAIN)
    print '---!!!---TRAIN={}'.format(TRAIN)

    original_shape = (1280, 1918)
    all_rles = []
    all_img_filenames = []

    img_list = []
    for i, (images, names) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Predicting'):
        images = Variable(images.cuda(), volatile=True)

        outputs = model(images)
        outputs = F.upsample(outputs, size=original_shape, mode='bilinear')
        output_probs = F.sigmoid(outputs)
        if save_probs:
            probs_np = np.squeeze(output_probs.data.cpu().numpy())
            if len(probs_np.shape) == 2:
                probs_np = probs_np[np.newaxis, ...]
            assert len(probs_np.shape) == 3, probs_np.shape
            prob_images = np.asarray(np.round(probs_np * 255), dtype=np.uint8)
            for probs_img, sample_name in izip(prob_images, names):
                cv2.imwrite(str(output_dir.joinpath(sample_name + '.png')), probs_img)

            masks = (probs_np > threshold)
        else:
            masks = (output_probs > threshold).data.cpu().numpy()

        masks = np.squeeze(masks)
        if len(masks.shape) == 2:
            masks = masks[np.newaxis, ...]
        assert len(masks.shape) == 3, masks.shape
        for mask, sample_name in izip(masks, names):
            mask = np.asarray(mask, dtype=np.bool)
            rle = rle_to_string(rle_encode(mask))
            all_rles.append(rle)
            all_img_filenames.append(sample_name + '.jpg')

        if i <= 3:
            if len(mask.shape) != 3:
                mask = mask[:, :, np.newaxis]
            mask = mask.astype(np.float32)
            img = images.data.cpu()[-1].numpy().transpose(1, 2, 0)
            img = np.asarray(img * 255, dtype=np.uint8)
            img = cv2.resize(img, dsize=original_shape[::-1], interpolation=cv2.INTER_LINEAR)
            img_list.extend([img, mask])
        if dbg and i == 3:
            break

    return all_rles, all_img_filenames, img_list


def load_from_files(test_loader, probs_dir=None):
    if probs_dir is None or not probs_dir.exists():
        raise ValueError('Dir with probs was not found! {}'.format(str(probs_dir)))
    print 'Reading from', probs_dir

    all_rles = []
    all_img_filenames = []

    for i, (images, names) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Loading from files'):

        for sample_name in names:
            probs_img = cv2.imread(str(probs_dir.joinpath(sample_name + '.png')), cv2.IMREAD_GRAYSCALE)

            mask = (probs_img >= 128)
            rle = rle_to_string(rle_encode(mask))
            all_rles.append(rle)
            all_img_filenames.append(sample_name + '.jpg')

    return all_rles, all_img_filenames


def create_submission(rles, img_paths, output_path):
    print('Create submission...')
    t = pd.read_csv(join(config.input_data_dir, 'sample_submission.csv'))
    assert len(rles) == len(img_paths) == len(t), '{} rles'.format(len(rles))
    t['rle_mask'] = rles
    t['img'] = map(os.path.basename, img_paths)
    print t.head(2)
    t.to_csv('{}.gz'.format(output_path), index=False, compression='gzip')
    print 'Saved'


def load_checkpoint(ckpt_dir, epoch=None):
    if ckpt_dir is not None:
        if Path(ckpt_dir).is_file():
            ckpt_path = Path(ckpt_dir)
        elif epoch is None:
            ckpt_path = Path(ckpt_dir) / 'model_best.pth.tar'
        else:
            ckpt_path = Path(ckpt_dir) / 'model_best_epoch{}.pth.tar'.format(epoch)
    else:
        raise ValueError('ckpt_dir must be not None')

    if ckpt_path.exists():
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(str(ckpt_path))
        print 'best_score', checkpoint['best_score']
        print 'arch', checkpoint['arch']
        print 'epoch', checkpoint['epoch']
        if 'cur_score' in checkpoint:
            print 'cur_score', checkpoint['cur_score']
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        if epoch is None:
            out_path = ckpt_path.with_name('model_best_epoch{}.pth.tar'.format(checkpoint['epoch']))
            if not out_path.exists():
                shutil.copy(str(ckpt_path), str(out_path))
    else:
        raise IOError("=> no checkpoint found at '{}'".format(ckpt_path))
    checkpoint['path'] = ckpt_path
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--seed', type=int, default=1993, help='random seed')
    parser.add_argument('--epoch', type=int, default=None, help='checkpoint epoch to use')
    parser.add_argument('-imsize', '--image_size', type=int, default=1024, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-no_cudnn', '--no_cudnn', action='store_true',
                        help='dont use cudnn?')
    parser.add_argument('-no_hq', '--no_hq', action='store_true',
                        help='do not use hq images?')
    parser.add_argument('-o', '--ckpt_dir', default=None)
    parser.add_argument('-net', '--network', default='Unet')
    parser.add_argument('-load', '--load', action='store_true')

    args = parser.parse_args()
    dbg = False

    torch.manual_seed(args.seed)
    print 'CudNN:', torch.backends.cudnn.version()
    print 'Run on {} GPUs'.format(torch.cuda.device_count())
    torch.backends.cudnn.benchmark = not args.no_cudnn  # Enable use of CudNN
    checkpoint = load_checkpoint(args.ckpt_dir, epoch=args.epoch)
    filters_sizes = checkpoint['filter_sizes']

    should_normalize = False
    if args.network == 'Unet5':
        model = torch.nn.DataParallel(Unet5(is_deconv=False, filters=filters_sizes)).cuda()
    elif args.network in ['vgg11v1', 'vgg11v2']:
        assert args.network[-2] == 'v'
        v = int(args.network[-1:])
        should_normalize = True
        model = torch.nn.DataParallel(UnetVgg11(n_classes=1, num_filters=filters_sizes.item(), v=v)).cuda()
    elif args.network in ['vgg11av1', 'vgg11av2']:
        assert args.network[-2] == 'v'
        v = int(args.network[-1:])
        model = torch.nn.DataParallel(Vgg11a(n_classes=1,
                                      num_filters=filters_sizes.item(),
                                      v=v)).cuda()
    else:
        model = torch.nn.DataParallel(Unet(is_deconv=False, filters=filters_sizes)).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model from checkpoint (epoch {})".format(checkpoint['epoch']))

    rescale_size = (args.image_size, args.image_size)
    if args.image_size == -1:
        print 'Use full size. Use padding'
        is_full_size = True
        rescale_size = (1920, 1280)

    transforms_seq = [data_utils.Rescale(rescale_size),
                      transforms.ToTensor()]
    if should_normalize:
        print 'Use VGG normalization!'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transforms_seq.append(normalize)
    transform = transforms.Compose(transforms_seq)

    test_dataset = CARVANA(root=config.input_data_dir,
                           subset='test',
                           image_size=args.image_size,
                           transform=transform,
                           seed=args.seed,
                           is_hq=not args.no_hq,
                           v=2)
    probs_output_dir = Path(config.submissions_dir) / \
        'test_probs_{}_epoch{}'.format(checkpoint['path'].parts[-2], checkpoint['epoch'])

    probs_calculated = list()
    if not args.load and probs_output_dir.exists():
        probs_calculated = glob.glob(str(probs_output_dir) + '/*.png')
        print 'Num precalculated:', len(probs_calculated)
        probs_calculated = set(map(lambda x: os.path.basename(x)[:-4], probs_calculated))
        before = len(test_dataset.data_path)
        test_dataset.data_path = filter(lambda x: os.path.basename(x)[:-4] not in probs_calculated,
                                        test_dataset.data_path)
        print 'Skipped {} images as the probs for them were already calculated'.format(before - len(test_dataset.data_path))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=2)

    if args.load:
        rles, all_img_filenames = load_from_files(test_loader, probs_dir=probs_output_dir)
    else:
        rles, all_img_filenames, _ = predict(test_loader, model, threshold=0.5, dbg=dbg, save_probs=True, output_dir=probs_output_dir)
        if len(probs_calculated):
            test_dataset.reload_all_test_paths()
            rles, all_img_filenames = load_from_files(test_loader, probs_dir=probs_output_dir)

    output_path = Path(config.submissions_dir) / ('test_{}_epoch{}.csv'.format(
        checkpoint['path'].parts[-2], checkpoint['epoch']))
    create_submission(rles, all_img_filenames, str(output_path))


if __name__ == '__main__':
    main()

import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F


def inspect_checkpoint(ckpt_path):
    if ckpt_path.exists():
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(str(ckpt_path))
        print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        del checkpoint['state_dict']
        del checkpoint['optimizer']
        print ''
        print '--'
        for key, val in checkpoint.iteritems():
            print '{}: {}'.format(key, val)
        print '--'
    else:
        raise IOError("=> no checkpoint found at '{}'".format(ckpt_path))
    return checkpoint


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

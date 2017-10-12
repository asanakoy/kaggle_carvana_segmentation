import argparse
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from torch.autograd import Variable
import time
import numpy as np
import random
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
from os.path import join

import config
from data_utils import TrainTransform
import unet
import vgg_unet
from vgg_unet import UnetVgg11
from dataset import CARVANA, save_checkpoint
from dataset import CarvanaPlus
from losses import DiceScore
from losses import SoftDiceLoss
from losses import CombinedLoss

ORIGINAL_SHAPE = (1280, 1918)


def train(train_loader, model, optimizer, epoch, num_epochs, criterion,
          num_grad_acc_steps=5, log_aggr=1, logger=None):
    model.train()

    dice_score_obj = DiceScore().cuda()
    sum_epoch_loss = 0
    sum_dice_score_carvana = 0
    num_carvana = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    start = time.time()
    for i, (images, labels, image_ids) in pbar:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        outputs = model(images)
        loss, bce_loss, soft_dice_loss = criterion(outputs, labels)
        loss_val = loss.data[0]
        sum_epoch_loss += loss_val

        idxs_carvana = torch.cuda.LongTensor(np.nonzero(np.array(map(lambda x: x.count('_'), image_ids)) == 1)[0])
        num_carvana += len(idxs_carvana)
        if len(idxs_carvana):
            sum_dice_score_carvana += dice_score_obj(outputs[idxs_carvana], labels[idxs_carvana]).data[0] * len(idxs_carvana)

        # accumulate gradients
        if i == 0:
            optimizer.zero_grad()
        loss.backward()
        if i % num_grad_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # assume no effects on bn for accumulating grad

        iter_num = epoch * len(train_loader) + i + 1
        aggr_iter_num = iter_num / log_aggr
        if iter_num % log_aggr == 0:
            dice_score = dice_score_obj(outputs, labels).data[0]
            logger.add_scalar('(train)loss', loss_val, aggr_iter_num)
            logger.add_scalar('(train)bce_loss', bce_loss.data[0], aggr_iter_num)
            logger.add_scalar('(train)soft_dice_loss', soft_dice_loss.data[0], aggr_iter_num)
            logger.add_scalar('(train)dice_score', dice_score, aggr_iter_num)

        if i == 0:
            output_masks = (F.sigmoid(outputs) > 0.5).float()
            logger.add_image('model/(train)output', make_grid(output_masks.data),
                             aggr_iter_num)
        pbar.set_description(
            '[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f)(crv dice %.5f) (%.2f im/s)'
            % (epoch + 1, num_epochs, loss_val,
               sum_epoch_loss / (i + 1),
               sum_dice_score_carvana / num_carvana if num_carvana else 0,
               len(images) / (time.time() - start)))
        start = time.time()
    logger.add_scalar('(train)avg_loss', sum_epoch_loss / len(train_loader), epoch + 1)
    logger.add_scalar('(train)carvana_avg_dice', sum_dice_score_carvana / num_carvana
                      if num_carvana else 0, epoch + 1)


def validate(val_loader, model, epoch, logger, is_eval=True, is_full_size=False):
    model.train(not is_eval)

    avg_score = 0
    avg_soft_dice_loss = 0
    dice_score_obj = DiceScore().cuda()
    soft_dice_loss_obj = SoftDiceLoss().cuda()

    cnt = 0
    for i, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader),
                                    desc='Validation'):
        images = Variable(images.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)

        outputs = model(images)
        if is_full_size:
            assert outputs.size()[3] == 1920, outputs.size()
            outputs = outputs[:, :, :, :1918].contiguous()
        else:
            outputs = F.upsample(outputs, size=ORIGINAL_SHAPE, mode='bilinear')

        score = dice_score_obj(outputs, labels)
        soft_dice_loss = soft_dice_loss_obj(outputs, labels)
        avg_score += score.data[0] * len(images)
        avg_soft_dice_loss += soft_dice_loss.data[0] * len(images)
        cnt += len(images)
        if i == 0:
            output_masks = (F.sigmoid(outputs) > 0.5).float()
            logger.add_image('val/(val)output', make_grid(output_masks.data), epoch)
    avg_score /= cnt
    avg_soft_dice_loss /= cnt
    logger.add_scalar('(val)dice_score', avg_score, epoch)
    logger.add_scalar('(val)soft_dice_score', 1 - avg_soft_dice_loss, epoch)

    print '[VAL] epoch {}: dice={:.6f}  soft_dice_score={:.6f}'.format(epoch,
                                                                    avg_score,
                                                                    1 - avg_soft_dice_loss)
    return avg_score


def parse_group(val):
    if val == 'all':
        return val
    else:
        return int(val)


def parse_int_pair(val):
    val = val.strip('()')
    pair = map(int, val.split(','))
    if len(pair) != 2:
        raise ValueError('Cannot parse a pair of int: {}'.format(val))
    return pair


class CyclicLr(object):
    def __init__(self, start_epoch, init_lr=1e-4, num_epochs_per_cycle=12, epochs_pro_decay=2,
                  lr_decay_factor=0.5):
        self.start_epoch = start_epoch
        self.init_lr = init_lr
        self.lr_decay_factor = lr_decay_factor
        self.num_epochs_per_cycle = num_epochs_per_cycle
        self.epochs_pro_decay = epochs_pro_decay

    def __call__(self, epoch):
        cur_epoch_in_cycle = (epoch - self.start_epoch) % self.num_epochs_per_cycle
        lr = self.init_lr * (self.lr_decay_factor ** int(cur_epoch_in_cycle / self.epochs_pro_decay))
        return lr


class VggCyclicLr(object):
    def __init__(self, start_epoch, init_lr=1e-4, num_epochs_per_cycle=20, duration=1):
        self.start_epoch = start_epoch
        self.init_lr = init_lr
        self.num_epochs_per_cycle = num_epochs_per_cycle
        self.duration = duration
        print 'Cycle duration coeff:', duration

    def __call__(self, epoch):
        cur_epoch_in_cycle = (epoch - self.start_epoch) % self.num_epochs_per_cycle
        if cur_epoch_in_cycle < (10 * self.duration):
            lr = self.init_lr
        elif cur_epoch_in_cycle < int(self.duration * 15):
            lr = self.init_lr * 0.1
        else:
            lr = self.init_lr * 0.1**2
        return lr


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-o', '--output_dir', default=None, help='output dir')
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
    parser.add_argument('-lr', '--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('-reset_lr', '--reset_lr', action='store_true',
                        help='should reset lr cycles? If not count epochs from 0')
    parser.add_argument('-opt', '--optimizer', default='sgd', choices=['sgd', 'adam', 'rmsprop'],
                        help='optimizer type')
    parser.add_argument('--decay_step', type=float, default=100, metavar='EPOCHS',
                        help='learning rate decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.5,
                        help='learning rate decay coeeficient')
    parser.add_argument('--cyclic_lr', type=int, default=None,
                        help='(int)Len of the cycle. If not None use cyclic lr with cycle_len) specified')
    parser.add_argument('--cyclic_duration', type=float, default=1.0,
                        help='multiplier of the duration of segments in the cycle')

    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='L2 regularizer weight')
    parser.add_argument('--seed', type=int, default=1993, help='random seed')
    parser.add_argument('--log_aggr', type=int, default=None, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-gacc', '--num_grad_acc_steps', type=int, default=1, metavar='N',
                        help='number of vatches to accumulate gradients')
    parser.add_argument('-imsize', '--image_size', type=int, default=1024, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-f', '--fold', type=int, default=0, metavar='N',
                        help='fold_id')
    parser.add_argument('-nf', '--n_folds', type=int, default=0, metavar='N',
                        help='number of folds')
    parser.add_argument('-fv', '--folds_version', type=int, default=1, choices=[1, 2],
                        help='version of folds (1) - random, (2) - stratified on mask area')
    parser.add_argument('-group', '--group', type=parse_group, default='all',
                        help='group id')
    parser.add_argument('-no_cudnn', '--no_cudnn', action='store_true',
                        help='dont use cudnn?')
    parser.add_argument('-aug', '--aug', type=int, default=None,
                        help='use augmentations?')
    parser.add_argument('-no_hq', '--no_hq', action='store_true',
                        help='do not use hq images?')
    parser.add_argument('-dbg', '--dbg', action='store_true',
                        help='is debug?')
    parser.add_argument('-is_log_dice', '--is_log_dice', action='store_true',
                        help='use -log(dice) in loss?')
    parser.add_argument('-no_weight_loss', '--no_weight_loss', action='store_true',
                        help='do not weight border in loss?')

    parser.add_argument('-suf', '--exp_suffix', default='', help='experiment suffix')
    parser.add_argument('-net', '--network', default='Unet')

    args = parser.parse_args()
    print 'aug:', args.aug
    # assert args.aug, 'Careful! No aug specified!'
    if args.log_aggr is None:
        args.log_aggr = 1
    print 'log_aggr', args.log_aggr

    random.seed(42)
    torch.manual_seed(args.seed)
    print 'CudNN:', torch.backends.cudnn.version()
    print 'Run on {} GPUs'.format(torch.cuda.device_count())
    torch.backends.cudnn.benchmark = not args.no_cudnn  # Enable use of CudNN

    experiment = "{}_s{}_im{}_gacc{}{}{}{}_{}fold{}.{}".format(args.network,
                                                               args.seed,
                                                               args.image_size,
                                                               args.num_grad_acc_steps,
                                                               '_aug{}'.format(args.aug) if args.aug is not None else '',
                                                               '_nohq' if args.no_hq else '',
                                                               '_g{}'.format(
                                                                   args.group) if args.group != 'all' else '',
                                                               'v2' if args.folds_version == 2 else '',
                                                               args.fold, args.n_folds)
    if args.output_dir is None:
        ckpt_dir = join(config.models_dir, experiment + args.exp_suffix)
        if os.path.exists(join(ckpt_dir, 'checkpoint.pth.tar')):
            args.output_dir = ckpt_dir
    if args.output_dir is not None and os.path.exists(args.output_dir):
        ckpt_path = join(args.output_dir, 'checkpoint.pth.tar')
        if not os.path.isfile(ckpt_path):
            print "=> no checkpoint found at '{}'\nUsing model_best.pth.tar".format(ckpt_path)
            ckpt_path = join(args.output_dir, 'model_best.pth.tar')
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            if 'filter_sizes' in checkpoint:
                filters_sizes = checkpoint['filter_sizes']
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
        else:
            raise IOError("=> no checkpoint found at '{}'".format(ckpt_path))
    else:
        checkpoint = None
        if args.network == 'Unet':
            filters_sizes = np.asarray([32, 64, 128, 256, 512, 1024, 1024])
        elif args.network == 'UNarrow':
            filters_sizes = np.asarray([32, 32, 64, 128, 256, 512, 768])
        elif args.network == 'Unet7':
            filters_sizes = np.asarray([48, 96, 128, 256, 512, 1024, 1536, 1536])
        elif args.network == 'Unet5':
            filters_sizes = np.asarray([32, 64, 128, 256, 512, 1024])
        elif args.network == 'Unet4':
            filters_sizes = np.asarray([24, 64, 128, 256, 512])
        elif args.network in ['vgg11v1', 'vgg11v2']:
            filters_sizes = np.asarray([64])
        elif args.network in ['vgg11av1', 'vgg11av2']:
            filters_sizes = np.asarray([32])
        else:
            raise ValueError('Unknown Net: {}'.format(args.network))
    if args.network in ['vgg11v1', 'vgg11v2']:
        assert args.network[-2] == 'v'
        v = int(args.network[-1:])
        model = torch.nn.DataParallel(UnetVgg11(n_classes=1, num_filters=filters_sizes.item(), v=v)).cuda()
    elif args.network in ['vgg11av1', 'vgg11av2']:
        assert args.network[-2] == 'v'
        v = int(args.network[-1:])
        model = torch.nn.DataParallel(vgg_unet.Vgg11a(n_classes=1,
                                                      num_filters=filters_sizes.item(),
                                                      v=v)).cuda()
    else:
        unet_class = getattr(unet, args.network)
        model = torch.nn.DataParallel(
            unet_class(is_deconv=False, filters=filters_sizes)).cuda()

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    rescale_size = (args.image_size, args.image_size)
    is_full_size = False
    if args.image_size == -1:
        print 'Use full size. Use padding'
        is_full_size = True
        rescale_size = (1920, 1280)
    elif args.image_size == -2:
        rescale_size = (1856, 1248)

    train_dataset = CarvanaPlus(root=config.input_data_dir,
                                subset='train',
                                image_size=args.image_size,
                                transform=TrainTransform(rescale_size,
                                                     aug=args.aug,
                                                     resize_mask=True,
                                                     should_pad=is_full_size,
                                                     should_normalize=args.network.startswith('vgg')),
                                seed=args.seed,
                                is_hq=not args.no_hq,
                                fold_id=args.fold,
                                n_folds=args.n_folds,
                                group=args.group,
                                return_image_id=True,
                                v=args.folds_version
                                )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4 if torch.cuda.device_count() > 1 else 1)

    val_dataset = CARVANA(root=config.input_data_dir,
                          subset='val',
                          image_size=args.image_size,
                          transform=TrainTransform(rescale_size,
                                                   aug=None,
                                                   resize_mask=False,
                                                   should_pad=is_full_size,
                                                   should_normalize=args.network.startswith('vgg')),
                          seed=args.seed,
                          is_hq=not args.no_hq,
                          fold_id=args.fold,
                          n_folds=args.n_folds,
                          group=args.group,
                          v=args.folds_version,
                          )
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size * 2,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=4 if torch.cuda.device_count() > 4 else torch.cuda.device_count())

    print 'Weight loss:', not args.no_weight_loss
    print '-log(dice) in loss:', args.is_log_dice
    criterion = CombinedLoss(is_weight=not args.no_weight_loss,
                             is_log_dice=args.is_log_dice).cuda()

    if args.optimizer == 'adam':
        print 'Using adam optimizer!'
        optimizer = optim.Adam(model.parameters(),
                               weight_decay=args.weight_decay,
                               lr=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)  # For Tiramisu weight_decay=0.0001
    else:
        optimizer = optim.SGD(model.parameters(),
                              weight_decay=args.weight_decay,
                              lr=args.lr,
                              momentum=0.9,
                              nesterov=False)

    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = join(config.models_dir, experiment + args.exp_suffix)
    print 'Model dir:', out_dir
    if args.dbg:
        out_dir = 'dbg_runs'
    logger = SummaryWriter(log_dir=out_dir)

    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']
        print 'Best score:', best_score
        print 'Current score:', checkpoint['cur_score']
        model.load_state_dict(checkpoint['state_dict'])
        print 'state dict loaded'
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
            param_group['initial_lr'] = args.lr
        # validate(val_loader, model, start_epoch * len(train_loader), logger,
        #   is_eval=args.batch_size > 1, is_full_size=is_full_size)
    else:
        start_epoch = 0
        best_score = 0
        # validate(val_loader, model, start_epoch * len(train_loader), logger,
        #          is_eval=args.batch_size > 1, is_full_size=is_full_size)

    if args.cyclic_lr is None:
        scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)
        print 'scheduler.base_lrs=', scheduler.base_lrs
    elif args.network.startswith('vgg'):
        print 'Using VggCyclic LR!'
        cyclic_lr = VggCyclicLr(start_epoch if args.reset_lr else 0, init_lr=args.lr, num_epochs_per_cycle=args.cyclic_lr,
                                duration=args.cyclic_duration)
        scheduler = LambdaLR(optimizer, lr_lambda=cyclic_lr)
        scheduler.base_lrs = list(map(lambda group: 1.0, optimizer.param_groups))
    else:
        print 'Using Cyclic LR!'
        cyclic_lr = CyclicLr(start_epoch if args.reset_lr else 0, init_lr=args.lr, num_epochs_per_cycle=args.cyclic_lr,
                             epochs_pro_decay=args.decay_step,
                             lr_decay_factor=args.decay_gamma
                             )
        scheduler = LambdaLR(optimizer, lr_lambda=cyclic_lr)
        scheduler.base_lrs = list(map(lambda group: 1.0, optimizer.param_groups))

    logger.add_scalar('data/batch_size', args.batch_size, start_epoch)
    logger.add_scalar('data/num_grad_acc_steps', args.num_grad_acc_steps, start_epoch)
    logger.add_text('config/info', 'filters sizes: {}'.format(filters_sizes))

    last_lr = 100500

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        scheduler.step(epoch=epoch)
        if last_lr != scheduler.get_lr()[0]:
            last_lr = scheduler.get_lr()[0]
            print 'LR := {}'.format(last_lr)
        logger.add_scalar('data/lr', scheduler.get_lr()[0], epoch)
        logger.add_scalar('data/aug', args.aug if args.aug is not None else -1, epoch)
        logger.add_scalar('data/weight_decay', args.weight_decay, epoch)
        logger.add_scalar('data/is_weight_loss', not args.no_weight_loss, epoch)
        logger.add_scalar('data/is_log_dice', args.is_log_dice, epoch)
        train(train_loader, model, optimizer, epoch, args.epochs, criterion,
              num_grad_acc_steps=args.num_grad_acc_steps,
              logger=logger,
              log_aggr=args.log_aggr)
        dice_score = validate(val_loader, model, epoch + 1, logger,
                              is_eval=args.batch_size > 1, is_full_size=is_full_size)

        # store best loss and save a model checkpoint
        is_best = dice_score > best_score
        prev_best_score = best_score
        best_score = max(dice_score, best_score)
        ckpt_dict = {
            'epoch': epoch + 1,
            'arch': experiment,
            'state_dict': model.state_dict(),
            'best_score': best_score,
            'cur_score': dice_score,
            'optimizer': optimizer.state_dict(),
        }
        ckpt_dict['filter_sizes'] = filters_sizes

        if is_best:
            print 'Best snapshot! {} -> {}'.format(prev_best_score, best_score)
            logger.add_text('val/best_dice',
                            'best val dice score: {}'.format(dice_score),
                            global_step=epoch + 1)
        save_checkpoint(ckpt_dict, is_best, filepath=join(out_dir, 'checkpoint.pth.tar'))

    logger.close()


if __name__ == '__main__':
    main()

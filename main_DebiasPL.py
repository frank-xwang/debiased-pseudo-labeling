import argparse
import builtins
import math
import os
import shutil
import time
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import data.datasets as datasets
import backbone as backbone_models
from models import get_fixmatch_model
from utils import utils, lr_schedule, get_norm, dist_utils
import data.transforms as data_transforms
from engine import validate
from torch.utils.tensorboard import SummaryWriter

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--trainindex_x', default=None, type=str, metavar='PATH',
                    help='path to train annotation_x (default: None)')
parser.add_argument('--trainindex_u', default=None, type=str, metavar='PATH',
                    help='path to train annotation_u (default: None)')
parser.add_argument('--arch', metavar='ARCH', default='FixMatch',
                    help='model architecture')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epoch', default=0, type=int, metavar='N',
                    help='number of epochs for learning warmup')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='use nesterov momentum')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--self-pretrained', default='', type=str, metavar='PATH',
                    help='path to MoCo pretrained model (default: none)')
parser.add_argument('--super-pretrained', default='', type=str, metavar='PATH',
                    help='path to supervised pretrained model (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# FixMatch configs:
parser.add_argument('--anno-percent', type=float, default=0.1,
                    help='number of labeled data')
parser.add_argument('--split-seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--mu', default=5, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--lambda-u', default=10, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='pseudo label threshold')
parser.add_argument('--eman', action='store_true', default=False,
                    help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--weak-type', default='DefaultTrain', type=str,
                    help='the type for weak augmentation')
parser.add_argument('--strong-type', default='RandAugment', type=str,
                    help='the type for strong augmentation')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')
# online_net.backbone for BYOL
parser.add_argument('--model-prefix', default='encoder_q', type=str,
                    help='the model prefix of self-supervised pretrained state_dict')
# additional hyperparameters
parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--output', default='checkpoints/DebiasPL/', type=str,
                    help='the path to checkpoints')
parser.add_argument('--tau', default=1.0, type=float,
                    help='strength for debiasing')
parser.add_argument('--multiviews', action='store_true', default=False,
                    help='augmentation invariant mapping')
parser.add_argument('--CLDLoss', action='store_true', default=False,
                    help='apply instance-group discrimination loss, in probability space')
parser.add_argument('--lambda-cld', default=0.3, type=float,
                    help='weights of CLD loss')
parser.add_argument('--masked-qhat', action='store_true', default=False,
                    help='update qhat with instances passing a threshold')
parser.add_argument('--use_clip', action='store_true', default=False,
                    help='add predictions from CLIP')
parser.add_argument('--qhat_m', default=0.999, type=float,
                    help='momentum for updating q_hat')
best_acc1 = 0


def main():
    args = parser.parse_args()
    assert args.warmup_epoch < args.schedule[0]
    print(args)

    if args.seed is not None:
        seed = args.seed + dist_utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)

    assert 0 < args.anno_percent < 1

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def causal_inference(current_logit, qhat, exp_idx, tau=0.5):
    # de-bias pseudo-labels
    debiased_prob = F.softmax(current_logit - tau*torch.log(qhat), dim=1)
    return debiased_prob

def initial_qhat(class_num=1000):
    # initialize qhat of predictions (probability)
    qhat = (torch.ones([1, class_num], dtype=torch.float)/class_num).cuda()
    print("qhat size: ".format(qhat.size()))
    return qhat

def update_qhat(probs, qhat, momentum, qhat_mask=None):
    if qhat_mask is not None:
        mean_prob = probs.detach()*qhat_mask.detach().unsqueeze(dim=-1)
    else:
        mean_prob = probs.detach().mean(dim=0)
    qhat = momentum * qhat + (1 - momentum) * mean_prob
    return qhat

def get_centroids(prob):
    N, D = prob.shape
    K = D
    cl = prob.argmin(dim=1).long().view(-1)  # -> class index
    Ncl = cl.view(cl.size(0), 1).expand(-1, D)
    unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
    labels_count_all = torch.ones([K]).long().cuda() # -> counts of each class
    labels_count_all[unique_labels[:,0]] = labels_count
    c = torch.zeros([K, D], dtype=prob.dtype).cuda().scatter_add_(0, Ncl, prob) # -> class centroids
    c = c / labels_count_all.float().unsqueeze(1)
    return cl, c

def CLDLoss(prob_s, prob_w, mask=None, weights=None):
    cl_w, c_w = get_centroids(prob_w)
    affnity_s2w = torch.mm(prob_s, c_w.t())
    if mask is None:
        loss = F.cross_entropy(affnity_s2w.div(0.07), cl_w, weight=weights)
    else:
        loss = (F.cross_entropy(affnity_s2w.div(0.07), cl_w, reduction='none', weight=weights) * (1 - mask)).mean()
    return loss

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
    model_func = get_fixmatch_model(args.arch)
    norm = get_norm(args.norm)
    model = model_func(
        backbone_models.__dict__[args.backbone],
        eman=args.eman,
        momentum=args.ema_m,
        norm=norm
    )
    print(model)
    print("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))

    if args.self_pretrained:
        if os.path.isfile(args.self_pretrained):
            print("=> loading checkpoint '{}'".format(args.self_pretrained))
            checkpoint = torch.load(args.self_pretrained, map_location="cpu")

            # rename self pre-trained keys to model.main keys
            state_dict = checkpoint['state_dict']
            model_prefix = 'module.' + args.model_prefix
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith(model_prefix) and not k.startswith(model_prefix + '.fc'):
                    # replace prefix
                    new_key = k.replace(model_prefix, "main.backbone")
                    state_dict[new_key] = state_dict[k]
                    if model.ema is not None:
                        new_key = k.replace(model_prefix, "ema.backbone")
                        state_dict[new_key] = state_dict[k].clone()
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            if len(msg.missing_keys) > 0:
                print("missing keys:\n{}".format('\n'.join(msg.missing_keys)))
            if len(msg.unexpected_keys) > 0:
                print("unexpected keys:\n{}".format('\n'.join(msg.unexpected_keys)))
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.self_pretrained, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.self_pretrained))
    elif args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model from '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                new_key = k.replace("module.", "")
                state_dict[new_key] = state_dict[k]
                del state_dict[k]
            model_num_cls = state_dict['fc.weight'].shape[0]
            if model_num_cls != args.cls:
                # if num_cls don't match, remove the last layer
                del state_dict['fc.weight']
                del state_dict['fc.bias']
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, \
                    "missing keys:\n{}".format('\n'.join(msg.missing_keys))
            else:
                model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    else:
        from datetime import datetime
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m-%d-%Y-%H:%M")
        print("date and time =", dt_string)	
        args.output = args.output + "-" + dt_string

    if args.rank in [-1, 0]:
        os.makedirs(args.output, exist_ok=True)
        writer = SummaryWriter(args.output)
        print("Writer is initialized")
    else:
        writer = None

    cudnn.benchmark = True

    # Supervised Data loading code
    if args.trainindex_x is not None and args.trainindex_u is not None:
        print("load index from {}/{}".format(args.trainindex_x, args.trainindex_u))
        index_info_x = os.path.join(args.data, 'indexes', args.trainindex_x)
        index_info_u = os.path.join(args.data, 'indexes', args.trainindex_u)
        index_info_x = pd.read_csv(index_info_x)
        trainindex_x = index_info_x['Index'].tolist()
        index_info_u = pd.read_csv(index_info_u)
        trainindex_u = index_info_u['Index'].tolist()
        train_dataset_x, train_dataset_u, val_dataset = get_imagenet_ssl(
            args.data, trainindex_x, trainindex_u,
            weak_type=args.weak_type, strong_type=args.strong_type, 
            multiviews=args.multiviews)
    else:
        print("random sampling {} percent of data".format(args.anno_percent * 100))
        train_dataset_x, train_dataset_u, val_dataset = get_imagenet_ssl_random(
            args.data, args.anno_percent, weak_type=args.weak_type, strong_type=args.strong_type, 
            multiviews=args.multiviews)
    print("train_dataset_x:\n{}".format(train_dataset_x))
    print("train_dataset_u:\n{}".format(train_dataset_u))
    print("val_dataset:\n{}".format(val_dataset))

    # Data loading code
    train_sampler = DistributedSampler if args.distributed else RandomSampler

    train_loader_x = DataLoader(
        train_dataset_x,
        sampler=train_sampler(train_dataset_x),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    train_loader_u = DataLoader(
        train_dataset_u,
        sampler=train_sampler(train_dataset_u),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = args.start_epoch
    qhat = initial_qhat(class_num=1000)

    # load zero-shot predictions from CLIP
    clip_predictions = torch.load('imagenet/indexes/{}_clip_predictions_ranked.pth.tar'.format(args.trainindex_u.split('.csv')[0])) if args.use_clip else None
    clip_probs_list = clip_predictions['probs_list'].cuda() if args.use_clip else None
    clip_preds_list = clip_predictions['preds_list'].cuda() if args.use_clip else None
    # clip_imagenet_index_list = clip_predictions['imagenet_index_list'].cuda() if args.use_clip else None
    # clip_target_list = clip_predictions['target_list'].cuda() if args.use_clip else None
    # clip_sampler_index_list = clip_predictions['sampler_index_list'].cuda() if args.use_clip else None

    for epoch in range(args.start_epoch, args.epochs):
        if epoch >= args.warmup_epoch:
            lr_schedule.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        qhat = train(train_loader_x, train_loader_u, model, optimizer, epoch, args, qhat, writer, \
            clip_probs_list, clip_preds_list)

        is_best = False
        if (epoch + 1) % args.eval_freq == 0:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch = epoch
            if args.rank in [-1, 0]:
                writer.add_scalar('test/1.test_acc', acc1, epoch)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, ckpt_path=args.output)

        # print(qhat)

    print('Best Acc@1 {0} @ epoch {1}'.format(best_acc1, best_epoch + 1))
    print('checkpoint saved in: ', args.output)
    if args.rank in [-1, 0]:
        writer.close()


def train(train_loader_x, train_loader_u, model, optimizer, epoch, args, qhat=None, writer=None, \
          clip_probs_list=None, clip_preds_list=None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    losses_x = utils.AverageMeter('Loss_x', ':.4e')
    losses_u = utils.AverageMeter('Loss_u', ':.4e')
    losses_cld = utils.AverageMeter('Loss_cld', ':.4e')
    top1_x = utils.AverageMeter('Acc_x@1', ':6.2f')
    top5_x = utils.AverageMeter('Acc_x@5', ':6.2f')
    top1_u = utils.AverageMeter('Acc_u@1', ':6.2f')
    top5_u = utils.AverageMeter('Acc_u@5', ':6.2f')
    mask_info = utils.AverageMeter('Mask', ':6.3f')
    ps_vs_gt_correct = utils.AverageMeter('PseudoAcc', ':6.3f')
    curr_lr = utils.InstantMeter('LR', '')
    progress = utils.ProgressMeter(
        len(train_loader_u),
        [curr_lr, batch_time, data_time, losses, losses_x, losses_u, losses_cld, top1_x, top5_x, top1_u, top5_u, mask_info, ps_vs_gt_correct],
        prefix="Epoch: [{}/{}]\t".format(epoch, args.epochs))

    epoch_x = epoch * math.ceil(len(train_loader_u) / len(train_loader_x))
    if args.distributed:
        print("set epoch={} for labeled sampler".format(epoch_x))
        train_loader_x.sampler.set_epoch(epoch_x)
        print("set epoch={} for unlabeled sampler".format(epoch))
        train_loader_u.sampler.set_epoch(epoch)

    train_iter_x = iter(train_loader_x)
    # switch to train mode
    model.train()
    if args.eman:
        print("setting the ema model to eval mode")
        if hasattr(model, 'module'):
            model.module.ema.eval()
        else:
            model.ema.eval()

    end = time.time()
    # for i, (images_u, targets_u) in enumerate(train_loader_u):
    for i, (images_u, targets_u, indexs_u) in enumerate(train_loader_u):
        try:
            images_x, targets_x, _ = next(train_iter_x)
        except Exception:
            epoch_x += 1
            print("reshuffle train_loader_x at epoch={}".format(epoch_x))
            if args.distributed:
                print("set epoch={} for labeled sampler".format(epoch_x))
                train_loader_x.sampler.set_epoch(epoch_x)
            train_iter_x = iter(train_loader_x)
            images_x, targets_x, _ = next(train_iter_x)

        # prepare data and targets
        if args.multiviews:
            images_u_w, images_u_w2, images_u_s, images_u_s2 = images_u
            images_u_w = torch.cat([images_u_w.cuda(args.gpu, non_blocking=True), images_u_w2.cuda(args.gpu, non_blocking=True)], dim=0)
            images_u_s = torch.cat([images_u_s.cuda(args.gpu, non_blocking=True), images_u_s2.cuda(args.gpu, non_blocking=True)], dim=0)
        else:
            images_u_w, images_u_s = images_u

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images_x = images_x.cuda(args.gpu, non_blocking=True)
            images_u_w = images_u_w.cuda(args.gpu, non_blocking=True)
            images_u_s = images_u_s.cuda(args.gpu, non_blocking=True)

        targets_x = targets_x.cuda(args.gpu, non_blocking=True)
        targets_u = targets_u.cuda(args.gpu, non_blocking=True)

        # warmup learning rate
        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * len(train_loader_u)
            curr_step = epoch * len(train_loader_u) + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        # model forward
        if args.multiviews:
            logits = model(images_x, images_u_w, images_u_s)
            logits_x, logits_u_w, logits_u_s = logits
            logits_u_w1, logits_u_w2 = logits_u_w.chunk(2)
            logits_u_s1, logits_u_s2 = logits_u_s.chunk(2)
            logits_u_w = (logits_u_w1 + logits_u_w2) / 2
            logits_u_s = (logits_u_s1 + logits_u_s2) / 2
        else:
            logits_x, logits_u_w, logits_u_s = model(images_x, images_u_w, images_u_s)

        # producing debiased pseudo-labels
        pseudo_label = causal_inference(logits_u_w.detach(), qhat, exp_idx=0, tau=args.tau)
        if args.multiviews:
            pseudo_label1 = causal_inference(logits_u_w1.detach(), qhat, exp_idx=0, tau=args.tau)
            max_probs1, pseudo_targets_u1 = torch.max(pseudo_label1, dim=-1)
            mask1 = max_probs1.ge(args.threshold).float()
            pseudo_label2 = causal_inference(logits_u_w2.detach(), qhat, exp_idx=0, tau=args.tau)
            max_probs2, pseudo_targets_u2 = torch.max(pseudo_label2, dim=-1)
            mask2 = max_probs2.ge(args.threshold).float()

        max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        # update qhat
        qhat_mask = mask if args.masked_qhat else None
        qhat = update_qhat(torch.softmax(logits_u_w.detach(), dim=-1), qhat, momentum=args.qhat_m, qhat_mask=qhat_mask)

        # adaptive marginal loss
        delta_logits = torch.log(qhat)
        if args.multiviews:
            logits_u_s1 = logits_u_s1 + args.tau*delta_logits
            logits_u_s2 = logits_u_s2 + args.tau*delta_logits
        else:
            logits_u_s = logits_u_s + args.tau*delta_logits

        # loss for labeled samples
        loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')

        # loss for unlabeled samples
        per_cls_weights = None
        if args.multiviews:
            loss_u = 0
            pseudo_targets_list = [pseudo_targets_u, pseudo_targets_u1, pseudo_targets_u2]
            masks_list = [mask, mask1, mask2]
            logits_u_list = [logits_u_s1, logits_u_s2]
            for idx, targets_u in enumerate(pseudo_targets_list):
                for logits_u in logits_u_list:
                    loss_u += (F.cross_entropy(logits_u, targets_u, reduction='none', weight=per_cls_weights) * masks_list[idx]).mean()
            loss_u = loss_u/(len(pseudo_targets_list)*len(logits_u_list))
        else:
            loss_u = (F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none', weight=per_cls_weights) * mask).mean()
        
        # loss_u_ssl = F.cross_entropy(logits_u_s, pseudo_targets_u, reduction='none', weight=per_cls_weights) * mask
        
        if args.use_clip:
            # add clip's predictions
            indexs_u = indexs_u.cuda(args.gpu, non_blocking=True)
            targets_u_clip = clip_preds_list[indexs_u][:,0].view(-1)
            targets_u_clip = targets_u_clip.cuda(args.gpu, non_blocking=True)
            # add mask for clip with thresholding
            probs_list = clip_probs_list[indexs_u].cuda(args.gpu, non_blocking=True)
            max_probs, _ = torch.max(probs_list, dim=-1)
            mask_clip = max_probs.ge(0.5).float()
            
            # apply clip predictions to low-confidence predictions
            mask_delta = (mask_clip - mask - mask1 - mask2).ge(0.01).float()
            loss_u_clip = [F.cross_entropy(logits_u, targets_u_clip, reduction='none', weight=per_cls_weights) * mask_delta for logits_u in logits_u_list]
            loss_u = (torch.stack(loss_u_clip, dim=0).mean() + loss_u) / 2.0

        # CLD loss for unlabled samples (optional)
        if args.CLDLoss:
            prob_s = torch.softmax(logits_u_s, dim=-1)
            prob_w = torch.softmax(logits_u_w.detach(), dim=-1)
            loss_cld = CLDLoss(prob_s, prob_w, mask=None, weights=per_cls_weights)
        else:
            loss_cld = 0

        # total loss
        loss = loss_x + args.lambda_u * loss_u + args.lambda_cld*loss_cld

        if mask.sum() > 0:
            # measure pseudo-label accuracy
            _, targets_u_ = torch.topk(pseudo_label, 1)
            targets_u_ = targets_u_.t()
            ps_vs_gt = targets_u_.eq(targets_u.view(1, -1).expand_as(targets_u_))
            ps_vs_gt_all = (ps_vs_gt[0]*mask).view(-1).float().sum(0).mul_(100.0/mask.sum())
            ps_vs_gt_correct.update(ps_vs_gt_all.item(), mask.sum())

        # measure accuracy and record loss
        losses.update(loss.item())
        losses_x.update(loss_x.item(), images_x.size(0))
        losses_u.update(loss_u.item(), images_u_w.size(0))
        if args.CLDLoss:
            losses_cld.update(loss_cld.item(), images_u_w.size(0))
        acc1_x, acc5_x = utils.accuracy(logits_x, targets_x, topk=(1, 5))
        top1_x.update(acc1_x[0], logits_x.size(0))
        top5_x.update(acc5_x[0], logits_x.size(0))
        acc1_u, acc5_u = utils.accuracy(logits_u_w, targets_u, topk=(1, 5))
        top1_u.update(acc1_u[0], logits_u_w.size(0))
        top5_u.update(acc5_u[0], logits_u_w.size(0))
        mask_info.update(mask.mean().item(), mask.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        # update the ema model
        if args.eman:
            if hasattr(model, 'module'):
                model.module.momentum_update_ema()
            else:
                model.momentum_update_ema()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    if args.rank in [-1, 0]:
        writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
        writer.add_scalar('train/4.mask', mask_info.avg, epoch)
        writer.add_scalar('train/5.pseudo_vs_gt', ps_vs_gt_correct.avg, epoch)
        if args.CLDLoss:
            writer.add_scalar('train/7.train_loss_CLD', losses_cld.avg, epoch)
    
    return qhat


def get_imagenet_ssl(image_root, trainindex_x, trainindex_u,
                     train_type='DefaultTrain', val_type='DefaultVal', weak_type='DefaultTrain',
                     strong_type='RandAugment', multiviews=False):
    traindir = os.path.join(image_root, 'train')
    valdir = os.path.join(image_root, 'val')
    transform_x = data_transforms.get_transforms(train_type)
    weak_transform = data_transforms.get_transforms(weak_type)
    strong_transform = data_transforms.get_transforms(strong_type)
    if multiviews:
        weak_transform2 = data_transforms.get_transforms(weak_type)
        strong_transform2 = data_transforms.get_transforms(strong_type)
        transform_u = data_transforms.FourCropsTransform(weak_transform, weak_transform2, strong_transform, strong_transform2)
    else:
        transform_u = data_transforms.TwoCropsTransform(weak_transform, strong_transform)
    transform_val = data_transforms.get_transforms(val_type)

    train_dataset_x = datasets.ImageFolderWithIndex(
        traindir, trainindex_x, transform=transform_x)

    train_dataset_u = datasets.ImageFolderWithIndex(
        traindir, trainindex_u, transform=transform_u)

    val_dataset = datasets.ImageFolder(
        valdir, transform=transform_val)

    return train_dataset_x, train_dataset_u, val_dataset


def x_u_split(labels, percent, num_classes):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        label_per_class = max(1, round(percent * len(idx)))
        np.random.shuffle(idx)
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[label_per_class:])
    print('labeled_idx ({}): {}, ..., {}'.format(len(labeled_idx), labeled_idx[:5], labeled_idx[-5:]))
    print('unlabeled_idx ({}): {}, ..., {}'.format(len(unlabeled_idx), unlabeled_idx[:5], unlabeled_idx[-5:]))
    return labeled_idx, unlabeled_idx


def get_imagenet_ssl_random(image_root, percent, train_type='DefaultTrain',
                            val_type='DefaultVal', weak_type='DefaultTrain', strong_type='RandAugment',
                            multiviews=False):
    traindir = os.path.join(image_root, 'train')
    valdir = os.path.join(image_root, 'val')
    transform_x = data_transforms.get_transforms(train_type)
    weak_transform = data_transforms.get_transforms(weak_type)
    strong_transform = data_transforms.get_transforms(strong_type)
    if multiviews:
        strong_transform2 = data_transforms.get_transforms(strong_type)
        strong_transform3 = data_transforms.get_transforms(strong_type)
        transform_u = data_transforms.FourCropsTransform(weak_transform, strong_transform, strong_transform2, strong_transform3)
    else:
        transform_u = data_transforms.TwoCropsTransform(weak_transform, strong_transform)
    transform_val = data_transforms.get_transforms(val_type)

    base_dataset = datasets.ImageFolder(traindir)

    train_idxs_x, train_idxs_u = x_u_split(
        base_dataset.targets, percent, len(base_dataset.classes))

    train_dataset_x = datasets.ImageFolderWithIndex(
        traindir, train_idxs_x, transform=transform_x)

    train_dataset_u = datasets.ImageFolderWithIndex(
        traindir, train_idxs_u, transform=transform_u)

    val_dataset = datasets.ImageFolder(
        valdir, transform=transform_val)

    return train_dataset_x, train_dataset_u, val_dataset


def save_checkpoint(state, is_best, ckpt_path='', filename='checkpoint.pth.tar'):
    torch.save(state, "{}/{}".format(ckpt_path, filename))
    if is_best:
        shutil.copyfile("{}/{}".format(ckpt_path, filename), "{}/{}".format(ckpt_path, "model_best.pth.tar"))

if __name__ == '__main__':
    main()

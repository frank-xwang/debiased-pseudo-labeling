import math


def warmup_learning_rate(optimizer, curr_step, warmup_step, args):
    """linearly warm up learning rate"""
    lr = args.lr
    scalar = float(curr_step) / float(max(1, warmup_step))
    scalar = min(1., max(0., scalar))
    lr *= scalar
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        progress = float(epoch - args.warmup_epoch) / float(args.epochs - args.warmup_epoch)
        lr *= 0.5 * (1. + math.cos(math.pi * progress))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_official(optimizer, epoch, args, num_cycles=7./16.):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        progress = float(epoch - args.warmup_epoch) / float(args.epochs - args.warmup_epoch)
        lr *= max(0., math.cos(math.pi * num_cycles * progress))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_with_min(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        min_lr = args.cos_min_lr
        progress = float(epoch - args.warmup_epoch) / float(args.epochs - args.warmup_epoch)
        lr = min_lr + 0.5 * (lr - min_lr) * (1. + math.cos(math.pi * progress))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

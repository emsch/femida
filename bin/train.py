#!/usr/bin/env python3

import argparse
import pathlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
from femida_detect.detect import select, data_loader


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser('Training for detection model')
parser.add_argument('--paug', type=float, default=.5, help='Augmentation probability')
parser.add_argument('--imgsize', type=int, default=28, help='Image (Size x Size)')
parser.add_argument('--data-dir', type=pathlib.Path, help='Data directory')
parser.add_argument('--epoches', '-e', type=int, default=30, help='Number of epoches')
parser.add_argument('--lr', type=float, default=.001, help='Initial learning rate')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--save-dir', type=pathlib.Path, default='results', help='Directory for results')
parser.add_argument('--log-every', type=int, default=10, help='Log frequency (batches)')
parser.add_argument('--augment', type=str2bool, default=True, help='Augmentation for training')
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--gpu', type=str2bool, default=True)
parser.add_argument('--name', type=str, help='Name for experiment')
parser.add_argument('--threads', type=int, default=16, help='Data Loading Threads')
parser.add_argument('-v', type=str, help='Model Version')
parser.add_argument('-a', type=str, default=1, help='Aug Version')
parser.add_argument('--opt', type=str, choices=('adam', 'sgd'), help='Model Optimizer')
parser.add_argument('--wd', type=float, default=0.005, help='Weight decay')
parser.add_argument('--seed', type=int, default=42, help='Random seed')


class LogFile(object):
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            os.unlink(path)

    def write(self, buffer):
        with open(self.path, 'a') as f:
            f.write(buffer)
        print(buffer)


def check_args(args):
    # --save_dir
    args.save_dir = args.save_dir / args.name
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.model_dir = args.save_dir / 'model'
    # --result_dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    args.log_dir = args.save_dir / 'logs'
    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    if not args.epoches >= 1:
        raise ValueError('#epoches should be >= 1')
    # --batch_size
    if not args.batch_size >= 1:
        raise ValueError('batch-size should be >= 1')
    with open(args.log_dir / 'params.json', 'w') as f:
        j = dict((k, str(v)) for k, v in args.__dict__.items())
        json.dump(j, f)

    return args


def main():
    args = check_args(parser.parse_args())
    torch.manual_seed(args.seed)
    val_loader = data_loader(
        root=args.data_dir / 'validate',
        batch_size=args.batch_size,
        shuffle=False,
        shape=args.imgsize,
        augment=False,
        paug=args.paug,
        workers=args.threads,
    )
    train_loader = data_loader(
        root=args.data_dir / 'train',
        batch_size=args.batch_size,
        shuffle=True,
        shape=args.imgsize,
        augment=args.augment,
        paug=args.paug,
        workers=args.threads,
        vaug='v%s' % args.a
    )
    inp_shape = next(iter(train_loader))[0].shape
    model = select['v%s' % args.v](inp_shape[1], inp_shape[2], output_dim=1)
    bce_loss = nn.BCELoss()
    if args.opt == 'adam':
        optimizer = optim.Adam(
                model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                weight_decay=args.wd
        )
    elif args.opt == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
    else:
        raise RuntimeError(args.opt)
    if args.gpu:
        model.cuda()
        bce_loss.cuda()
    best_acc = -float('inf')
    LOG_TEMPLATE = 'e[{perc:.2f}/{e}/{total}]\tloss={loss:.3f}\tacc={acc:.4f}'
    LOG_TEMPLATE_VAL = '\nVALIDATION e[{e}/{total}]\tloss={loss:.3f}\tacc={acc:.4f}\n'
    logfile = LogFile(args.log_dir/'log.txt')

    def train_once(e):
        model.train()
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            if args.gpu:
                X, y = X.cuda(), y.cuda()
            yh = model(X)
            ce = bce_loss(yh.view(-1), y.float())
            ce.backward()
            optimizer.step()
            if i % args.log_every == 0:
                logfile.write(LOG_TEMPLATE.format(
                    perc=i/len(train_loader),
                    e=e+1,
                    total=args.epoches,
                    loss=ce.item(),
                    acc=((yh.view(-1) > .5).long() == y).float().mean().item()
                ))

    def validate(e):
        correct = total = loss = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                if args.gpu:
                    X, y = X.cuda(), y.cuda()
                yh = model(X)
                loss += bce_loss(yh.view(-1), y.float(),).item()*X.shape[0]
                correct += ((yh.view(-1) > .5).long() == y).float().sum().item()
                total += X.shape[0]
        ret = dict(acc=correct/total, bce=loss/total, e=e)
        logfile.write(LOG_TEMPLATE_VAL.format(
            e=e+1,
            total=args.epoches,
            loss=ret['bce'],
            acc=ret['acc']
        ))
        return ret
    evals = {}
    for epoch in range(args.epoches):
        train_once(epoch)
        evals = validate(epoch)
        if evals['acc'] > best_acc:
            best_acc = evals['acc']
            state = model.state_dict()
            state.update(evals)
            torch.save(state, args.model_dir / 'best_model.t7')
            logfile.write('NEW BEST MODEL!\n')
    state = model.state_dict()
    state.update(evals)
    torch.save(state, args.model_dir / 'model.t7')


if __name__ == '__main__':
    main()

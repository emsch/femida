#!/usr/bin/env python3

import argparse
import pathlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from femida_detect.detect import Modelv1, data_loader


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
parser.add_argument('--log-every', type=int, default=100, help='Log frequency (batches)')
parser.add_argument('--augment', type=str2bool, default=True, help='Augmentation for training')
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--gpu', type=str2bool, default=True)
parser.add_argument('--name', type=str, help='Name for experiment')


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
    return args


def main():
    args = check_args(parser.parse_args())
    val_loader = data_loader(
        root=args.data_dir / 'validate',
        batch_size=args.batch_size,
        shuffle=False,
        shape=args.imgsize,
        augment=False,
        paug=args.paug,
        workers=1,
    )
    train_loader = data_loader(
        root=args.data_dir / 'train',
        batch_size=args.batch_size,
        shuffle=True,
        shape=args.imgsize,
        augment=args.augment,
        paug=args.paug,
        workers=1,
    )
    inp_shape = next(iter(train_loader))[0].shape
    model = Modelv1(inp_shape[1], inp_shape[2], output_dim=1)
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    if args.gpu:
        model.cuda()
        bce_loss.cuda()
    best_acc = -float('inf')
    LOG_TEMPLATE = 'e[{perc:.2f}/{e}/{total}]\tloss={loss:.3f}\tacc={acc:.4f}'

    def train_once(e):
        model.train()
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            if args.gpu:
                X, y = X.cuda(), y.cuda()
            y = y.float()
            yh = model(X)
            ce = bce_loss(yh.view(-1), y)
            ce.backward()
            optimizer.step()
            if i % args.log_every == 0:
                print(LOG_TEMPLATE.format(
                    perc=i/len(train_loader),
                    e=e,
                    total=args.epoches,
                    loss=ce.mean().item(),
                    acc=(yh.argmax() == y.long()).float().mean().item()
                ))

    def validate(e):
        correct = total = loss = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                if args.gpu:
                    X, y = X.cuda(), y.cuda()
                yh = model(X)
                loss += bce_loss(yh.view(-1), y).sum().item()
                correct += (yh.argmax(-1) == y).float().sum().item()
                total += X.shape[0]
        return dict(acc=correct/total, bce=loss/total)

    for epoch in range(args.epoches):
        train_once(epoch)
        evals = validate(epoch)
        if evals['acc'] > best_acc:
            best_acc = evals['acc']
            torch.save(model.state_dict(), args.model_path / 'best_model.t7')

    torch.save(model.state_dict(), args.model_path / 'model.t7')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pathlib
import exman
from femida_detect.detect import select, data_loader


parser = exman.ExParser('Training for detection model', root=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'exman'
    ))
parser.add_argument('--paug', type=float, default=.5, help='Augmentation probability')
parser.add_argument('--imgsize', type=int, default=28, help='Image (Size x Size)')
parser.add_argument('--data-dir', type=pathlib.Path, help='Data directory', default='./data/')
parser.add_argument('--epoches', '-e', type=int, default=30, help='Number of epoches')
parser.add_argument('--lr', type=float, default=.001, help='Initial learning rate')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--log-every', type=int, default=10, help='Log frequency (batches)')
parser.add_argument('--augment', type=bool, default=True, help='Augmentation for training')
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--name', type=str, help='Name for experiment')
parser.add_argument('--threads', type=int, default=16, help='Data Loading Threads')
parser.add_argument('-v', type=str, help='Model Version', required=True)
parser.add_argument('-a', type=str, default=1, help='Aug Version')
parser.add_argument('--opt', type=str, choices=('adam', 'sgd'), help='Model Optimizer', default='adam')
parser.add_argument('--wd', type=float, default=0.005, help='Weight decay')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.register_validator(lambda args: args.epoches >= 1)
parser.register_validator(lambda args: args.batch_size >= 1)


class TeeStream(object):
    def __init__(self, stream, path):
        self.stream = stream
        self.tee = open(path, 'w')

    def write(self, buffer):
        self.tee.write(buffer)
        self.stream.write(buffer)

    def flush(self):
        self.tee.flush()
        self.stream.flush()


def check_args(args):
    args.model_dir = os.path.join(args.root, 'models')
    os.makedirs(args.model_dir, exist_ok=True)
    args.log_dir = os.path.join(args.root, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    sys.stderr = TeeStream(sys.stderr, os.path.join(args.log_dir, 'err'))
    sys.stdout = TeeStream(sys.stderr, os.path.join(args.log_dir, 'out'))
    return args


LOG_TEMPLATE = 'e[{perc:.2f}/{e}/{total}]\tloss={loss:.3f}\tacc={acc:.4f}'
LOG_TEMPLATE_VAL = '\nVALIDATION e[{e}/{total}]\tloss={loss:.3f}\tacc={acc:.4f}\n'


def parse_args(kwargs):
    command = ''
    for key, val in kwargs.items():
        command += f'--{key}={val} '
    return parser.parse_args(command)


def main(args):
    if isinstance(args, dict):
        args = parse_args(args)
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
    meta = dict(
        v='v%s' % args.v,
        init_params=dict(input_dim=inp_shape[1], input_size=inp_shape[2], output_dim=1)
    )
    model = select[meta['v']](**meta['init_params'])
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
                print(LOG_TEMPLATE.format(
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
        print(LOG_TEMPLATE_VAL.format(
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
            evals['model'] = state
            evals['meta'] = meta
            torch.save(evals, os.path.join(args.model_dir,  'best_model.t7'))
            print('NEW BEST MODEL!\n')
    state = model.state_dict()
    evals['model'] = state
    evals['meta'] = meta
    torch.save(state, os.path.join(args.model_dir, 'model.t7'))


if __name__ == '__main__':
    args = check_args(parser.parse_args())
    main(args)

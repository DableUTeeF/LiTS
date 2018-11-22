from __future__ import print_function
import os
import warnings

warnings.simplefilter("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import tmodel
import platform
import time
import datagen
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


if __name__ == '__main__':

    args = DotDict({
        'batch_size': 1,
        'batch_mul': 8,
        'val_batch_size': 1,
        'cuda': True,
        'model': '',
        'train_plot': False,
        'epochs': 90,
        'try_no': 'ce3',
        'imsize': 224,
        'imsize_l': 256,
        'traindir': '/root/palm/DATA/plant/train',
        'valdir': '/root/palm/DATA/plant/validate',
        'workers': 0,
        'resume': False,
    })
    best_loss = float('inf')
    best_no = 0
    start_epoch = 1
    try:
        print(f'loading log: log/try_{args.try_no}.json')
        log = eval(open(f'log/try_{args.try_no}.json', 'r').read())
    except FileNotFoundError:
        log = {'acc': [], 'loss': [], 'val_acc': []}
        print(f'Log {args.try_no} not found')
    model = tmodel.Unet().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                # nesterov=False,
                                )
    scheduler = MultiStepLR(optimizer, [10, 30, 60])

    criterion = nn.CrossEntropyLoss().cuda()
    zz = 0
    if platform.system() == 'Windows':
        train_gen = datagen.Generator(r'D:\LiTS\Training_Batch2\media\nas\01_Datasets\CT\LITS\Training Batch 2')
        test_gen = datagen.Generator(r'D:\LiTS\Training_Batch1\media\nas\01_Datasets\CT\LITS\Training Batch 1')
    else:
        train_gen = datagen.Generator('/root/palm/DATA/LITS/media/nas/01_Datasets/CT/LITS/Training Batch 2',
                                      format='ce'
                                      )
        test_gen = datagen.Generator('/root/palm/DATA/LITS/media/nas/01_Datasets/CT/LITS/Training Batch 1',
                                     format='ce'
                                     )
    trainloader = torch.utils.data.DataLoader(train_gen,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        test_gen,
        batch_size=args.val_batch_size,
        num_workers=args.workers,
        pin_memory=False)

    # model = torch.nn.parallel.DistributedDataParallel(model).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = args.batch_size > 1
    if args.resume:
        if args.resume is True:
            args['resume'] = f'./checkpoint/try_{args.try_no}best.t7'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['acc']
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    def train(epoch):
        print('\nEpoch: %d/%d' % (epoch, args.epochs))
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        start_time = time.time()
        last_time = start_time
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            inputs = inputs[0]
            targets = targets[0]
            iterations = max(int(np.ceil(max(inputs.shape[0], 0) / args.batch_mul)), 1)
            for i in range(iterations):
                outputs = model(inputs[i:i + args.batch_mul, :, :, :])
                loss = criterion(outputs, targets[i:i + args.batch_mul, 0, :, :].long()) / iterations
                loss.backward()
                train_loss += loss.item() * args.batch_mul
                total += targets.size(0)

            optimizer.step()
            optimizer.zero_grad()
            step_time = time.time() - last_time
            last_time = time.time()
            try:
                print(f'\r{" "*(len(lss))}', end='')
            except NameError:
                pass
            lss = f'{batch_idx}/{len(trainloader)} | ' + \
                  f'ETA: {format_time(step_time*(len(trainloader)-batch_idx))} - ' + \
                  f'loss: {train_loss/(batch_idx+1):.{5}}'
            print(f'\r{lss}', end='')

        print(f'\r '
              f'ToT: {format_time(time.time() - start_time)} - '
              f'loss: {train_loss/(batch_idx+1):.{5}}', end='')
        optimizer.step()
        optimizer.zero_grad()
        # scheduler2.step()
        log['acc'].append(100. * correct / total)
        log['loss'].append(train_loss / (batch_idx + 1))


    def test(epoch):
        global best_loss, best_no
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                inputs = inputs[0]
                targets = targets[0]
                iterations = max(int(np.ceil(max(inputs.shape[0], 0) / args.batch_mul)), 1)
                for i in range(iterations):
                    outputs = model(inputs[i:i + args.batch_mul, :, :, :])
                    loss = criterion(outputs, targets[i:i + args.batch_mul, 0, :, :].long()) / iterations
                    test_loss += loss.item() * args.batch_mul
                    total += targets.size(0)

                # progress_bar(batch_idx, len(val_loader), 'Acc: %.3f%%'
                #              % (100. * correct / total))
        print(f' - val_loss: {test_loss / (batch_idx+1):.{5}}')
        # platue.step(correct)
        log['val_acc'].append(100. * correct / total)
        loss = test_loss / (batch_idx + 1)
        # Save checkpoint.
        # print('Saving..')
        state = {
            'optimizer': optimizer.state_dict(),
            'net': model.state_dict(),
            'acc': loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if loss < best_loss and loss != 0:
            torch.save(state, f'./checkpoint/try_{args.try_no}best.t7')
            best_loss = loss
        torch.save(state, f'./checkpoint/try_{args.try_no}temp.t7')
        with open('log/try_{}.json'.format(args.try_no), 'w') as wr:
            wr.write(log.__str__())


    for epoch in range(start_epoch, start_epoch + args.epochs):
        scheduler.step()
        train(epoch)
        test(epoch)
        print(f'best: {best_loss}')

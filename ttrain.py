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
from sklearn.metrics import f1_score


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


def f1(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy().astype('float32').flatten()
    y_pred = y_pred.cpu().detach().numpy().flatten()
    y_pred = np.round(y_pred)
    # print(y_true.dtype, y_pred.dtype)

    return f1_score(y_true, y_pred, average='micro')


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
        'batch_size': 32,
        'batch_mul': 1,
        'val_batch_size': 1,
        'data_format': 'bin',
        'cuda': True,
        'model': '',
        'train_plot': False,
        'epochs': 90,
        'try_no': '2dmse1',
        'imsize': 224,
        'imsize_l': 256,
        'traindir': '/root/palm/DATA/plant/train',
        'valdir': '/root/palm/DATA/plant/validate',
        'workers': 16,
        'resume': False,
    })
    best_loss = float('inf')
    best_no = 0
    start_epoch = 1
    best_acc = 0
    try:
        print(f'loading log: log/try_{args.try_no}.json')
        log = eval(open(f'log/try_{args.try_no}.json', 'r').read())
    except FileNotFoundError:
        log = {'acc': [], 'loss': [], 'val_acc': []}
        print(f'Log {args.try_no} not found')
    model = tmodel.Unet(3).cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4,
                                # nesterov=False,
                                )
    scheduler = MultiStepLR(optimizer, [10, 30, 60])

    criterion = nn.MSELoss().cuda()
    zz = 0
    if platform.system() == 'Windows':
        train_gen = datagen.Generator(r'D:\LiTS\Training_Batch2\media\nas\01_Datasets\CT\LITS\Training Batch 2',
                                      out_format=args.data_format)
        test_gen = datagen.Generator(r'D:\LiTS\Training_Batch1\media\nas\01_Datasets\CT\LITS\Training Batch 1',
                                     out_format=args.data_format)
    else:
        train_gen = datagen.Generator('/root/palm/DATA/LITS/media/nas/01_Datasets/CT/LITS/Training Batch 2',
                                      out_format=args.data_format,
                                      dim=2
                                      )
        test_gen = datagen.Generator('/root/palm/DATA/LITS/media/nas/01_Datasets/CT/LITS/Training Batch 1',
                                     out_format=args.data_format,
                                     dim=2
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
        acc = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            # inputs = inputs[0]
            # targets = targets[0]
            iterations = max(int(np.ceil(max(inputs.shape[0], 0) / args.batch_mul)), 1)
            # for i in range(inputs.shape[0]-8):
            #     outputs = model(inputs[i:i + args.batch_mul, :, :, :])
            #     loss = criterion(outputs, targets[i:i + args.batch_mul, :, :, :].float()) / (inputs.shape[0]-8)
            #     loss.backward()
            #     train_loss += loss.item() * args.batch_mul
            #     # acc += f1(targets[i:i + args.batch_mul, :, :, :], outputs)
            #     total += targets.size(0)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            train_loss += loss.item() * args.batch_mul
            # acc += f1(targets[i:i + args.batch_mul, :, :, :], outputs)
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
        # log['acc'].append(100. * correct / total)
        log['loss'].append(train_loss / (batch_idx + 1))


    def test(epoch):
        global best_loss, best_no, best_acc
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        acc = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                # inputs = inputs[0]
                # targets = targets[0]
                iterations = max(int(np.ceil(max(inputs.shape[0], 0) / args.batch_mul)), 1)
                # for i in range(inputs.shape[0]-8):
                #     outputs = model(inputs[i:i + args.batch_mul, :, :, :])
                #     loss = criterion(outputs, targets[i:i + args.batch_mul, :, :, :].float()) / (inputs.shape[0]-8)
                #     test_loss += loss.item() * args.batch_mul
                #     acc += f1(targets[i:i + args.batch_mul, :, :, :], outputs)
                #     total += targets.size(0)
                outputs = model(inputs.float())
                loss = criterion(outputs, targets.float())
                test_loss += loss.item() * args.batch_mul
                total += targets.size(0)

                # progress_bar(batch_idx, len(val_loader), 'Acc: %.3f%%'
                #              % (100. * correct / total))
        print(f' - val_acc: {acc / (total/8):.{5}} - val_loss: {test_loss / (batch_idx+1):.{5}}')
        # platue.step(correct)
        loss = test_loss / (batch_idx + 1)
        acc = acc / (total/8)
        log['val_acc'].append(acc)
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
        if acc > best_acc and loss != 0:
            torch.save(state, f'./checkpoint/try_{args.try_no}best.t7')
            best_loss = loss
            best_acc = acc
        torch.save(state, f'./checkpoint/try_{args.try_no}temp.t7')
        with open('log/try_{}.json'.format(args.try_no), 'w') as wr:
            wr.write(log.__str__())


    for epoch in range(start_epoch, start_epoch + args.epochs):
        scheduler.step()
        train(epoch)
        test(epoch)
        print(f'best: {best_loss} - {best_acc}')

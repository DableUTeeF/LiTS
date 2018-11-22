import torch.nn as nn
import torch.utils.data.distributed
from torchvision.models.densenet import densenet201
from torchvision import transforms
from torchvision import datasets
import json
import tmodel
import platform
import nibabel as nib
import datagen
import numpy as np


if __name__ == '__main__':
    dev = 'cuda'
    model = tmodel.Unet().to(dev)
    # print(model)
    checkpoint = torch.load('checkpoint/try_ce3temp.t7')
    model.load_state_dict(checkpoint['net'])
    if platform.system() == 'Windows':
        rootpath = r'D:\LiTS'
    else:
        rootpath = r'/media/palm/Unimportant/LITS'
    # img = nib.load('Test/test-volume-0.nii')
    out = []
    c = 0
    test_gen = datagen.Generator(r'D:\LiTS\Training_Batch1\media\nas\01_Datasets\CT\LITS\Training Batch 1',
                                 format='torch')
    model.eval()
    correct = 0
    total = 0
    val_loader = torch.utils.data.DataLoader(
        test_gen,
        batch_size=1,
        num_workers=0,
        pin_memory=False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(dev), targets.to(dev)
            inputs = inputs[0]
            targets = targets[0]
            t = inputs.shape
            iterations = max(int(np.ceil(max(inputs.shape[0], 0) / 8)), 1)
            mask = np.zeros(targets.shape)
            for i in range(iterations):
                outputs = model(inputs[i:i + 8, :, :, :])
                mask[i:i + 8, :, :, :] += outputs.cpu().detach().numpy()
                if 0 < i < iterations - 2:
                    mask[i:i + 7, :, :, :] /= 2
            segment = np.argmax(mask, axis=1)
            import pickle as pk
            with open('segment.pk', 'wb') as wr:
                pk.dump(segment, wr)
            with open('mask.pk', 'wb') as wr:
                pk.dump(mask, wr)
            break

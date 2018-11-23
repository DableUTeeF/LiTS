from keras.utils import Sequence
import numpy as np
import nibabel as nib
import os
import cv2

size = 512


class Generator(Sequence):
    def __init__(self, directory, groundtruth=True, shuffle=True, format='keras'):
        self.directory = directory
        self.fileslist = os.listdir(self.directory)
        self.groundtruth = groundtruth
        self.shuffle = shuffle
        self.format = format
        self.segmentations = []
        self.volumes = []
        for file in self.fileslist:
            if file.startswith('segmentation'):
                self.segmentations.append(file)
            elif file.startswith('volume') or file.startswith('test-volume'):
                self.volumes.append(file)
        self.volumes = sorted(self.volumes)
        self.segmentations = sorted(self.segmentations)
        self.seen = []

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        if len(self.seen) == len(self.volumes):
            self.seen = []
        if self.shuffle:
            while 1:
                rnd = int(np.round(np.random.rand() * len(self))) - 1
                if rnd not in self.seen:
                    break
            self.seen.append(rnd)
            index = rnd
        img = nib.load(os.path.join(self.directory, self.volumes[index]))
        image = img.get_fdata()
        image = (np.array(image, dtype='uint16') >> 8).astype('float32').reshape((*image.shape, 1))
        # todo
        image = np.rollaxis(image, 2)
        image /= 127.5
        image -= 1
        # todo
        if not self.groundtruth:
            return image
        else:
            img = nib.load(os.path.join(self.directory, self.segmentations[index]))
            gt = img.get_fdata()
            gt = np.array(gt, dtype='float32').reshape((*gt.shape, 1))
            gt = np.rollaxis(gt, 2)
            # todo
            temp = np.zeros((*gt.shape[:-1], 3), dtype='float32')
            temp[:, :, :, 0] = gt[:, :, :, 0] == 0
            temp[:, :, :, 1] = gt[:, :, :, 0] == 1
            temp[:, :, :, 2] = gt[:, :, :, 0] == 2
            if self.format == 'torch':
                image = np.rollaxis(image, 3, 1)
                temp = np.rollaxis(temp, 3, 1)
            elif self.format == 'ce':
                image = np.rollaxis(image, 3, 1)
                temp = np.rollaxis(gt, 3, 1)
            elif self.format == 'bin':
                image = np.rollaxis(image, 3, 1)
                temp = np.rollaxis(gt, 3, 1).astype('bool').astype('uint8')
            return image, temp


class SummaryGenerator(Sequence):
    def __init__(self, directory, groundtruth=True, shuffle=True, format='keras'):
        self.directory = directory
        self.fileslist = os.listdir(self.directory)
        self.groundtruth = groundtruth
        self.shuffle = shuffle
        self.format = format
        self.segmentations = []
        self.volumes = []
        for file in self.fileslist:
            if file.startswith('segmentation'):
                self.segmentations.append(file)
            elif file.startswith('volume') or file.startswith('test-volume'):
                self.volumes.append(file)
        self.volumes = sorted(self.volumes)
        self.segmentations = sorted(self.segmentations)
        self.seen = []

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        if len(self.seen) == len(self.volumes):
            self.seen = []
        if self.shuffle:
            while 1:
                rnd = int(np.round(np.random.rand() * len(self))) - 1
                if rnd not in self.seen:
                    break
            self.seen.append(rnd)
            index = rnd
        img = nib.load(os.path.join(self.directory, self.volumes[index]))
        image = img.get_fdata()
        image = np.array(image, dtype='uint16').reshape((*image.shape, 1))
        xplane = np.mean(image, axis=1).astype('uint16') / 255# >> 8
        # yplane = cv2.resize(np.mean(image, axis=1), (512, 512)).reshape((1, 512, 512, 1))
        # zplane = cv2.resize(np.mean(image, axis=2), (512, 512)).reshape((1, 512, 512, 1))
        # image = np.concatenate((xplane, yplane, zplane), axis=3)
        # image = np.rollaxis(image, 3, 1)
        if not self.groundtruth:
            return xplane
        else:
            img = nib.load(os.path.join(self.directory, self.segmentations[index]))
            gt = img.get_fdata()
            gt = np.array(gt, dtype='float32').reshape((*gt.shape, 1))
            gt = np.rollaxis(gt, 2)
            # todo
            temp = np.zeros((*gt.shape[:-1], 3), dtype='float32')
            temp[:, :, :, 0] = gt[:, :, :, 0] == 0
            temp[:, :, :, 1] = gt[:, :, :, 0] == 1
            temp[:, :, :, 2] = gt[:, :, :, 0] == 2
            if self.format == 'torch':
                image = np.rollaxis(image, 3, 1)
                temp = np.rollaxis(temp, 3, 1)
            elif self.format == 'ce':
                image = np.rollaxis(image, 3, 1)
                temp = np.rollaxis(gt, 3, 1)
            elif self.format == 'bin':
                image = np.rollaxis(image, 3, 1)
                temp = np.rollaxis(gt, 3, 1).astype('bool').astype('uint8')
            return image, temp


if __name__ == '__main__':
    from PIL import Image
    train_gen = SummaryGenerator(r'D:\LiTS\Training_Batch2\media\nas\01_Datasets\CT\LITS\Training Batch 2',
                                 shuffle=False,
                                 groundtruth=False)
    x = train_gen[0]
    x = np.rollaxis(x, 1, 0)
    # y = np.reshape(y, y.shape[:-1])
    print(np.max(x))
    # xt = np.mean(x.astype('float32').reshape((3, size, size)), axis=-1).astype('int8')
    # print(np.max(xt))
    xt = x[:, :, 0]
    imx = Image.fromarray(xt, mode='L')
    imx.show()

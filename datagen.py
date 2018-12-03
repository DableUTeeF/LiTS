from keras.utils import Sequence
import numpy as np
import nibabel as nib
import os
import cv2

size = 512


def norm(x, dtype='float32'):
    x = x.astype(dtype)
    x /= 127.5
    x -= 1
    return x


class Generator(Sequence):
    def __init__(self, directory,
                 groundtruth=True,
                 shuffle=True,
                 out_format='keras',
                 norm=norm,
                 dim=3,
                 bits_manage='divide',
                 ):
        self.directory = directory
        self.fileslist = os.listdir(self.directory)
        self.groundtruth = groundtruth
        self.shuffle = shuffle
        self.out_format = out_format
        self.norm = norm
        self.dim = dim
        self.bits_manage = bits_manage

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

    def _3dimage(self, image, index):
        if self.bits_manage == 'shift':
            former = np.array(image >> 8, dtype='uint16').astype('float32').reshape((*image.shape, 1))
            latter = np.array(image << 8 >> 8, dtype='uint16').astype('float32').reshape((*image.shape, 1))
            image = np.concatenate((former, latter))
        else:
            image = np.array(image >> 8, dtype='uint16').astype('float32').reshape((*image.shape, 1))

        if self.out_format != 'keras':
            image = np.rollaxis(image, 2)

        if self.norm:
            image = self.norm(image)
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
            if self.out_format == 'torch':
                image = np.rollaxis(image, 3, 1)
                temp = np.rollaxis(temp, 3, 1)
            elif self.out_format == 'ce':
                image = np.rollaxis(image, 3, 1)
                temp = np.rollaxis(gt, 3, 1)
            elif self.out_format == 'bin':
                image = np.rollaxis(image, 3, 1)
                temp = np.rollaxis(gt, 3, 1).astype('bool').astype('uint8')
            return image, temp

    def _2dimage(self, image, index):
        image = np.array(image, dtype='uint16').reshape(image.shape)
        if self.bits_manage == 'shift':
            xplanef = np.mean(image >> 8, axis=0).astype('uint8')
            xplanef = cv2.resize(xplanef, (512, 512))
            xplanel = np.mean(image << 8 >> 8, axis=0).astype('uint8')
            xplanel = cv2.resize(xplanel, (512, 512))
            xplane = np.dstack((xplanef, xplanel)).reshape((1, 512, 512, 2))
            yplanef = np.mean(image >> 8, axis=1).astype('uint16')
            yplanef = cv2.resize(yplanef, (512, 512))
            yplanel = np.mean(image << 8 >> 8, axis=1).astype('uint16')
            yplanel = cv2.resize(yplanel, (512, 512))
            yplane = np.dstack((yplanef, yplanel)).reshape((1, 512, 512, 2))
            zplanef = np.mean(image >> 8, axis=2).astype('uint16')
            zplanel = np.mean(image << 8 >> 8, axis=2).astype('uint16')
            zplane = np.dstack((zplanef, zplanel)).reshape((1, 512, 512, 2))
            image = np.concatenate((xplane, yplane, zplane), axis=3)
        else:
            xplane = np.mean(image >> 8, axis=0).astype('uint16')
            xplane = cv2.resize(xplane, (512, 512))
            yplane = np.mean(image >> 8, axis=1).astype('uint16')
            yplane = cv2.resize(yplane, (512, 512))
            zplane = np.mean(image >> 8, axis=2).astype('uint16')
            image = np.dstack((xplane, yplane, zplane)).reshape((1, 512, 512, 3))
        if self.out_format != 'keras':
            image = np.rollaxis(image, 3, 1).astype('uint8')
        if not self.groundtruth:
            return image
        else:
            img = nib.load(os.path.join(self.directory, self.segmentations[index]))
            gt = img.get_fdata()
            gt = np.array(gt, dtype='uint8').reshape(gt.shape)
            xplane = np.max(gt, axis=0).astype('uint8')
            xplane = cv2.resize(xplane, (512, 512))
            yplane = np.max(gt, axis=1).astype('uint8')
            yplane = cv2.resize(yplane, (512, 512))
            zplane = np.max(gt, axis=2).astype('uint8')
            gt = np.dstack((xplane, yplane, zplane)).reshape((1, 512, 512, 3))
            if self.out_format != 'keras':
                gt = np.rollaxis(gt, 3, 1)
            return image, np.clip(gt, 0, 1)

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
        if self.dim == 3:
            return self._3dimage(image, index)
        else:
            return self._2dimage(image, index)


if __name__ == '__main__':
    from PIL import Image
    train_gen = Generator(r'D:\LiTS\Training_Batch2\media\nas\01_Datasets\CT\LITS\Training Batch 2',
                          shuffle=False,
                          groundtruth=False,
                          dim=2,
                          out_format='torch',
                          )
    x = train_gen[0]
    x = np.rollaxis(x, 1, 0)
    # y = np.reshape(y, y.shape[:-1])
    print(np.max(x))
    # xt = np.mean(x.astype('float32').reshape((3, size, size)), axis=-1).astype('int8')
    # print(np.max(xt))
    xt = x[:, :, 0]
    imx = Image.fromarray(xt, mode='L')
    imx.show()

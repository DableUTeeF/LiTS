from keras.utils import Sequence
import numpy as np
import nibabel as nib
import os
import cv2


size = 512


class Generator(Sequence):
    def __init__(self, directory, groundtruth=True, shuffle=True):
        self.directory = directory
        self.fileslist = os.listdir(self.directory)
        self.groundtruth = groundtruth
        self.shuffle = shuffle
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
        image = (np.array(image, dtype='uint16') / 256).astype('uint8').reshape((1, *image.shape, 1))
        # todo
        resizedimage = np.zeros((1, size, size, size, 1), dtype='uint8')
        temp = np.zeros((1, size, size, image.shape[-2], 1), dtype='uint8')
        for i in range(image.shape[-1]):
            temp[0, :, :, i, 0] = cv2.resize(image[0, :, :, i], (size, size))
        for i in range(size):
            resizedimage[0, :, i, :, 0] = cv2.resize(temp[0, :, i, :], (size, size))
        # todo

        if not self.groundtruth:
            return image
        else:
            img = nib.load(os.path.join(self.directory, self.segmentations[index]))
            gt = img.get_fdata()
            # todo
            groundtruth = np.zeros((1, size, size, size, 1), dtype='uint8')
            temp = np.zeros((1, size, size, gt.shape[-1], 1), dtype='uint8')
            for i in range(gt.shape[-1]):
                temp[0, :, :, i, 0] = cv2.resize(gt[:, :, i], (size, size))
            for i in range(size):
                groundtruth[0, :, i, :, 0] = cv2.resize(temp[0, :, i, :], (size, size))
            # todo
            gt = np.array(gt, dtype='uint8').reshape((1, *gt.shape, 1)) - 1
            return image, gt

#
# if __name__ == '__main__':
#     from PIL import Image
#     train_gen = Generator(r'D:\LiTS\Training_Batch2\media\nas\01_Datasets\CT\LITS\Training Batch 2')
#     x, y = train_gen[0]
#     print(np.max(x))
#     xt = np.mean(x.astype('float32').reshape((size, size, x.shape[-2])), axis=-1).astype('int8')
#     print(np.max(xt))
#     imx = Image.fromarray(xt, mode='L')
#     imx.show()

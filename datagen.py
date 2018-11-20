from keras.utils import Sequence
import numpy as np
import nibabel as nib
import os


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
                rnd = int(np.round(np.random.rand() * len(self)))
                if rnd not in self.seen:
                    break
            self.seen.append(rnd)
            index = rnd
        img = nib.load(os.path.join(self.directory, self.volumes[index]))
        image = img.get_fdata()
        image = (np.array(image, dtype='uint16') / 256).astype('uint8').reshape((1, *image.shape, 1))
        if not self.groundtruth:
            return image
        else:
            img = nib.load(os.path.join(self.directory, self.segmentations[index]))
            gt = img.get_fdata()
            gt = np.array(gt, dtype='uint8').reshape((1, *gt.shape, 1)) - 1
            return image, gt

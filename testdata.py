import nibabel as nib
import numpy as np

img = nib.load('/media/palm/Unimportant/LITS/Test/test-volume-0.nii')
image = img.get_fdata()
image = np.array(image, dtype='uint16')
shape = image.shape
image1 = np.zeros(image.shape, dtype='uint8')
image2 = np.zeros(image.shape, dtype='uint8')

for height in range(shape[0]):
    for width in range(shape[1]):
        for depth in range(shape[2]):
            hexval = bin(int(image[height, width, depth]))[2:]
            image1[height, width, depth] = int(hexval[:8], 2)
            if len(hexval) > 8:
                image2[height, width, depth] = int(hexval[8:], 2)
            else:
                image2[height, width, depth] = 0

import nibabel as nib
import numpy as np
import platform
from PIL import Image
import cv2
if platform.system() == 'Windows':
    rootpath = r'D:\LiTS'
else:
    rootpath = r'/media/palm/Unimportant/LITS'
# img = nib.load(rootpath+'/Test/test-volume-0.nii')
img = nib.load(rootpath+'/Test/test-volume-0.nii')
image = img.get_fdata()
image = np.array(image, dtype='uint16')
shape = image.shape
image1 = np.zeros(image.shape, dtype='uint8')
image2 = np.zeros(image.shape, dtype='uint8')

# for height in range(shape[0]):
#     for width in range(shape[1]):
#         for depth in range(shape[2]):
#             hexval = bin(int(image[height, width, depth]))[2:]
#             image1[height, width, depth] = int(hexval[:8], 2)
#             if len(hexval) > 8:
#                 image2[height, width, depth] = int(hexval[8:], 2)
#             else:
#                 image2[height, width, depth] = 0
for i in range(3):
    imx = np.mean(image >> 8, i).astype('uint8')
    imx = Image.fromarray(imx, mode='L')
    imx = imx.resize((512, 512))
    imx.show()
    im = image << 8 >> 8
    # im = im
    im = np.mean(im, i).astype('uint8')
    s = im.shape
    im = Image.fromarray(im, mode='L')
    im = im.resize((512, 512))
    im.show()
    dummy = np.zeros((512, 512), dtype='uint8') + 0
    img = np.dstack((dummy, imx, im))
    img = Image.fromarray(img)
    img.show()
    # img = np.dstack((dummy, imx, dummy))
    # img = Image.fromarray(img)
    # img.show()

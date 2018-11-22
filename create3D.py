import binvox_rw
import pickle as pk
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np

if __name__ == '__main__':
    file = 'segment.pk'
    with open(file, 'rb') as wr:
        array = pk.load(wr)
    print(array.shape)
    print(np.max(array))
    for i in range(3):
        im = np.max(array, i) * 127
        img = Image.fromarray(im, mode='L')
        img.show()

    # array = array.astype('bool')
    # vox = binvox_rw.Voxels(data=array,
    #                        dims=[841, 512, 512],
    #                        scale=1,
    #                        translate=1,
    #                        axis_order='xyz'
    #                        )
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    #
    # ax.voxels(array, edgecolor="k")
    #
    # plt.show()


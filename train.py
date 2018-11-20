import model as m
import datagen
from keras import optimizers as ko

if __name__ == '__main__':
    model = m.conv1()
    model.compile(ko.adam(),
                  loss='mse'
                  )
    print(model.summary())
    train_gen = datagen.Generator(r'D:\LiTS\Training_Batch2\media\nas\01_Datasets\CT\LITS\Training Batch 2')
    test_gen = datagen.Generator(r'D:\LiTS\Training_Batch1\media\nas\01_Datasets\CT\LITS\Training Batch 1')
    model.fit_generator(train_gen, validation_data=test_gen)
    model.save_weights('weights/test.h5')

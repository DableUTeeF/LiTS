from keras import layers as kl, models as km


def lstm1():
    inp = kl.Input((128, 128, None, 1))
    x = kl.Conv3D(8, 3, strides=2, padding='same', activation='relu')(inp)
    x = kl.Conv3D(8, 3, strides=2, padding='same', activation='relu')(x)
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(32, return_sequences=True)))(x)
    x = kl.Permute((2, 1, 3, 4))(x)
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(32, return_sequences=True)))(x)
    x = kl.Permute((1, 3, 2, 4))(x)
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(32, return_sequences=True)))(x)
    x = kl.Permute((3, 1, 2, 4))(x)
    x = kl.Conv3D(4, 1, activation='relu')(x)
    x = kl.Reshape((128, 128, -1, 1))(x)
    x = kl.Conv3D(1, 1, activation='tanh')(x)
    return km.Model(inp, x)


def lstm2():
    inp = kl.Input((512, 512, None, 1))
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(32, return_sequences=True)))(inp)
    x = kl.Permute((2, 1, 3, 4))(x)
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(32, return_sequences=True)))(x)
    x = kl.Permute((1, 3, 2, 4))(x)
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(32, return_sequences=True)))(x)
    x = kl.Permute((3, 1, 2, 4))(x)
    x = kl.Conv3D(1, 1, activation='tanh')(x)
    return km.Model(inp, x)


def lstm3():
    inp = kl.Input((512, 512, 1))
    x = kl.TimeDistributed(kl.LSTM(128, return_sequences=True))(inp)
    x = kl.Permute((2, 1, 3))(x)
    x = kl.TimeDistributed(kl.LSTM(128, return_sequences=True))(x)
    x = kl.Permute((2, 1, 3))(x)
    x = kl.Conv2D(1, 1, activation='tanh')(x)
    return km.Model(inp, x)


def conv1():
    inp = kl.Input((512, 512, None, 1))
    x = kl.Conv3D(8, 3, strides=1, padding='same', activation='relu', use_bias=False)(inp)
    x = kl.Conv3D(8, 3, strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Conv3D(8, 3, strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Conv3D(8, 3, strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Conv3D(8, 3, strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Conv3D(8, 3, strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Conv3D(1, 1, activation='tanh', use_bias=False)(x)
    return km.Model(inp, x)


def conv2():
    inp = kl.Input((512, 512, 1))
    x = kl.Conv2D(8, 3, strides=2, padding='same', activation='relu', use_bias=False)(inp)
    x = kl.Conv2D(16, 3, strides=2, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Conv2D(64, 3, strides=2, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Conv2D(64, 3, strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Conv2D(64, 3, strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Conv2D(64, 3, strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = kl.Reshape((512, 512, 1))(x)
    x = kl.Conv2D(1, 1, activation='tanh', use_bias=False)(x)
    return km.Model(inp, x)


def ublock(x, kernels, depth):
    x = kl.Conv2D(kernels, 3, strides=2, padding='same', use_bias=False)(x)
    a = kl.BatchNormalization()(x)
    a = kl.Activation('relu')(a)
    a = kl.Conv2D(kernels, 3, strides=1, padding='same', use_bias=False)(a)
    a = kl.BatchNormalization()(a)
    a = kl.Activation('relu')(a)
    if depth > 0:
        a = ublock(a, kernels*2, depth-1)
        a = kl.UpSampling2D()(a)
        a = kl.Conv2D(kernels, 2, padding='same', use_bias=False)(a)
        a = kl.add([x, a])
    return a


def unet():
    inp = kl.Input((512, 512, 1))
    x = ublock(inp, 32, 5)
    x = kl.UpSampling2D()(x)
    x = kl.Conv2D(3, 1, activation='sigmoid')(x)
    return km.Model(inp, x)

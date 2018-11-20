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
    inp = kl.Input((128, 128, None, 1))
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(32, return_sequences=True)))(inp)
    x = kl.Permute((2, 1, 3, 4))(x)
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(32, return_sequences=True)))(x)
    x = kl.Permute((1, 3, 2, 4))(x)
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(32, return_sequences=True)))(x)
    x = kl.Permute((3, 1, 2, 4))(x)
    x = kl.Conv3D(1, 1, activation='tanh')(x)
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

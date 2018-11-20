from keras import layers as kl, models as km


def res20():
    inp = kl.Input((512, 512, None, 1))
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(64, return_sequences=True)))(inp)
    x = kl.Permute((2, 1, 3, 4))(x)
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(64, return_sequences=True)))(x)
    x = kl.Permute((1, 3, 2, 4))(x)
    x = kl.TimeDistributed(kl.TimeDistributed(kl.LSTM(64, return_sequences=True)))(x)
    x = kl.Permute((3, 1, 2, 4))(x)
    x = kl.Conv3D(1, 1, activation='tanh')(x)
    return km.Model(inp, x)

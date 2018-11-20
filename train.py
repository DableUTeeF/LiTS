import model as m
import datagen
from keras import optimizers as ko
from keras import backend as K
import numpy as np
import platform

class SGDAccum(ko.Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, accum_iters=1, **kwargs):
        super(SGDAccum, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.accum_iters = K.variable(accum_iters)
        self.initial_decay = decay
        self.nesterov = nesterov

    @ko.interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        accum_switch = K.equal(self.iterations % self.accum_iters, 0)
        accum_switch = K.cast(accum_switch, dtype='float32')

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        temp_grads = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, cg, m, tg in zip(params, grads, moments, temp_grads):
            g = cg + tg
            v = self.momentum * m - (lr * g / self.accum_iters)  # velocity
            self.updates.append(K.update(m, (1 - accum_switch) * m + accum_switch * v))
            self.updates.append(K.update(tg, (1 - accum_switch) * g))

            if self.nesterov:
                new_p = p + self.momentum * v - (lr * g / self.accum_iters)
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - accum_switch) * p + accum_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'accum_iters': self.accum_iters}
        base_config = super(SGDAccum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    batch_size = 8
    model = m.unet()
    print(model.summary())
    if platform.system() == 'Windows':
        train_gen = datagen.Generator(r'D:\LiTS\Training_Batch2\media\nas\01_Datasets\CT\LITS\Training Batch 2')
        test_gen = datagen.Generator(r'D:\LiTS\Training_Batch1\media\nas\01_Datasets\CT\LITS\Training Batch 1')
    else:
        train_gen = datagen.Generator('/root/palm/DATA/LITS/media/nas/01_Datasets/CT/LITS/Training Batch 2')
        test_gen = datagen.Generator('/root/palm/DATA/LITS/media/nas/01_Datasets/CT/LITS/Training Batch 1')

    for epoch in range(99):
        print('Epoch:', epoch)
        for idx, (x, y) in enumerate(train_gen):
            train_iter_count = 0
            train_loss = 0
            iterations = max(int(np.ceil(max(x.shape[0], 0) / batch_size)), 1)
            model.compile(SGDAccum(lr=0.01,
                                   momentum=0.9,
                                   accum_iters=iterations,
                                   ),
                          loss='mse'
                          )
            for i in range(iterations):
                loss = model.train_on_batch(x=x[i:i + batch_size, :, :, :],
                                            y=y[i:i + batch_size, :, :, :]
                                            )
                train_loss += loss
                train_iter_count += 1
                if not platform.system() == 'Windows':
                    print(idx, str(i)+'/'+str(iterations), 'Train loss:', '%.4f' % (train_loss / train_iter_count), end='\r')

            if platform.system() == 'Windows':
                print('Train loss:', train_loss / train_iter_count)
        for x, y in test_gen:
            test_iter_count = 0
            test_loss = 0
            iterations = max(int(np.ceil(max(x.shape[0], 0) / batch_size)), 1)
            for i in range(iterations):
                loss = model.train_on_batch(x=x[i:i + batch_size, :, :, :],
                                            y=y[i:i + batch_size, :, :, :]
                                            )
                test_loss += loss
                test_iter_count += 1
            if platform.system() == 'Windows':
                print('Test loss:', test_loss / test_iter_count)
            else:
                print('Train loss:', '%.4f' % (train_loss / train_iter_count), '- Test loss:', '%.4f' % (test_loss / test_iter_count), end='\r')

        # model.fit_generator(train_gen, validation_data=test_gen)
        model.save_weights('weights/test.h5')

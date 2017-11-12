# -*- coding: utf-8 -*-
from keras import backend as K
from keras.engine import Layer
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Lambda


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def f(X):
        b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1))
                                              , (0, half))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, i:i + ch, :, :]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = int(X.shape[axis]) // ratio_split

        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


'''
def convolution2Dgroup(n_group, nb_filter, nb_row, nb_col, **kwargs):
    def f(input):
        return Merge([
                         Convolution2D(nb_filter // n_group, nb_row, nb_col)(
                             splittensor(axis=1,
                                         ratio_split=n_group,
                                         id_split=i)(input))
                         for i in range(n_group)
                         ], mode='concat', concat_axis=1)

    return f
'''

class Softmax4D(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape

class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)
        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=1)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

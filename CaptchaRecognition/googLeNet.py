#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/17 下午9:21
# @Author  : tudoudou
# @File    : googLeNet.py
# @Software: PyCharm

import warnings

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape




def conv2D_bn(inputs, filters, kernel_size, padding='same', axis=1):
    t = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, data_format='channels_first')(inputs)
    t = BatchNormalization(axis=axis, scale=False)(t)
    t = Activation('relu')(t)
    return t


def my_InceptionV3():
    inputs = Input(shape=(1, 40, 50))

    t = conv2D_bn(inputs=inputs, filters=8, kernel_size=(5, 5),padding='valid') #
    t = conv2D_bn(t, 8, (5, 5),padding='valid')
    t = conv2D_bn(t, 16, (2, 2))
    t = MaxPooling2D((2, 2), (1, 1))(t)

    branch1x1 = conv2D_bn(t, 16, (1, 1))

    branch5x5 = conv2D_bn(t, 12, (1, 1))
    branch5x5 = conv2D_bn(branch5x5, 16, (5, 5))

    branch3x3dbl = conv2D_bn(t, 16, (1, 1))
    branch3x3dbl = conv2D_bn(branch3x3dbl, 16, (3, 3))
    branch3x3dbl = conv2D_bn(branch3x3dbl, 16, (3, 3))

    branch_pool = AveragePooling2D((2, 2), strides=(1, 1),padding='same')(t)
    branch_pool = conv2D_bn(branch_pool, 8, (1, 1))
    t = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=1,
        name='mixed0')

    # branch1x1 = conv2D_bn(t, 16, (1, 1))
    #
    # branch5x5 = conv2D_bn(t, 12, (1, 1))
    # branch5x5 = conv2D_bn(branch5x5, 16, (5, 5))
    #
    # branch3x3dbl = conv2D_bn(t, 16, (1, 1))
    # branch3x3dbl = conv2D_bn(branch3x3dbl, 16, (3, 3))
    # branch3x3dbl = conv2D_bn(branch3x3dbl, 16, (3, 3))
    #
    # branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(t)
    # branch_pool = conv2D_bn(branch_pool, 16, (1, 1))
    # t = layers.concatenate(
    #     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
    #     axis=1,
    #     name='mixed1')

    t = GlobalAveragePooling2D(name='avg_pool')(t)

    t = [Dense(11, activation='softmax', name='sotfmax11_%d' % (i + 1))(t) for i in range(4)]

    model = Model(inputs, t, name='my_InceptionV3')
    return model


# model = my_InceptionV3()
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# from keras.utils import plot_model
#
# plot_model(model, to_file='model.png', show_shapes=True)

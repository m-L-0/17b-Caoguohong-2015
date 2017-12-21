#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/11 上午11:13
# @Author  : tudoudou
# @File    : test.py
# @Software: PyCharm

import cv2
import numpy as np
import numpy


def reduction(data, data_type='file', out_type='otsu'):
    if data_type not in ['file', 'img'] or out_type not in ['otsu', 'mean']:
        raise ValueError
    if data_type == 'file':
        img = cv2.imread(data, 0)
    elif data_type == 'img':
        img = data
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if out_type == 'otsu':
        # Otsu 滤波
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th2
    elif out_type == 'mean':
        a = 0
        for i in img:
            b = 0
            for j in i:
                if j < img.mean():  # 比对均值
                    img[a][b] = 0
                else:
                    img[a][b] = 255
                b += 1
            a += 1
            del b
        return img


def read_tfrecord_p(typ='train', num=1000):
    import matplotlib.pyplot as mpl
    import tensorflow as tf
    import numpy as np
    import cv2
    if typ not in ['val', 'train', 'test']:
        raise print('tpy 參數錯誤')
    file_queue = []
    if typ == 'train':
        for i in range(8):
            file_queue.append('./data/' + str(i) + '.tfrecords')
    elif typ == 'test':
        file_queue = ['./data/9.tfrecords']
    elif typ == 'val':
        file_queue = ['./data/8.tfrecords']
    file_queue = tf.train.string_input_producer(file_queue)
    reader = tf.TFRecordReader()
    _, example = reader.read(file_queue)
    features = tf.parse_single_example(example,
                                       features={
                                           'lab': tf.FixedLenFeature([], tf.string),
                                           'img': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img'], tf.uint8)
    lab = tf.cast(features['lab'], tf.string)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images = []
        labels = []
        _ = []
        for i in range(num):
            image, label = sess.run([img, lab])
            image = np.resize(image, [40, 50, 3])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _.append(label)
            label = turn_data(int(label), 1)
            # image=reduction(image,data_type='img')
            images.append([image])
            labels.append(label)
        coord.request_stop()
        coord.join(threads)
        labels = turn_list(labels)
        # images = np.array(images)
        return images, labels, _


def my_read_tfrecord(typ='train', num=1000):
    import matplotlib.pyplot as mpl
    import tensorflow as tf
    import numpy as np
    import cv2
    if typ not in ['val', 'train', 'test']:
        raise print('tpy 參數錯誤')
    file_queue = []
    if typ == 'train':
        for i in range(8):
            file_queue.append('./data1/' + str(i) + '.tfrecords')
    elif typ == 'test':
        file_queue = ['./data1/9.tfrecords']
    elif typ == 'val':
        file_queue = ['./data1/8.tfrecords']
    file_queue = tf.train.string_input_producer(file_queue)
    reader = tf.TFRecordReader()
    _, example = reader.read(file_queue)
    features = tf.parse_single_example(example,
                                       features={
                                           'lab': tf.FixedLenFeature([], tf.string),
                                           'img': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img'], tf.uint8)
    lab = tf.cast(features['lab'], tf.string)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        images = []
        labels = []
        _ = []
        for i in range(num):
            image, label = sess.run([img, lab])
            image = np.resize(image, [40, 50, 3])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _.append(label)
            label = turn_data(int(label), 1)
            # image=reduction(image,data_type='img')
            images.append([image])
            labels.append(label)
        coord.request_stop()
        coord.join(threads)
        labels = turn_list(labels)
        # images = np.array(images)
        return images, labels, _


def turn_list(lis):
    import numpy
    result = [[], [], [], []]
    for i in lis:
        for t in range(4):
            result[t].append(i[t])
    for i in range(4):
        result[i] = numpy.array(result[i])
    return result


def turn_data(data, num):
    import numpy
    data = fill(data)
    val = numpy.zeros((4, 11))
    num = 0
    for i in data:
        if i == '@':
            i = 10
        else:
            i = int(i)
        val[num][i] = 1
        num += 1
    return val


def fill(data):
    data = str(data)
    le = len(data)
    for i in range(4 - le):
        data += '@'
    return data


# print(turn_data(12, 0))


from model import create_model, use_model, tr_model
from googLeNet import my_InceptionV3
from my_model import my_model


def train():
    model = create_model()
    a, b, _ = read_tfrecord_p(num=32000)
    model.fit(a, b, epochs=5, batch_size=128)
    model.save('./test.h5')
    tX, ty, _ = read_tfrecord_p(typ='val', num=4000)
    tX = numpy.array(tX)
    use_model('./test.h5', tX, ty, typ='tfrecord')


def test():
    import numpy
    tX, ty, _ = read_tfrecord_p(typ='val', num=4000)
    tX = numpy.array(tX)
    X, y, _ = read_tfrecord_p(num=32000)
    for i in range(2):
        tr_model('./test.h5', X=X, y=y, batch_size=128, epochs=5, eval_X=tX, eval_y=ty)
        use_model('./test.h5', tX, ty, typ='tfrecord')


# train()
# test()

def my_test1():
    model = my_model()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    a, b, _ = read_tfrecord_p(num=32000)
    model.fit(a, b, epochs=5, batch_size=128)
    model.save('./my_model.h5')


def re_my_test1():
    import numpy
    tX, ty, _ = read_tfrecord_p(typ='val', num=4000)
    teX, tey, _ = read_tfrecord_p(typ='test', num=4000)
    teX = numpy.array(teX)
    tX = numpy.array(tX)
    X, y, _ = read_tfrecord_p(num=32000)
    use_model('./test1.h5', X, y, typ='tfrecord')
    for i in range(10):
        tr_model('./test1.h5', X=X, y=y, eval_X=tX, eval_y=ty, batch_size=128, epochs=2)
        use_model('./test1.h5', teX, tey, typ='tfrecord')


# my_test1()
# re_my_test1()



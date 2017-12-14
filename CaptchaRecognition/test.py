#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/11 上午11:13
# @Author  : tudoudou
# @File    : test.py
# @Software: PyCharm


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


def train():
    model = create_model()
    a, b, _ = read_tfrecord_p(num=10000)
    model.fit(a, b, epochs=10, batch_size=128)
    model.save('./test1.h5')


def tr():
    X, y, _ = read_tfrecord_p(num=32000)
    tr_model('./test1.h5', X=X, y=y, batch_size=128, epochs=10)

# train()
def test():
    import numpy
    X, y, _ = read_tfrecord_p(typ='val', num=2)
    X = numpy.array(X)
    print(y)
    use_model('./test1.h5', X, y, typ='tfrecord')



test()
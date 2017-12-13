#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/11 上午11:13
# @Author  : tudoudou
# @File    : test.py
# @Software: PyCharm


import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import csv

label = []
image_data = []
for root, _, filename_list in os.walk('./data/images/'):
    with open('./data/labels/labels.csv') as data:
        data = csv.reader(data)
        da = []
        for i in data:
            da.append([i[0].split('/')[-1], i[1]])
        data = dict(da)
        del da
        for i in filename_list:
            a = Image.open(str(os.path.join(root, i)))
            a = a.resize((50, 40))
            a = np.array(a)
            a = np.resize(a, (6000))
            label.append(data[i])
            image_data.append(a)

li = list(range(len(label)))
random.shuffle(li)

print('開始寫入')

for i in range(40000):
    if i % 4000 == 0:
        name=str(i//4000)
        writer = tf.python_io.TFRecordWriter('./data/' + name + '.tfrecords')
    if (i + 1) % 1000 == 0:
        print('已處理{}數據集{}張'.format(name, i))
    img = image_data[li[i]].tostring()
    lab = label[li[i]].encode()
    example = tf.train.Example(features=tf.train.Features(feature={
        'lab': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lab])),
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
    }))
    writer.write(example.SerializeToString())

    if i % 4000 == 3999:
        print('{}數據集處理完成'.format(name))
        writer.close()



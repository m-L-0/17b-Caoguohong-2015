#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/21 上午9:10
# @Author  : tudoudou
# @File    : predict.py
# @Software: PyCharm





def predict(filename):
    import cv2
    import numpy
    from keras.models import load_model
    model = load_model('./tools/test1.h5')
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (50, 40))
    img = numpy.resize(img, (1, 1, 40, 50))
    temp = model.predict([img])
    print(img)
    arr = []
    result = ''
    for i in temp:
        arr.append(numpy.argmax(i))
    del temp,model
    for i in arr:
        i = str(i)
        if i != '10':
            result += i
    return result



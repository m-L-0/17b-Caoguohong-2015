#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/11 上午11:13
# @Author  : tudoudou
# @File    : main.py
# @Software: PyCharm


import cv2
import numpy as np
import matplotlib.pyplot as plt


def reduction(data, data_type='file', out_type='otsu'):
    if data_type not in ['file','img'] or out_type not in ['otsu','mean']:
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


img = cv2.imread('./data/4288.jpg', 0)
print(reduction(img, 'img', 'mean'))
img=reduction(img, 'img', 'mean')

plt.imshow(img,'gray')
plt.show()
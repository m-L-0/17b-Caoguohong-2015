#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/11 上午11:13
# @Author  : tudoudou
# @File    : test.py
# @Software: PyCharm

import cv2
import matplotlib.pyplot as plt



# opsu's 自適應算法
def turn_two_color(name):
    img = cv2.imread(name)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    a = 0
    for i in grayImage:
        b = 0
        for j in i:
            if j < ret2:  # 比对均值
                grayImage[a][b] = 0
            else:
                grayImage[a][b] = 255
            b += 1
        a += 1
        del b
    plt.imshow(grayImage)
    plt.axis('off')
    plt.show()
    # cv2.imwrite('123.jpg', grayImage)

# 均值算法
def turn_two_color2(name):
    img = cv2.imread(name)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = 0
    for i in grayImage:
        b = 0
        for j in i:
            if j < grayImage.mean():  # 比对均值
                grayImage[a][b] = 0
            else:
                grayImage[a][b] = 255
            b += 1
        a += 1
        del b
    plt.imshow(grayImage)
    plt.axis('off')
    plt.show()
#
# turn_two_color('186.jpg')
# turn_two_color2('186.jpg')


def yanzm():
    import string
    import random
    from captcha.image import ImageCaptcha
    # characters = string.digits+string.ascii_uppercase
    characters = string.digits

    width, height, n_len, n_class = 170, 80, 4, len(characters)

    generator = ImageCaptcha(width=width, height=height)
    random_str = ''.join([random.choice(characters) for j in range(3)])
    img = generator.generate_image(random_str)

    plt.imshow(img)
    plt.title(random_str)
    plt.show()

# yanzm()


def tongji():
    import csv
    with open('./data/labels/labels.csv') as data:
        data=csv.reader(data)
        result={}
        tem=[]
        num=0
        for i in data:
            num+=1
            if len(i[1]) in result:
                result[len(i[1])]+=1
            else:
                result[len(i[1])]=1
        print(result)
        print(num)
        for i in result.keys():
            result[i]=result[i]/num
        print(result)

tongji()

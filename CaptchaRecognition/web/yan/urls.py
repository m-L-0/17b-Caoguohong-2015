#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/14 上午8:12
# @Author  : tudoudou
# @File    : urls.py
# @Software: PyCharm

from django.conf.urls import url
from .views import *

urlpatterns = [
    url(r'^ya',yan,name='yanzhengma')
]

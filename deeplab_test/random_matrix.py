#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : random_matrix.py
# @Author: WangYe
# @Date  : 2019/6/24
# @Software: PyCharm
import tensorflow as tf

train_list = tf.random_normal([400,384,876,16],stddev=1.0,dtype=tf.float32,seed=None,name=None)
label_list = tf.random_normal([400,384,876,1],stddev=1.0,dtype=tf.float32,seed=None,name=None)
# print(v1)
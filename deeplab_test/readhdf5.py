#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : readhdf5.py
# @Author: WangYe
# @Date  : 2019/5/31
# @Software: PyCharm
import h5py
import os
import numpy as np

def readdata(path):


    h5f = h5py.File(path, "r")

    matrix = h5f['data'][:]
    return matrix

def train_data(matrix):
    # print(matrix.shape)
    m1 = np.asarray(matrix).reshape(384, 576)
    # print(m1.shape)
    matrix_list = []

    for i in range(4):
        matrix_list.append(m1)
    matrix = np.asarray(matrix_list)
    m2 = np.asarray(matrix).reshape(384, 576, 4)
    # print(m2.shape)

    list_train = []
    for i in range(5):
        list_temp = m2
        list_train.append(list_temp)
    train = np.asarray(list_train)
    # print(train.shape)
    return train
def label_data(matrix):
    m2 = np.asarray(matrix).reshape(384,576,1)
    list_train = []
    for i in range(5):
        list_temp = m2
        list_train.append(list_temp)
    out = np.asarray(list_train)
    # print(out.shape)
    return out


if __name__ == '__main__':
    path = 'test1.h5'
    matrix = readdata(path)
    train_data(matrix)
    label_data(matrix)



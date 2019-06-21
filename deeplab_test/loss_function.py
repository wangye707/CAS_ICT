#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : loss_function.py
# @Author: WangYe
# @Date  : 2019/6/21
# @Software: PyCharm
import numpy as np
# max()
matrix1 = np.asarray([[0,1,0],[1,1,0]])
matrix2 = np.asarray([[1,1,1],[0,1,1]])

def loss(matrix1,matrix2):
    m1 = find_x(matrix1,1)
    m2 = find_x(matrix2,1)

    insec = find_x(matrix1+matrix2,2)

    loss_value = insec/(m1+m2-insec)
    print("insec",insec)
    return (1-loss_value)

def find_x(matrix, target):
    sum = np.sum(matrix == target)
    return sum

loss_num = loss(matrix1,matrix2)

if __name__ == '__main__':
    matrix1 = np.asarray([[0, 1, 0], [1, 1, 0]])
    matrix2 = np.asarray([[1, 1, 1], [0, 1, 1]])
    loss(matrix1,matrix2)

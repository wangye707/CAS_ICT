#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : readdata.py
# @Author: WangYe
# @Date  : 2019/5/22
# @Software: PyCharm
import os
from PIL import Image
import numpy as np
from  readhdf5 import readdata
path = './H5'

def walk_file(path):
    fatherLists = os.listdir(path)  # 主目录
    train_list = []
    label_list = []
    for name in fatherLists:
        # print(name)
        file_path = path + '/' + name

        train,label = readdata(file_path)
        label = np.asarray(label).reshape(384,576,1)
        # print(label.shape)
        # print(train.shape)
        train_list.append(train)
        '''label为384*576，改为384*576*1'''
        label_list.append(label)
    train_list = np.asarray(train_list)
    label_list = np.asarray(label_list)
    return train_list,label_list




def read(path):
    with open(path) as f:
        data = f.readlines()
    data_out = []
    for x in data:
        temp = x.replace('\n','')
        data_out.append(temp)
    return data_out
def read_image(list_name,path,str,dim):  #str 是图片的后缀例如.jpg   ，   .png

    list_out =[]  #读图片矩阵
    list_out_pad = []  #图片填充矩阵
    for root, dirs, files in os.walk(path):
        # root代表路径,dirs代表目录，files代表文件名

        for _dir in dirs:  # 若是目录，跳过
            pass

        #查找datapath中的文件
        for name in list_name:
            x_name = name + str
            image_file = os.path.join(root,x_name)  # 拼接目录
            # x = Image.open(path).convert("L")  # 将图片转换为矩阵
            # matrix = np.asarray(x, 'f')
            x = Image.open(image_file)    # 打开图片
            matrix = np.asarray(x)  # 转换为矩阵
            # te = matrix.reshape()
            # print(matrix)
            # print(matrix.shape)
            list_out.append(matrix)
        list_out_pad = image_padding(list_out,dim)
    # print(list_out_pad.shape)
    return list_out_pad

def image_padding(data,dim):  #图片填充  data是输入的图片列表，列表里面是每个图片的维度，dim是图片的通道数
                                                            #dim = 1 是灰色的，3是彩色的
    data_out =[]
    length = []
    weight = []
    for x in data:
        weidu = x.shape
        # print(weidu[0])
        # print(weidu[1])
        # print(type(weidu))
        length.append(weidu[0])
        weight.append(weidu[1])
    # print(max(length))#450
    # print(max(weight))#450
    MAX_l = max(length)  #读取长的最大值
    MAX_W = max(weight)  #宽的最大值
    '''
    开始填充矩阵
    '''
    for x in data:
        temp_tuple = x.shape  # 获取当前矩阵维度
        length = temp_tuple[0]  # 读取长
        width = temp_tuple[1]  # 读取宽
        pad_length_up = int((MAX_l - length) / 2)  # 长的向上填充
        pad_length_down = int(MAX_l - length - pad_length_up)
        pad_width_left = int((MAX_W - width) / 2)
        pad_width_right = int(MAX_W - width - pad_width_left)
        matrix_pad = []
        if int(dim) == 3:
            matrix_pad = np.pad(x,
                                pad_width=((pad_length_up, pad_length_down),
                                           (pad_width_left, pad_width_right),
                                           (0, 0)  # 三维处理成一维之后就不用了
                                           ),
                                mode="constant", constant_values=(0, 0))
        if int(dim) == 1:   #即1维灰度图
            matrix_pad = np.pad(x,
                                pad_width=((pad_length_up, pad_length_down),
                                           (pad_width_left, pad_width_right),
                                           # (0, 0)  # 三维处理成一维之后就不用了
                                           ),
                                mode="constant", constant_values=(0, 0))



        # print(matrix_pad.shape)
        np.array(matrix_pad).reshape((MAX_l,MAX_W,dim))
        temp_list = []
        temp_list.append(matrix_pad)
        # np.asarray(matrix_pad).reshape(MAX_l,MAX_W,1)
        q=np.asarray(temp_list).reshape(MAX_l,MAX_W,dim)
        # print(q.shape)
        data_out.append(q)
    out = np.asarray(data_out)
    # out.reshape(MAX_l,MAX_W,1)
    # print(out.shape)
    return out  #输出是已经处理好的矩阵，例如 （1450，500，500，3）1450张500*500的三通道图片


if __name__ == '__main__':
    # datapath = r'train.txt'
    # trainpath = r'D:\code\python\DeepLab_v3\data\datasets\VOCdevkit\VOC2012\JPEGImages'
    # labelpath = r'D:\code\python\DeepLab_v3\data\datasets\VOCdevkit\VOC2012\SegmentationObject'
    # name_list = read(datapath)
    # train_list = read_image(path=trainpath,list_name=name_list,str = '.jpg',dim = 3)#(1464, 500, 500, 3)
    # label_list = read_image(path=labelpath,list_name=name_list,str = '.png',dim = 1 )#(1464, 500, 500, 1)
    path = './H5'
    walk_file(path)
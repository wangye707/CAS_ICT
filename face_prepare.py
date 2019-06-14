#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : face_prepare.py
# @Author: WangYe
# @Date  : 2019/4/17
# @Software: PyCharm

from PIL import Image
import numpy as np
import os
def readData():
    image_dir = r"./face_data/"
    image_data = []
    image_lable = []
    image_length = []
    image_width = []
    for root, dirs, files in os.walk(image_dir):
        # root代表路径,dirs代表目录，files代表文件名

        for _dir in dirs:  # 若是目录，跳过
            pass
        for _file in files:
            image_file = os.path.join(root, _file) #拼接目录

            if "images" in root:

                x = Image.open(image_file).convert("L") #打开图片
                matrix = np.asarray(x,'f')      #转换为矩阵
                #te = matrix.reshape()
                # print(matrix)
                '''
                这里得到的矩阵维度是不同的，比如
                (854, 613, 3)
                (651, 855, 3)
                虽然都是3维，但是长和宽我们要进行填充
                所以先找出长和宽的最大值
                最大维度：1569*1000*3
                image_length.append(temp_tuple[0])
                image_width.append(temp_tuple[1])
                '''

                temp_tuple = matrix.shape#获取当前矩阵维度
                #print(temp_tuple)
                #print(type(matrix))  #<class 'numpy.ndarray'>
                '''
                开始填充矩阵
                '''
                length = temp_tuple[0]  #读取长
                width = temp_tuple[1]   #读取宽
                matrix = matrix.reshape(length,width,1)
                pad_length_up = int((1750 -length) / 2)  #长的向上填充，1559按1750算，整数
                pad_length_down = int(1750 - length - pad_length_up)
                pad_width_left = int((1000 - width) / 2)
                pad_width_right = int(1000 - width - pad_width_left)

                matrix_pad = np.pad(matrix,
                                    pad_width=((pad_length_up, pad_length_down),
                                               (pad_width_left, pad_width_right),
                                               (0,0)  #三维处理成一维之后就不用了
                                               ),
                            mode="constant", constant_values=(0, 0))


                image_data.append(matrix_pad)       #存储矩阵
            elif "label" in root:

                x = Image.open(image_file).convert('L')  # 打开图片
                matrix = np.asarray(x,'f')  # 转换为矩阵
                '''
                所有标签都是(250, 250, 3)
                '''
                matrix = matrix.reshape(250,250,1)
                #temp_tuple = matrix.shape
                #print(temp_tuple)
                # print(type(matrix))  #<class 'numpy.ndarray'>
                matrix1 = np.ravel(matrix)

                image_lable.append(matrix1)  # 存储矩阵

    # print(max(image_length))#1569
    # print(max(image_width))#1000
    data = np.asarray(image_data) #(1028, 1600, 1000, 3)
    label = np.asarray(image_lable)#(1028,250, 250, 3)

    return data,label
#readData()
if __name__ == '__main__':
    readData()

'''
下面的这个代码是因为data和label有些的数据对不上，所以写了个它来处理有data
没label的数据。现在已经我手动处理过了，所以每个data都有属于自己的label

'''

# image_dir1 = r"./face_data/face_label"
# image_dir = r"./face_data/images"
# temp = []
# for root, dirs, files in os.walk(image_dir):
#     # root代表路径,dirs代表目录，files代表文件名
#     temp = []
#     for _dir in dirs:  # 若是目录，跳过
#         pass
#     for _file in files:
#         #image_file = os.path.join(root, _file)  # 拼接目录
#         #print(_file)
#         temp.append(_file)
# # print(temp)
#
# temp1 = []
# for root, dirs, files in os.walk(image_dir1):
#     # root代表路径,dirs代表目录，files代表文件名
#
#     for _dir in dirs:  # 若是目录，跳过
#         pass
#     for _file in files:
#         #image_file = os.path.join(root, _file)  # 拼接目录
#         #print(_file)
#         if _file in temp:
#             #print(_file)
#             pass
#         else:
#             temp1.append(_file)
#             pass
# print(len(temp1))   #11个
# print(temp1)

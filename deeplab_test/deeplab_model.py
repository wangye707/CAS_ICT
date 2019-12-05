#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : cifar10_atrous_conv2d.py
# @Author: WangYe
# @Date  : 2019/3/26
# @Software: PyCharm

# coding:utf-8
# 导入官方cifar10模块
#from tensorflow.image.cifar10 import cifar10
# import cifar10
# import tensorflow as tf
#
# # tf.app.flags.FLAGS是tensorflow的一个内部全局变量存储器
# FLAGS = tf.app.flags.FLAGS
# # cifar10模块中预定义下载路径的变量data_dir为'/tmp/cifar10_eval',预定义如下：
# # tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
# #                           """Path to the CIFAR-10 data directory.""")
# # 为了方便，我们将这个路径改为当前位置
# FLAGS.data_dir = './cifar10_data'
#
# # 如果不存在数据文件则下载，并且解压
# cifar10.maybe_download_and_extract()
#from loss_function import loss1
from readdata import walk_file
import tensorflow as tf
import numpy as np
import time

#cifar10.maybe_download_and_extract()

# datapath = r'train.txt'
# trainpath = r'D:\code\python\DeepLab_v3\data\datasets\VOCdevkit\VOC2012\JPEGImages'
# labelpath = r'D:\code\python\DeepLab_v3\data\datasets\VOCdevkit\VOC2012\SegmentationObject'
# name_list = read(datapath)
# train_list = read_image(path=trainpath, list_name=name_list, str='.jpg', dim=3)  # (1464, 500, 500, 3)
# label_list = read_image(path=labelpath, list_name=name_list, str='.png', dim=1)  # (1464, 500, 500, 1)

# def variable_with_weight_loss(shape, stddev, w1):
#     var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
#     if w1 is not None:
#         weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
#         tf.add_to_collection('losses', weight_loss)
#     return var

path = '/root/hdf5/h5output/'
train_list,label_list = walk_file(path)
#train_list = tf.random_normal([40,384,576,3],stddev=1.0,dtype=tf.float32,seed=None,name=None)
#label_list = tf.random_normal([40,384,576,1],stddev=1.0,dtype=tf.float32,seed=None,name=None)
print(train_list.shape,label_list.shape)

max_steps = 3000
batch_size = 2
train_x1 = 384  #训练集矩阵第一维度
train_x2 = 576  #训练集矩阵第二维度
train_dim = 16  #训练集图片通道数
label_y1 = 384  #标签矩阵第一维度
label_y2 = 576  #标签矩阵第二维度
label_dim = 1   #标签图片通道数
temp_position = [0,100,200,300,400,420]
train_data = []
label_data = []
for i in range(len(temp_position)-1):
    # print(i)
    train_new_list = train_list[temp_position[i]:temp_position[i+1],0:train_x1,0:train_x2,0:train_dim]
    print(train_new_list.shape)
    label_new_list = label_list[temp_position[i]:temp_position[i+1],0:label_y1,0:label_y2,0:label_dim]
    train_data.append(train_new_list)
    label_data.append(label_new_list)


image_holder = tf.placeholder(tf.float32, [batch_size, train_x1, train_x2, train_dim])
label_holder = tf.placeholder(tf.float32, [batch_size, label_y1, label_y2, label_dim])

def tf_conv(inputs,filters,kernel_size,
            strides=1,stddev=0.01,padding = 'SAME'):  #卷积层
    out = tf.layers.conv2d(
        inputs=inputs,  #输入tensor
        filters=filters, #输出维度
        kernel_size=kernel_size,#卷积核大小,例如 [3,3]是3*3的大小
        strides=strides,        #步长
        padding=padding,        #是否填充
        activation=tf.nn.relu,
        # kernel_initializer=tf.truncated_normal_initializer(stddev=stddev)#卷积核的初始化，一般不管
    )
    return out

def tf_pooling(inputs,pool_size,strides):   #池化层
    out =tf.layers.max_pooling2d(
        inputs=inputs,   #输入tensor
        pool_size=pool_size, #池化大小
        strides=strides    #步长
    )
    return out

def tf_atrous_conv(inputs, filters,kernel_size, rate, padding='SAME'):#空洞卷积层
    # out = tf.nn.atrous_conv2d(
    #     value=inputs,  # 输入tensor
    #     filters=filters_kernel,  # 输出维度+卷积核  [卷积核的高度，卷积核的宽度，输入通道数，输入通道数]
    #     rate=rate,  #空洞卷积步长,[1,1]指上下各填充1
    #     padding=padding,  # 是否填充
    # )

    out = tf.layers.conv2d(inputs=inputs, #输入tensor
                           filters=filters, #输出维度
                           kernel_size=kernel_size,#卷积核大小
                            padding=padding,
                           dilation_rate=(rate, rate)#卷积孔大小
                           )

    return out

'''
残差网络是非连续层数之间的一个参数传递，例如第5层和第10层，那第10层就是 9->10->10+5
就像SDU一样，绳索速降，从10楼直接顺着个绳就到1楼了，跳过中间楼层了
所以只有best of the best 才能加入SDU
'''
def tf_resnet(net1,net2):  #残差网络

    out = tf.add(net1, net2)
    return out


def tf_trans_conv(inputs,filters,kernel_size,strides,padding='valid'):
    out = tf.layers.conv2d_transpose(
        inputs=inputs,     #输入的张量
        filters=filters,   #输出维度
        kernel_size=kernel_size,  #卷积核大小
        strides=strides,   #放大倍数
        padding=padding
    )
    return out

def atrous_spatial_pyramid_pooling(inputs, filters=256, regularizer=None):  #ASPP层
    '''
    Atrous Spatial Pyramid Pooling (ASPP) Block
    '''
    pool_height = tf.shape(inputs)[1]
    pool_width = tf.shape(inputs)[2]

    resize_height = pool_height
    resize_width = pool_width

    # Atrous Spatial Pyramid Pooling
    # Atrous 1x1
    aspp1x1 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(1, 1),
                               padding='same', kernel_regularizer=regularizer,
                               name='aspp1x1')
    # Atrous 3x3, rate = 6
    aspp3x3_1 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3),
                                 padding='same', dilation_rate=(12, 12), kernel_regularizer=regularizer,
                                 name='aspp3x3_1')
    # Atrous 3x3, rate = 12
    aspp3x3_2 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3),
                                 padding='same', dilation_rate=(24, 24), kernel_regularizer=regularizer,
                                 name='aspp3x3_2')
    # Atrous 3x3, rate = 18
    aspp3x3_3 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3),
                                 padding='same', dilation_rate=(36, 36), kernel_regularizer=regularizer,
                                 name='aspp3x3_3')
    # Image Level Pooling
    image_feature = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    image_feature = tf.layers.conv2d(inputs=image_feature, filters=filters, kernel_size=(1, 1),
                                     padding='same')
    image_feature = tf.image.resize_bilinear(images=image_feature,
                                             size=[resize_height, resize_width],
                                             align_corners=True, name='image_pool_feature')
    # Merge Poolings
    outputs = tf.concat(values=[aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, image_feature],
                        axis=3, name='aspp_pools')
    outputs = tf.layers.conv2d(inputs=outputs, filters=filters, kernel_size=(1, 1),
                               padding='same', kernel_regularizer=regularizer, name='aspp_outputs')

    return outputs


def model():

    #第一堆
    resnet2 = tf_conv(inputs=image_holder,filters=64,strides=1,kernel_size = [3,3])
    conv = tf_conv(inputs=image_holder,filters=64,strides=2,kernel_size = [8,8],padding='SAME')
    pool = tf_pooling(inputs=conv,pool_size=[2,2],strides=2)
    print('1',pool)
    #第二堆
    for i in range(3):
        conv = tf_conv(inputs=pool,filters=64,kernel_size=[1,1])
        conv = tf_conv(inputs=conv,filters=64,kernel_size=[3,3])
        conv = tf_conv(inputs=conv,filters=256,kernel_size=[3,3])
    resnet1 = conv
    print('2',conv)
    #第三堆
    for i in range(4):
        conv = tf_conv(inputs=conv,filters=128,kernel_size=[1,1])
        conv = tf_conv(inputs=conv, filters=128, kernel_size=[3,3])
        conv = tf_conv(inputs=conv, filters=512, kernel_size=[1, 1])
    print("3",conv)
    #第四堆
    for i in range(6):
        conv = tf_conv(inputs=conv, filters=256, kernel_size=[1, 1])
        atr_conv = tf_atrous_conv(inputs=conv,filters=256,kernel_size=[3,3],rate=2)
        conv = tf_conv(inputs=atr_conv,filters=1024,kernel_size=[1,1])
    print("4",conv)
    #第五维度
    for i in range(3):
        conv = tf_conv(inputs=conv, filters=512, kernel_size=[1, 1])
        atr_conv = tf_atrous_conv(inputs=conv, filters=512,kernel_size=[3,3], rate=4)
        conv = tf_conv(inputs=atr_conv, filters=1024, kernel_size=[1, 1])
    print("5",conv)
    #第五维度，ASPP层
    ASPP = atrous_spatial_pyramid_pooling(conv,filters=256)
    print("5,ASPP",ASPP)

    #第六维度
    '''
    论文中图片此处维度有问题，这里更改了步长，将维度变相等，否则维度对不上
    '''
    conv = tf_conv(inputs=ASPP,filters=256,kernel_size=[1,1],strides=2)
    trans_conv = tf_trans_conv(inputs=conv,filters=256,kernel_size=[2,2],strides=2)
    # print('121',trans_conv)
    conv = tf_conv(inputs=resnet1,filters=256,kernel_size=[3,3])
    resnet_conv = tf_resnet(conv,trans_conv)
    print('6',resnet_conv)

    #第七维度
    conv = tf_conv(inputs=resnet_conv,filters=256,kernel_size=[3,3])
    conv = tf_conv(inputs=conv, filters=256, kernel_size=[3, 3])
    trans_conv = tf_trans_conv(inputs=conv,filters=256,kernel_size=[2,2],strides=2)
    trans_conv = tf_trans_conv(inputs=trans_conv, filters=256, kernel_size=[2, 2], strides=2)
            #加入残差网络的预备卷积
    conv = tf_conv(inputs=resnet2,filters=256,kernel_size=[3,3])
    resnet_conv = tf_resnet(conv,trans_conv)
    print('7',resnet_conv)
    #第八维度
    conv = tf_conv(inputs=resnet_conv,filters=256,kernel_size=[3,3])
    conv = tf_conv(inputs=conv,filters=128,kernel_size=[3,3])
    conv = tf_conv(inputs=conv,filters=1,kernel_size=[1,1])
    print('8',conv)
    #out = conv
    out = tf.sigmoid(conv)  #全部归一化处理

    return out


logits=model()


def get_Batch(data, label, batch_size):
    with tf.device('/cpu:0'):
        X_batch = []
        Y_batch = []
        for i in range(len(data)):
            #print(data.shape, label.shape)
            data_in = data[i]
            label_in = label[i]
            input_queue = tf.train.slice_input_producer([data_in, label_in], num_epochs=None, shuffle=True)
            x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=128, allow_smaller_final_batch=False)
            print('x_batch',x_batch.shape)
            print('y_batch',y_batch.shape)
            X_batch.append(x_batch)
            Y_batch.append(y_batch)
        return X_batch, Y_batch

def loss_wy(logits,labels):

    labels = tf.cast(labels, tf.float32)
    cross_entropy = tf.subtract(logits,labels)
    return tf.reduce_sum(tf.abs(cross_entropy))
    # return tf.reduce_sum(tf.abs(tf.subtract(logits,labels)))
    # a = tf.bincount()
    # return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # return tf.reduce_sum(tf.subtract(logits,labels))
# def wy2_acc(logits,labels):
#     l
def loss_initializer(logits,labels):

    labels_linear = tf.reshape(labels, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(labels_linear,ignore_label))
    # The locations represented by indices in indices take value on_value, while all other locations take value off_value.
    # For example, ignore label 255 in VOC2012 dataset will be set to zero vector in onehot encoding (looks like the not ignore mask is not required)
    onehot_labels = tf.one_hot(indices=labels_linear, depth=self.num_classes, on_value=1.0, off_value=0.0)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=tf.reshape(self.outputs, shape=[-1, self.num_classes]), weights=not_ignore_mask)

    return loss


# def focal_loss(predictions,labels,alpha,gamma):
#     zeros = tf.zeros_like(predictions,dtype=predictions.dtype)
#     pos_corr = tf.where(labels > zeros,labels - predictions , zeros)
#     neg_corr = tf.where(labels >zeros, zeros, predictions)
#     fl_loss = - alpha * (pos_corr**gamma)*tf.log(predictions) -\
#               (1-alpha)*(neg_corr**gamma)*tf.log(1.0 - predictions)
#     return tf.reduce_sum(fl_loss)


def focal_loss(y_true, y_pred):

    gamma = 0.89
    alpha = 0.11
    y_pred1  = tf.abs(y_pred)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred1, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred1, tf.zeros_like(y_pred))

    # pt_1 = tf.clip(pt_1, 1e-3, .999)
    # pt_0 = K.clip(pt_0, 1e-3, .999)
    print(pt_0)
    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) - tf.reduce_sum(
        (1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))
    
#def loss(logits, labels):
#    labels = tf.cast(labels, tf.float32)
#    cross_entropy = logits - labels
#    # tf.nn.sparse_softmax_cross_entropy_with_logits\
#    # (logits=logits, labels=labels,
    #   # name='cross_entropy_per_example'
    #  )
#    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#    tf.add_to_collection('losses', cross_entropy_mean)
#    return tf.add_n(tf.get_collection('losses'), name='total_loss')
def acc_wy(loss_wy,logits,labels):
    labels = tf.cast(labels, tf.float32)
    logit = tf.reduce_sum(logits)
    label = tf.reduce_sum(labels)
    both = tf.div(tf.subtract(tf.add(logit,label),loss_wy),2)  #并集
    return tf.div(both,label)

def position(logits,labels):
    return tf.reduce_sum(logits),tf.reduce_sum(labels)
    # return tf.metrics.mean_iou(labels,logits,num_classes=1)[1]
loss = focal_loss(
                label_holder,logits
                # logits,label_holder,
               # ,alpha=0.25,gamma=2
               )

train_op = tf.train.AdamOptimizer(0.00001).minimize(loss)


accuary = acc_wy(loss,logits, label_holder)
pre,lab = position(logits, label_holder)
#top_k_op = tf.nn.in_top_k(logits, tf.cast(label_holder, tf.int64), 1)
image_batch,label_batch = get_Batch(train_data,label_data,batch_size)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
tf.train.start_queue_runners(coord = coord)
for step in range(max_steps):
    loss_avg = []
    examples_per_sec_avg = []
    sec_per_batch_avg = []
    lab_list =[]
    pre_list = []
    start_time = time.time()
    for i in range(len(temp_position)-1):
    # image_batch, label_batch = sess.run([images_train, labels_train])
        date,label= sess.run([image_batch[i],label_batch[i]])
        acc,_, loss_value,pre1,lab1 = sess.run([accuary,train_op, loss,pre,lab], feed_dict={image_holder: date, label_holder: label})
        duration = time.time() - start_time
        if step % 1 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            # print("acc",acc)
            # print("step:",step,'loss:',loss_value)
            for i in range(5):   #按切分去显示，就是因OOM的问题的处理方式
            # print(loss_value)
                loss_avg.append(loss_value)
                examples_per_sec_avg.append(examples_per_sec)
                sec_per_batch_avg.append(sec_per_batch)
                pre_list.append(pre1)
                lab_list.append(lab1)
                # lab.append()
                if len(loss_avg) == len(temp_position)-1:   #将step = 10的loss求平均，因为之前处理过batch
                    loss_out = (sum(loss_avg)/len(temp_position))
                    examples_per_sec_out = (sum(examples_per_sec_avg)/len(temp_position))
                    sec_per_batch_out = (sum(sec_per_batch_avg)/len(temp_position))
                    pre_out = (sum(pre_list)/len(temp_position))#预测的目标和
                    lab_out = (sum(lab_list)/len(temp_position))#标签中的目标和
                    print('step',step,'acc',acc,'loss',loss_out,'pre_out',pre_out,'lab_out',lab_out)
                    # format_str = ('step %d,acc=%d,loss=%.2f,pre %d,lab %d,(%.1f examples/sec;%.3f sec/batch)')
                    # print(format_str % (step,acc,loss_out,pre_out,lab_out,examples_per_sec_out,sec_per_batch_out))
# builder = tf.saved_model.builder.SavedModelBuilder("./fuck")

#
# num_examples = 10000
# import math
#
# num_iter = int(math.ceil(num_examples / batch_size))
# true_count = 0
# total_sample_count = num_iter * batch_size
# step = 0
# # with tf.Session() as sess:
# while step < num_iter:
#     image_batch, label_batch = sess.run([images_test, labels_test])
#     predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
#                                                   label_holder: label_batch})
#     true_count += np.sum(predictions)
#     step += 1
#     if step % 10 == 0:
#         print(true_count)
#
# precision = float(true_count) / total_sample_count
# print('precision @ 1 =%.3f' % precision)

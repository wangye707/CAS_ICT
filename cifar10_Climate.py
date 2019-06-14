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
import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

#cifar10.maybe_download_and_extract()
max_steps = 3000
batch_size = 128
data_dir = r'./cifar10_data/cifar-10-batches-bin'


def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [batch_size])
#第一层


#输入为  24*24*3
weight1 = variable_with_weight_loss(shape=[7,7, 3, 64], stddev=0.05, w1=0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1,2, 2, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
conv1 = tf.nn.relu(kernel1 + bias1)    #12*12*64
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)    #6*6*64



#第二层  输入6*6*64
weight2 = variable_with_weight_loss(shape=[1, 1, 64, 64], stddev=0.05, w1=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(kernel2 + bias2)    #6*6*64

#第三层  输入6*6*128
weight3 = variable_with_weight_loss(shape=[3,3 , 64, 64], stddev=0.05, w1=0.0)
kernel3 = tf.nn.conv2d(conv2, weight3, [1, 1, 1, 1], padding='SAME')
bias3 = tf.Variable(tf.constant(0.1, shape=[64]))
conv3= tf.nn.relu(kernel3 + bias3)    #6*6*64

#第四层
weight4 = variable_with_weight_loss(shape=[1,1 , 64, 256], stddev=0.05, w1=0.0)
kernel4 = tf.nn.conv2d(conv3, weight4, [1, 1, 1, 1], padding='SAME')
bias4 = tf.Variable(tf.constant(0.1, shape=[256]))
conv4= tf.nn.relu(kernel4+ bias4)    #6*6*256


#第五层  第二次第二层
weight5 = variable_with_weight_loss(shape=[1, 1, 256, 64], stddev=0.05, w1=0.0)
kernel5 = tf.nn.conv2d(conv4, weight5, [1, 1, 1, 1], padding='SAME')
bias5 = tf.Variable(tf.constant(0.1, shape=[64]))
conv5 = tf.nn.relu(kernel5 + bias5)    #6*6*64


#第六层   第二次第三层

weight6 = variable_with_weight_loss(shape=[3, 3, 64, 64], stddev=0.05, w1=0.0)
kernel6 = tf.nn.conv2d(conv5, weight6, [1, 1, 1, 1], padding='SAME')
bias6 = tf.Variable(tf.constant(0.1, shape=[64]))
conv6 = tf.nn.relu(kernel6 + bias6)    #6*6*64


#第七层  第二次第四层
weight7 = variable_with_weight_loss(shape=[1,1 , 64, 256], stddev=0.05, w1=0.0)
kernel7 = tf.nn.conv2d(conv6, weight7, [1, 1, 1, 1], padding='SAME')
bias7 = tf.Variable(tf.constant(0.1, shape=[256]))
conv7= tf.nn.relu(kernel7+ bias7)    #6*6*256


#第八层  第三次第二层
weight8 = variable_with_weight_loss(shape=[1, 1, 256, 64], stddev=0.05, w1=0.0)
kernel8 = tf.nn.conv2d(conv7, weight8, [1, 1, 1, 1], padding='SAME')
bias8 = tf.Variable(tf.constant(0.1, shape=[64]))
conv8 = tf.nn.relu(kernel8 + bias8)    #6*6*64


#第八层  第三次第三层

weight9 = variable_with_weight_loss(shape=[3, 3, 64, 64], stddev=0.05, w1=0.0)
kernel9 = tf.nn.conv2d(conv8, weight9, [1, 1, 1, 1], padding='SAME')
bias9 = tf.Variable(tf.constant(0.1, shape=[64]))
conv9 = tf.nn.relu(kernel9 + bias9)    #6*6*64


#第八层  第三次第四层
weight10 = variable_with_weight_loss(shape=[1,1 , 64, 256], stddev=0.05, w1=0.0)
kernel10 = tf.nn.conv2d(conv9, weight10, [1, 1, 1, 1], padding='SAME')
bias10 = tf.Variable(tf.constant(0.1, shape=[256]))
conv10= tf.nn.relu(kernel10+ bias10)    #6*6*256


#常规卷积
def con_wy(input,shape_weight,shape_bias,strides):
    weight = variable_with_weight_loss(shape_weight, stddev=0.05, w1=0.0)
    kernel = tf.nn.conv2d(input, weight, strides, padding='SAME')
    bias = tf.Variable(tf.constant(0.1, shape=[shape_bias]))
    conv= tf.nn.relu(kernel + bias)
    return conv
#空洞卷积
def atrous_conv_wy(input,shape_weight,shape_bias,rate):
    weight = variable_with_weight_loss(shape_weight, stddev=0.05, w1=0.0)
    kernel = tf.nn.atrous_conv2d(input, weight, rate=rate, padding='SAME')  # 空洞卷积会把上面的卷积核编程  10*10的大小
    bias = tf.Variable(tf.constant(0.1, shape=[shape_bias]))  # 输出维度128
    conv = tf.nn.relu(kernel + bias)  # 输出为b*7*7*128
    return conv

#第九到十七层
conv11 = con_wy(input=conv10,shape_weight=[1,1,256,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv12 = con_wy(input=conv11,shape_weight=[3,3,128,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv13 = con_wy(input=conv12,shape_weight=[1,1,128,512],shape_bias=512,strides=[1,1,1,1])  #6*6*512

conv14 = con_wy(input=conv13,shape_weight=[1,1,512,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv15 = con_wy(input=conv14,shape_weight=[3,3,128,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv16 = con_wy(input=conv15,shape_weight=[1,1,128,512],shape_bias=512,strides=[1,1,1,1])  #6*6*512

conv17 = con_wy(input=conv16,shape_weight=[1,1,512,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv18 = con_wy(input=conv17,shape_weight=[3,3,128,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv19 = con_wy(input=conv18,shape_weight=[1,1,128,512],shape_bias=512,strides=[1,1,1,1])  #6*6*512

conv20 = con_wy(input=conv19,shape_weight=[1,1,512,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv21 = con_wy(input=conv20,shape_weight=[3,3,128,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv22 = con_wy(input=conv21,shape_weight=[1,1,128,512],shape_bias=512,strides=[1,1,1,1])  #6*6*512

conv23 = con_wy(input=conv22,shape_weight=[1,1,512,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv24 = con_wy(input=conv23,shape_weight=[3,3,128,128],shape_bias=128,strides=[1,1,1,1])  #6*6*128
conv25 = con_wy(input=conv24,shape_weight=[1,1,128,512],shape_bias=512,strides=[1,1,1,1])  #6*6*512

#第十八到
# def atrous_conv_wy(input,shape_weight,shape_bias,rate):
#     weight = variable_with_weight_loss(shape_weight, stddev=0.05, w1=0.0)
#     kernel = tf.nn.atrous_conv2d(input, weight, rate=rate, padding='SAME')  # 空洞卷积会把上面的卷积核编程  10*10的大小
#     bias = tf.Variable(tf.constant(0.1, shape=[shape_bias]))  # 输出维度128
#     conv = tf.nn.relu(kernel + bias)  # 输出为b*7*7*128
#     return conv
#第十八层到三十七层
conv26 = con_wy(input=conv25,shape_weight=[1,1,512,256],shape_bias=256,strides=[1,1,1,1])#6*6*256
conv27 = atrous_conv_wy(input=conv26,shape_weight=[3,3,256,256],shape_bias=256,rate=2)  #6*6*256
conv28 = con_wy(input=conv27,shape_weight=[1,1,256,1024],shape_bias=1024,strides=[1,1,1,1])#6*6*1024

conv29 = con_wy(input=conv28,shape_weight=[1,1,1024,256],shape_bias=256,strides=[1,1,1,1])#6*6*256
conv30 = atrous_conv_wy(input=conv29,shape_weight=[3,3,256,256],shape_bias=256,rate=2)  #6*6*256
conv31 = con_wy(input=conv30,shape_weight=[1,1,256,1024],shape_bias=1024,strides=[1,1,1,1])#6*6*1024

conv32 = con_wy(input=conv31,shape_weight=[1,1,1024,256],shape_bias=256,strides=[1,1,1,1])#6*6*256
conv33 = atrous_conv_wy(input=conv32,shape_weight=[3,3,256,256],shape_bias=256,rate=2)  #6*6*256
conv34 = con_wy(input=conv33,shape_weight=[1,1,256,1024],shape_bias=1024,strides=[1,1,1,1])#6*6*1024

conv35 = con_wy(input=conv34,shape_weight=[1,1,1024,256],shape_bias=256,strides=[1,1,1,1])#6*6*256
conv36 = atrous_conv_wy(input=conv35,shape_weight=[3,3,256,256],shape_bias=256,rate=2)  #6*6*256
conv37 = con_wy(input=conv36,shape_weight=[1,1,256,1024],shape_bias=1024,strides=[1,1,1,1])#6*6*1024


conv38 = con_wy(input=conv37,shape_weight=[1,1,1024,256],shape_bias=256,strides=[1,1,1,1])#6*6*256
conv39 = atrous_conv_wy(input=conv38,shape_weight=[3,3,256,256],shape_bias=256,rate=2)  #6*6*256
conv40 = con_wy(input=conv39,shape_weight=[1,1,256,1024],shape_bias=1024,strides=[1,1,1,1])#6*6*1024

conv41 = con_wy(input=conv40,shape_weight=[1,1,1024,256],shape_bias=256,strides=[1,1,1,1])#6*6*256
conv42 = atrous_conv_wy(input=conv41,shape_weight=[3,3,256,256],shape_bias=256,rate=2)  #6*6*256
conv43 = con_wy(input=conv42,shape_weight=[1,1,256,1024],shape_bias=1024,strides=[1,1,1,1])#6*6*1024

conv44 = con_wy(input=conv43,shape_weight=[1,1,1024,256],shape_bias=256,strides=[1,1,1,1])#6*6*256
conv45 = atrous_conv_wy(input=conv44,shape_weight=[3,3,256,256],shape_bias=256,rate=2)  #6*6*256
conv46 = con_wy(input=conv45,shape_weight=[1,1,256,1024],shape_bias=1024,strides=[1,1,1,1])#6*6*1024


#第三十八到四十八层

conv47 = con_wy(input=conv46,shape_weight=[1,1,1024,512],shape_bias=512,strides=[1,1,1,1]) #6*6*512
conv48 = atrous_conv_wy(input=conv47,shape_weight=[3,3,512,512],shape_bias=512,rate=4)   #6*6*512
conv49 = con_wy(input=conv48,shape_weight=[1,1,512,2048],shape_bias=2048,strides=[1,1,1,1])#6*6*2048

conv50 = con_wy(input=conv49,shape_weight=[1,1,2048,512],shape_bias=512,strides=[1,1,1,1]) #6*6*512
conv51 = atrous_conv_wy(input=conv50,shape_weight=[3,3,512,512],shape_bias=512,rate=4)   #6*6*512
conv52 = con_wy(input=conv51,shape_weight=[1,1,512,2048],shape_bias=2048,strides=[1,1,1,1])#6*6*2048

conv53 = con_wy(input=conv52,shape_weight=[1,1,2048,512],shape_bias=512,strides=[1,1,1,1]) #6*6*512
conv54 = atrous_conv_wy(input=conv53,shape_weight=[3,3,512,512],shape_bias=512,rate=4)   #6*6*512
conv55 = con_wy(input=conv54,shape_weight=[1,1,512,2048],shape_bias=2048,strides=[1,1,1,1])#6*6*2048



#全链接
reshape = tf.reshape(conv55, [batch_size, -1])  #分开成为
dim = reshape.get_shape()[1].value       #
weight3 = variable_with_weight_loss([dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
#全连接层
weight4 = variable_with_weight_loss([384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
#输出层
weight5 = variable_with_weight_loss([192, 10], stddev=1 / 192, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.matmul(local4, weight5) + bias5


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


loss = loss(logits, label_holder)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, tf.cast(label_holder, tf.int64), 1)

# num_examples = 10000
# import math
# num_iter = int(math.ceil(num_examples/ batch_size))
# true_count = 0
# total_sample_count = num_iter * batch_size
# step = 0
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     tf.train.start_queue_runners()
#     for step in range(max_steps):
#         start_time = time.time()
#         image_batch,label_batch = sess.run([images_train,labels_train])
#         _,loss_value = sess.run([train_op,loss],feed_dict={image_holder: image_batch,label_holder:label_batch})
#         duration = time.time()-start_time
#         if step % 10 == 0:
#             examples_per_sec = batch_size /duration
#             sec_per_batch = float(duration)

#             format_str = ('step %d,loss=%.2f (%.1f examples/sec;%.3f sec/batch)')
#             print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))

#     while step< num_iter:
#         image_batch,label_batch = sess.run([images_test,labels_test])
#         predictions = sess.run([top_k_op],feed_dict = {image_holder:image_batch,
#                                                        label_holder:label_batch})
#         true_count += np.sum(predictions)
#         step+=1
#         if step % 10 ==0:
#             print true_count
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d,loss=%.2f (%.1f examples/sec;%.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 10000
import math

num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
# with tf.Session() as sess:
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                  label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1
    if step % 10 == 0:
        print(true_count)

precision = float(true_count) / total_sample_count
print('precision @ 1 =%.3f' % precision)

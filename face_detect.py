#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : face_detect.py
# @Author: WangYe
# @Date  : 2019/4/17
# @Software: PyCharm
import tensorflow as tf
import numpy as np
from .face_prepare import readData
from sklearn.model_selection import train_test_split
import time

max_steps = 3000
batch_size = 1
data = np.array(readData()[0])#(1028, 1750, 1000, 3)
label = [[10,10,5],[10,10,5],[10,10,5],[10,10,5],[10,10,5],[10,10,5],[10,10,5],[10,10,5],[10,10,5]]
# label = np.array(readData()[1])#(1028,250, 250, 3)
label = np.array(label)
# print(data[8].shape)
# print(data.shape)
#切分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.3,random_state=0)
print('x_train',len(x_train))
# print('x_test',x_test.shape())
# print('y_train',y_train.shape())
# print('y_test',y_test.shape())

def weight_variable(shape):
#这里是构建初始变量
  initial = tf.truncated_normal(shape, mean=0,stddev=0.1)
#创建变量
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#常规卷积函数
def con_wy(input,shape_weight,shape_bias,strides):
    weight = weight_variable(shape_weight)
    kernel = tf.nn.conv2d(input, weight, strides, padding='SAME')
    bias = tf.Variable(tf.constant(0.1, shape=[shape_bias]))
    conv= tf.nn.relu(kernel + bias)
    return conv
#空洞卷积函数
def atrous_conv_wy(input,shape_weight,shape_bias,rate):
    weight = weight_variable(shape_weight)
    kernel = tf.nn.atrous_conv2d(input, weight, rate=rate, padding='SAME')  # 空洞卷积会把上面的卷积核编程  10*10的大小
    bias = tf.Variable(tf.constant(0.1, shape=[shape_bias]))  # 输出维度128
    conv = tf.nn.relu(kernel + bias)  # 输出为b*7*7*128
    return conv

image_holder = tf.placeholder(tf.float32, [batch_size, 1750, 1000,1])
label_holder = tf.placeholder(tf.float32, [batch_size,None,None,None])



#输入为  1600*1000*3
weight1 = weight_variable(shape=[5,5, 1, 1])
kernel1 = tf.nn.conv2d(image_holder, weight1, [1,7, 4, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[3]))
# conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
conv1 = tf.nn.relu(kernel1 + bias1)    #250*250*3
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)    #400*250*64

# print(label_holder)
# def loss(logits, labels):
#     labels1 = tf.cast(labels, tf.float64)
#     logits1 = tf.cast(logits, tf.float64)
#     logits2 = tf.nn.softmax(logits1)
#     labels2 = tf.nn.softmax(labels1)
#     print("predict::::::::::::", local3)
#     print("label::::::::::::::", label_holder)
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=labels2,
#                                                                    name='cross_entropy_per_example')
#     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#     tf.add_to_collection('losses', cross_entropy_mean)
#     return tf.add_n(tf.get_collection('losses'), name='total_loss')
#两张图片的欧氏距离
# def image_dist():


# print("norm1",norm1)

norm0 = tf.reshape(norm1,[batch_size,-1])
dim = norm0.get_shape()[1].value
weight3 = weight_variable([dim, 250*250])
# dmi1 = np.ravel(dim)
bias3 = tf.Variable(tf.constant(0.1, shape=[250*250]))
# local3 = tf.matmul(norm0,weight3)
local3 = tf.matmul(norm0, weight3)+bias3
print("local3",local3)
print("label_holder",label_holder)
#
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=local3, labels=label_holder,
                                                                   name='cross_entropy_per_example')
# print("local3",local3)#
# dis = tf.square(local3-label_holder)
# # print("dis",dis)#
# dis1 =tf.reduce_sum(dis)
# # print("dis1",dis1)
# loss = tf.sqrt(tf.sqrt(tf.reduce_sum(dis1)))

opt = tf.train.AdamOptimizer(1e-4).compute_gradients(cross_entropy)
#
# print("euclidean",loss)


# dis = tf.norm(local3,label_holder)
# print(dis)
# dis = tf.sqrt(tf.reduce_sum(tf.square(label_holder-local3), 2))
# print(dis)


# with tf.Session() as sess:
#     # dis = tf.sqrt(tf.reduce_sum(tf.square(label_holder-local3), 2))
#     image_batch, label_batch = sess.run([x_train, y_train])
#     out1 = sess.run([local3-label_holder],feed_dict={image_holder: image_batch, label_holder: label_batch})
#     out2 = sess.run(tf.reduce_sum(tf.square(out1), 2))
#     out3 = sess.run(tf.sqrt(out2))
#     # print("out1",out1)
#     # print("out2",out2)
#     print("out3",out3)


# loss = loss(local3,label_holder)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_holder * tf.log(local3), reduction_indices=[1])) # 损失函数，交叉熵
# print("okkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
# train_op = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
# top_k_op = tf.nn.in_top_k(norm1, tf.cast(label_holder, tf.int64), 1)

# def get_batch_data(x,y):
#     images,label  = x,y
#     input_queue = tf.train.slice_input_producer([images, label], shuffle=False,num_epochs=2)
#     image_batch, label_batch = tf.train.batch(input_queue, batch_size=1,
#                                               num_threads=1, capacity=64,
#                                         allow_smaller_final_batch=False)
#     return image_batch,label_batch
#
# def next_batch(train_data, train_target, batch_size=4):
#     #打乱数据集
#     #index = [ i for i in range(0,len(train_target)) ]
#     # np.random.shuffle(index)
#     #建立batch_data与batch_target的空列表
#     batch_data = []
#     batch_target = []
#     #向空列表加入训练集及标签
#     # print(train_data.shape)
#     for i in range(0,batch_size):
#         # print(train_data[i].shape)
#         batch_data.append(train_data[i])
#         batch_target.append(train_target[i])
#     return batch_data, batch_target #返回







sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
for step in range(10000):
    start_time = time.time()
    # image_batch, label_batch = next_batch(data,label)
    # print("image_batch",image_batch)
    # print("label_batch",label_batch)
    for k in range(4):
        print('++++++++++++++')
        # print(x_train.shape())
        # print(y_train.shape())
        opt,test_out = sess.run([loss,
                             opt],
                            feed_dict={image_holder: [data[k]],
                                          label_holder: [label[k]]})
        print(test_out,opt)
    # _, loss_value = sess.run([train_op, cross_entropy], feed_dict={image_holder: image_batch, label_holder: label_batch})
    # duration = time.time() - start_time
    # if step % 10 == 0:
    #     examples_per_sec = batch_size / duration
    #     sec_per_batch = float(duration)
    #
    #     format_str = ('step %d,loss=%.2f (%.1f examples/sec;%.3f sec/batch)')
    #     print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
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
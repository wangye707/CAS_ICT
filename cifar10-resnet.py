#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : cifar10-resnet.py
# @Author: WangYe
# @Date  : 2019/4/16
# @Software: PyCharm

import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

#cifar10.maybe_download_and_extract()
max_steps = 3000
batch_size = 128
data_dir = r'./cifar10_data/cifar-10-batches-bin'

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
#在这里定义残差网络的id_block块，此时输入和输出维度相同
def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block):
        """
        Implementation of the identity block as defined in Figure 3

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            #first
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            b_conv1 = bias_variable([f1])
            X = tf.nn.relu(X+ b_conv1)

            #second
            W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            b_conv2 = bias_variable([f2])
            X = tf.nn.relu(X+ b_conv2)

            #third

            W_conv3 = weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            b_conv3 = bias_variable([f3])
            X = tf.nn.relu(X+ b_conv3)
            #final step
            add = tf.add(X, X_shortcut)
            #b_conv_fin = bias_variable([f3])
            #add_result = tf.nn.relu(add+b_conv_fin)

        #return add_result
        return add

#这里定义conv_block模块，由于该模块定义时输入和输出尺度不同，故需要进行卷积操作来改变尺度，从而得以相加
def convolutional_block( X_input, kernel_size, in_filter,
                            out_filters, stage, block, stride=2):

        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            #first
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='SAME')
            b_conv1 = bias_variable([f1])
            X = tf.nn.relu(X + b_conv1)

            #second
            W_conv2 =weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            b_conv2 = bias_variable([f2])
            X = tf.nn.relu(X+b_conv2)

            #third
            W_conv3 = weight_variable([1,1, f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='SAME')
            b_conv3 = bias_variable([f3])
            X = tf.nn.relu(X+b_conv3)
            #shortcut path
            W_shortcut =weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            #final
            add = tf.add(x_shortcut, X)
            #建立最后融合的权重
            b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add+ b_conv_fin)


        return add_result

images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)

print(type(images_test))
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [batch_size])



#输入为  24*24*3
weight1 = weight_variable(shape=[7,7, 3, 64])
kernel1 = tf.nn.conv2d(image_holder, weight1, [1,2, 2, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
conv1 = tf.nn.relu(kernel1 + bias1)    #12*12*64
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)    #6*6*64



#第二层  输入6*6*64
weight2 = weight_variable(shape=[1, 1, 64, 64])
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(kernel2 + bias2)    #6*6*64

#第三层  输入6*6*128
weight3 = weight_variable(shape=[3,3 , 64, 64])
kernel3 = tf.nn.conv2d(conv2, weight3, [1, 1, 1, 1], padding='SAME')
bias3 = tf.Variable(tf.constant(0.1, shape=[64]))
conv3= tf.nn.relu(kernel3 + bias3)    #6*6*64

#第四层
weight4 = weight_variable(shape=[1,1 , 64, 256])
kernel4 = tf.nn.conv2d(conv3, weight4, [1, 1, 1, 1], padding='SAME')
bias4 = tf.Variable(tf.constant(0.1, shape=[256]))
conv4= tf.nn.relu(kernel4+ bias4)    #6*6*256

#将第一层的网络残差传递给第四层  6*6*64到6*6*256

w_shortcut = weight_variable([1,1,64,256])
x_shortcut = tf.nn.conv2d(norm1,w_shortcut,strides=[1,1,1,1],padding='SAME')
add = tf.add(x_shortcut,conv4)  #6*6*256



#全链接
reshape = tf.reshape(add, [batch_size, -1])  #分开成为
dim = reshape.get_shape()[1].value       #
weight3 = weight_variable([dim, 384])
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
#全连接层
weight4 = weight_variable([384, 192])
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
#输出层
weight5 = weight_variable([192, 10])
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.matmul(local4, weight5) + bias5

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

print(logits)
print(label_holder)
loss = loss(logits, label_holder)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, tf.cast(label_holder, tf.int64), 1)


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


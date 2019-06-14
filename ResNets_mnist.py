#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : ResNets_mnist.py
# @Author: WangYe
# @Date  : 2019/4/16
# @Software: PyCharm

#tensorflow基于mnist数据集上的VGG11网络，可以直接运行
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#tensorflow基于mnist实现VGG11
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#x=mnist.train.images
#y=mnist.train.labels
#X=mnist.test.images
#Y=mnist.test.labels
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None, 10])
sess = tf.InteractiveSession()

def weight_variable(shape):
#这里是构建初始变量
  initial = tf.truncated_normal(shape, mean=0,stddev=0.1)
#创建变量
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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


#这里定义conv_block模块，由于该模块定义时输入和输出尺度不同，故需要进行卷积操作来改变尺度，
        # """
        # Implementation of the convolutional block as defined in Figure 4
        #
        # Arguments:
        # X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        # kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        # filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        # stage -- integer, used to name the layers, depending on their position in the network
        # block -- string/character, used to name the layers, depending on their position in the network
        # training -- train or test
        # stride -- Integer, specifying the stride to be used
        #
        # Returns:
        # X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        # """

        # defining name basis
# x2 = convolutional_block(X_input=x1, kernel_size=3,
#                           in_filter=64,
#                          out_filters=[64, 64, 256],
#                          stage=2, block='a',
#                          stride=1)
def convolutional_block( X_input, kernel_size, in_filter,
                            out_filters, stage, block, stride=2):

        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            #first  14x14x64
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='SAME')
            b_conv1 = bias_variable([f1])
            X = tf.nn.relu(X + b_conv1) #14*14*64

            #second
            W_conv2 =weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            b_conv2 = bias_variable([f2])
            X = tf.nn.relu(X+b_conv2)#14*14*64

            #third
            W_conv3 = weight_variable([1,1, f2,f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='SAME')
            b_conv3 = bias_variable([f3])
            X = tf.nn.relu(X+b_conv3) #7*7*256
            #shortcut path
            W_shortcut =weight_variable([1, 1, in_filter, f3])#1,1,64,256
                                    #14x14x64    1，1，64，256
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut,
                                      strides=[1, stride, stride, 1], padding='VALID')
                                                #第一次 1，1，1，1
            #     x_shortcut = 14 * 14 *256
            #     X = 7 * 7 *256
            #final
            add = tf.add(x_shortcut, X)
            #建立最后融合的权重
            b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add+ b_conv_fin)


        return add_result

x1 = tf.reshape(x, [-1, 28, 28, 1])
w_conv1 = weight_variable([2, 2, 1, 64])
x1 = tf.nn.conv2d(x1, w_conv1, strides=[1, 2, 2, 1], padding='SAME')
b_conv1 = bias_variable([64])
x1 = tf.nn.relu(x1 + b_conv1)
# 这里操作后变成14x14x64
x1 = tf.nn.max_pool(x1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

# stage 2
x2 = convolutional_block(X_input=x1, kernel_size=3, in_filter=64,
                         out_filters=[64, 64, 256], stage=2, block='a',
                         stride=1)
# 上述conv_block操作后，尺寸变为14x14x256
x2 = identity_block(x2, 3, 256, [64, 64, 256], stage=2, block='b')
x2 = identity_block(x2, 3, 256, [64, 64, 256], stage=2, block='c')
# 上述操作后张量尺寸变成14x14x256
x2 = tf.nn.max_pool(x2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 变成7x7x256
flat = tf.reshape(x2, [-1, 7 * 7 * 256])

w_fc1 = weight_variable([7 * 7 *256, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2


#建立损失函数，在这里采用交叉熵函数
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#初始化变量

sess.run(tf.global_variables_initializer())

for i in range(2000):
    batch = mnist.train.next_batch(50)
    #print("y",batch[1].shape)
    #accuracy ,losss= sess.run([accuracy, cross_entropy],feed_dict={x: batch[0], y: batch[1]})
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y: batch[1],
             keep_prob: 1.0
            })
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})


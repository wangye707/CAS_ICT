#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : attention.py
# @Author: WangYe
# @Date  : 2019/5/9
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
a =[ [[[0,0,0,0,0],
     [0,0,1,1,0],
     [0,0,1,1,0],
     [0,0,0,0,0],
     [0,0,0,0,0]]],
    [[[0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,0],
     [0,0,0,0,1]]],
    [[[0,0,0,0,0],
     [0,0,0,0,0],
     [0,1,1,1,0],
     [0,1,1,1,0],
     [0,1,1,1,0]]]]
b = [[[2,2,2]],[[4,4,0]],[[4,1,3]]]

X = np.array(a).reshape(3, 5, 5,1)
y = np.array(b).reshape(3, 3 , 1)
print(X.shape)#(3, 5, 5)
print(y.shape)#(3, 3)
batch_size = 1
X_holder = tf.placeholder(tf.float32, [batch_size,5, 5,1])
y_holder = tf.placeholder(tf.float32, [batch_size,3,1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
    前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """

    return tf.nn.conv2d(x, W, strides=[1, 2, 5, 1], padding='SAME')


#输入为  1,5,5,1
W_conv1 = weight_variable([2, 2, 1, 1])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv1 = bias_variable([1])
h_conv1 = tf.nn.elu(conv2d(X_holder, W_conv1) + b_conv1)
print(h_conv1)
y_conv = tf.reshape(h_conv1,(3,-1))
print(y_conv)  #Tensor("Reshape:0", shape=(3, 1), dtype=float32)
#计算IOU
def IOU(y,y_label):
    # print(y)
    # print("y_label",y_label)
    x1 = y[0]
    y1 = y[1]
    r1 = y[2]
    x2 = y_label[0]
    y2 = y_label[1]
    r2 = y_label[2]

    sqr_both = ((x1+r1)-x2)*(y1-(y2-r2))   #重叠区域
    dif = (r1*r1)-sqr_both                 #差异区域
    out = dif/(r1*r1)
    return out



loss = IOU(y_conv,y_holder[0])
# opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# opt_op = opt.minimize(loss)
train_step = tf.train.AdamOptimizer(1).minimize(loss) # 使用adam优化
aa = tf
accuracy = correct_prediction = tf.equal(y_conv, y_holder) # 计算准确度

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
line = []
saver=tf.train.Saver(max_to_keep=1)
def train_model():

    for step in range(1,100000):
        # image_batch, label_batch = sess.run([images_train, labels_train])
        num =  int((1000-step)%3)
        loss_value,_,acc = sess.run([loss,train_step,accuracy], feed_dict={X_holder: [X[num]], y_holder: [y[num]]})
        # print(y21[0][0][0],'2222222222')
        print('loss',loss_value)
        print("acc",acc)
        line.append(loss_value)
        # saver.save(sess, 'ckpt/attention.ckpt', global_step=step + 1)
    plt.plot(line)
    plt.show()

def load_model():
    with tf.Session() as sess:
        # saver.save(sess, './ckpt/attention.ckpt-1439.meta')
        saver.restore(sess, tf.train.latest_checkpoint("./ckpt/"))
        # print(sess.run())
        x_test = [[[0,0,0,0,0],
                    [0,0,1,1,0],
                    [0,0,1,1,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]]]
        x_test = np.array(x_test).reshape(1, 5, 5,1)

        y = sess.run([y_conv],feed_dict={X_holder:x_test})
        print(y)
train_model()
# load_model()
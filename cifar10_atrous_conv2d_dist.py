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
import os
import time
FLAGS = tf.app.flags.FLAGS
#cifar10.maybe_download_and_extract()
max_steps = 1000
batch_size = 500
data_dir = r'./cifar10_data/cifar-10-batches-bin'


# For distributed
tf.app.flags.DEFINE_string("ps_hosts","localhost:11111",
                           "Comma-separated list of hostname:port pairs")
#,localhost:111113,localhost:111114
tf.app.flags.DEFINE_string("worker_hosts", "localhost:111112,localhost:111113,localhost:111114",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "ps", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 1, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")
tf.app.flags.DEFINE_string("cuda", "0", "specify gpu")
#FLAGS = tf.app.flags.FLAGS
if FLAGS.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda







def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var




start = time.clock() #计算开始时间


def main(_):


    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

            images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                            data_dir=data_dir,
                                                            batch_size=batch_size)

            image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
            label_holder = tf.placeholder(tf.float32, [batch_size])
            # 第一层

            # 输入为   batch*24*24*3   卷积后为b*24*24*64   池化后为b*14*14*64   b 是batch的简写
            weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=0.05, w1=0)
            kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
            bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
            # conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
            conv1 = tf.nn.relu(kernel1 + bias1)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

            # 第二层  输入b*14*14*64  卷积后卫 b*14*14*64    池化后卫为   b*7*7*64
            weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=0.05, w1=0.0)
            kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
            bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
            conv2 = tf.nn.relu(kernel2 + bias1)
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            # 第三层   自己添加一层   Atrous Convolution卷积层  空洞卷积（膨胀卷积）
            weight3 = variable_with_weight_loss(shape=[5, 5, 64, 128], stddev=0.05, w1=0.0)
            """
            rate： 
            要求是一个int型的正数，正常的卷积操作应该会有stride（即卷积核的滑动步长），
            但是空洞卷积是没有stride参数的，这一点尤其要注意。取而代之，它使用了新的rate参数，
            那么rate参数有什么用呢？它定义为我们在输入图像上卷积时的采样间隔，
            你可以理解为卷积核当中穿插了（rate-1）数量的“0”，把原来的卷积核插出了很多“洞洞”，
            这样做卷积时就相当于对原图像的采样间隔变大了。具体怎么插得，可以看后面更加详细的描述。
            此时我们很容易得出rate=1时，就没有0插入，此时这个函数就变成了普通卷积。
            """

            # 输入为  b*7*7*64  空洞卷积缺少strides参数，所以仍然卷积之后是  b*7*7*128
            kernel3 = tf.nn.atrous_conv2d(pool2, weight3, rate=2, padding='SAME')  # 空洞卷积会把上面的卷积核编程  10*10的大小
            bias3 = tf.Variable(tf.constant(0.1, shape=[128]))  # 输出维度128
            conv3 = tf.nn.relu(kernel3 + bias3)  # 输出为b*7*7*128
            norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # strides的步长上下都是2，输出为b*4*4*128

            # 全链接
            reshape = tf.reshape(pool3, [batch_size, -1])  # 分开成为
            dim = reshape.get_shape()[1].value  #
            weight3 = variable_with_weight_loss([dim, 384], stddev=0.04, w1=0.004)
            bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
            local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
            # 全连接层
            weight4 = variable_with_weight_loss([384, 192], stddev=0.04, w1=0.004)
            bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
            local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
            # 输出层
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



            loss_value = loss(logits, label_holder)
            optimizer = tf.train.AdamOptimizer()
            grads_and_vars = optimizer.compute_gradients(loss_value)
            top_k_op = tf.nn.in_top_k(logits, tf.cast(label_holder, tf.int64), 1)

            if issync == 1:
                # 同步模式计算更新梯度
                rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=6,
                                                        #                     replica_id=FLAGS.task_index,
                                                        total_num_replicas=6,
                                                        use_locking=True)
                train_op = rep_op.apply_gradients(grads_and_vars,
                                                  global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()
            else:
                # 异步模式计算更新梯度
                train_op = optimizer.apply_gradients(grads_and_vars,
                                                     global_step=global_step)

            init_op = tf.initialize_all_variables()

            # saver = tf.train.Saver()
            tf.summary.scalar('cost', loss_value)
            summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 #  logdir="./checkpoint/",
                                 init_op=init_op,
                                 summary_op=None,
                                 #  saver=saver,
                                 global_step=global_step,
                                 # save_model_secs=60
                                 )

        with sv.prepare_or_wait_for_session(server.target) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)


            # sess = tf.InteractiveSession()
            # tf.global_variables_initializer().run()
            # tf.train.start_queue_runners()
            for step in range(max_steps):
                start_time = time.time()
                image_batch, label_batch = sess.run([images_train, labels_train])
                _, loss = sess.run([train_op, loss_value], feed_dict={image_holder: image_batch, label_holder: label_batch})
                duration = time.time() - start_time
                if step % 10 == 0:
                    examples_per_sec = batch_size / duration
                    sec_per_batch = float(duration)

                    format_str = ('step %d,loss=%.2f (%.1f examples/sec;%.3f sec/batch)')
                    print(format_str % (step, loss, examples_per_sec, sec_per_batch))

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
            end = time.clock()  # 计算程序结束时间
            out = (end - start)

            precision = float(true_count) / total_sample_count
            print('precision @ 1 =%.3f' % precision)
            print("running time is", out, "s")


if __name__ == "__main__":
    tf.app.run()
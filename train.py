#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' train module '

__author__ = 'Ma Cong'

import tensorflow as tf

import inference

class train_net(object):
    def __init__(self, n_in, n_hidden, n_out, checkpoint, epoch=1000, lr=0.01, lr_decay=0.99, regular_rate=0.001, moving_avg_decay=0.99):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_epoch = epoch
        self.checkpoint_path = checkpoint
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.regular_rate = regular_rate
        self.moving_avg_decay = moving_avg_decay

    def train(self, xs, ys):
        x_min, x_max = xs.min(axis=0), xs.max(axis=0)
        xs = (xs - x_min) / (x_max - x_min)
        y_min, y_max = ys.min(axis=0), ys.max(axis=0)
        ys = (ys - y_min) / (y_max - y_min)

        # 需要保存量程用于预测
        x_min = tf.Variable(x_min, dtype=tf.float32, name='x_min')
        x_max = tf.Variable(x_max, dtype=tf.float32, name='x_max')
        y_min = tf.Variable(y_min, dtype=tf.float32, name='y_min')
        y_max = tf.Variable(y_max, dtype=tf.float32, name='y_max')

        x = tf.placeholder(
            tf.float32,
            [None, self.n_in],
            name='x-input'
        )
        y_ = tf.placeholder(
            tf.float32,
            [None, self.n_out],
            name='y-input'
        )
        # regularizer = tf.contrib.layers.l2_regularizer(self.regular_rate)
        regularizer = None
        y = inference.inference(x, regularizer, self.n_in, self.n_out, self.n_hidden)
        global_step = tf.Variable(0, trainable=False)

        variable_averages = tf.train.ExponentialMovingAverage(
            self.moving_avg_decay, global_step
        )
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables()
        )

        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=y, labels=tf.arg_max(y_, 1)
        # )
        # cross_entropy_mean = tf.reduce_mean(cross_entropy)
        # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        mse = tf.reduce_mean(tf.square(y - y_))
        # loss = mse + tf.add_n(tf.get_collection('losses'))
        loss = mse

        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            global_step,
            1000,  # n_examples/batch_size
            self.lr_decay
        )
        # train_step = tf.train.GradientDescentOptimizer(
        #     self.learning_rate
        # ).minimize(loss, global_step=global_step)
        train_step = tf.train.AdamOptimizer(
            self.learning_rate
        ).minimize(loss, global_step=global_step)
        # with tf.control_dependencies([train_step, variable_averages_op]):
        with tf.control_dependencies([train_step]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for i in range(self.n_epoch):
                _, loss_value, step = sess.run(
                    [train_op, loss, global_step],
                    feed_dict={x: xs, y_: ys}
                )

                if (i+1) % 1000 == 0:
                    print('After %d training steps, loss is %g.'
                          % (step, loss_value))
                    saver.save(sess, self.checkpoint_path, global_step=global_step)



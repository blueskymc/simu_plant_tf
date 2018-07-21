#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' test module '

__author__ = 'Ma Cong'

import tensorflow as tf
import numpy as np

import inference

class Test(object):
    def __init__(self, n_in, n_hidden, n_out, checkpoint):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.checkpoint_path = checkpoint

    def test(self, x):
        x_min = tf.Variable(np.zeros(x.shape[1]), dtype=tf.float32, name='x_min')
        x_max = tf.Variable(np.zeros(x.shape[1]), dtype=tf.float32, name='x_max')
        y_min = tf.Variable([0], dtype=tf.float32, name='y_min')
        y_max = tf.Variable([0], dtype=tf.float32, name='y_max')

        input_x = tf.placeholder(
            tf.float32,
            [None, self.n_in],
            name='x-input'
        )
        y = inference.inference(input_x, None, self.n_in, self.n_out, self.n_hidden)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('ckpt')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                x_min, x_max, y_min, y_max = sess.run([x_min, x_max, y_min, y_max])
                x = (x - x_min) / (x_max - x_min)
                result = sess.run(y, feed_dict={input_x: x})
                result = result * (y_max - y_min) + y_min
                print(result)
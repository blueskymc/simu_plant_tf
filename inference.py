#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' net forward'

__author__ = 'Ma Cong'

import tensorflow as tf

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer, n_in, n_out, n_hidden):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [n_in, n_hidden],
            regularizer
        )
        biases = tf.get_variable(
            'biases', [n_hidden], initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [n_hidden, n_out],
            regularizer
        )
        biases = tf.get_variable(
            'biases', [n_out], initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.nn.sigmoid(tf.matmul(layer1, weights) + biases)

    return layer2
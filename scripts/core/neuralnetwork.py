from __future__ import division

from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layer




def batch_norm(input, is_training, scope_bn):
    bn_train = layer.batch_norm(input, decay=0.999, center=True, scale=True,
                                is_training=True,
                                updates_collections=None,
                                reuse=None,
                                trainable=True,
                                scope=scope_bn)

    bn_inference = layer.batch_norm(input, decay=0.999, center=True, scale=True,
                                    is_training=False,
                                    updates_collections=None,
                                    reuse=True,
                                    trainable=True,
                                    scope=scope_bn)

    output = tf.cond(is_training, lambda: bn_train, lambda: bn_inference)
    return output


def dropout(input, is_training, keep_prob):
    return tf.cond(is_training, lambda: tf.nn.dropout(input, keep_prob), lambda: tf.nn.dropout(input, 1.0))


def max_pool(name, input, factor):
    pool = tf.nn.max_pool(input, ksize= [1, factor, factor, 1], strides= [1, factor, factor, 1], padding="SAME", name=name)
    return pool


def average_pool(name, input, factor):
    pool = tf.nn.avg_pool(input, ksize= [1, factor, factor, 1], strides= [1, factor, factor, 1], padding="SAME", name=name)
    return pool


def conv2d(name, input, filter, stride, bias):
    conv = tf.nn.conv2d(input, filter=filter, strides= [1, stride, stride, 1], padding="SAME")
    conv = tf.nn.bias_add(conv, bias, name=name)
    return conv


def transpose_conv2d(name, input, filter, stride, bias, output_shape):
    if output_shape is None:
        output_shape = input.get_shape().as_list()
        output_shape[1] *= (output_shape[1] - 1) * stride + 1
        output_shape[2] *= (output_shape[2] - 1) * stride + 1
    conv = tf.nn.conv2d_transpose(input, filter, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    conv = tf.nn.bias_add(conv, bias, name=name)
    return conv









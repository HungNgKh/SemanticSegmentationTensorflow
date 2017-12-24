from __future__ import division

from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layer




def batch_norm(input, is_training, scope_bn):
    # bn_train = layer.batch_norm(input, decay=0.999, epsilon=1e-3, center=True, scale=True,
    #                             is_training=True,
    #                             updates_collections=None,
    #                             reuse=None,
    #                             trainable=True,
    #                             scope=scope_bn)
    #
    # bn_inference = layer.batch_norm(input, decay=0.999, epsilon=1e-3, center=True, scale=True,
    #                                 is_training=False,
    #                                 updates_collections=None,
    #                                 reuse=True,
    #                                 trainable=True,
    #                                 scope=scope_bn)
    #
    # output = tf.cond(is_training, lambda: bn_train, lambda: bn_inference)
    # return output
    with tf.variable_scope(scope_bn):
        n_output = input.get_shape().as_list()[3]
        beta = tf.Variable(tf.constant(0.0, shape=[n_output]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_output]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)


    return normed


def dropout(input, is_training, keep_prob):
    return tf.cond(is_training, lambda: tf.nn.dropout(input, keep_prob), lambda: tf.nn.dropout(input, 1.0))


def max_pool(name, input, factor):
    pool = tf.nn.max_pool(input, ksize= [1, factor, factor, 1], strides= [1, factor, factor, 1], padding="SAME", name=name)
    return pool


def average_pool(name, input, factor):
    pool = tf.nn.avg_pool(input, ksize= [1, factor, factor, 1], strides= [1, factor, factor, 1], padding="SAME", name=name)
    return pool


def conv2d(name, input, weight_shape, stride, weight_init, GPU_var_device = True, GPU_op_device = True):
    with tf.device("/GPU:0" if GPU_var_device else "/CPU:0"), tf.variable_scope(name) as scope:
        weights = tf.get_variable(name='weights', shape=weight_shape, initializer=weight_init)
        biases = tf.get_variable(name='biases', shape=weight_shape[-1], initializer=tf.zeros_initializer)

    with tf.device("/GPU:0" if GPU_op_device else "/CPU:0"):
        return tf.nn.bias_add(tf.nn.conv2d(input, filter=weights, strides= [1, stride, stride, 1], padding="SAME"), biases, name=name)


def transpose_conv2d(name, input, weight_shape, stride, output_shape, GPU_var_device = True, GPU_op_device = True):
    if output_shape is None:
        output_shape = input.get_shape().as_list()
        output_shape[1] *= (output_shape[1] - 1) * stride + 1
        output_shape[2] *= (output_shape[2] - 1) * stride + 1

    with tf.device("/GPU:0" if GPU_var_device else "/CPU:0"), tf.variable_scope(name) as scope:
        weights = get_bilinear_filter(weight_shape, upscale_factor=stride, name='weights')

    with tf.device("/GPU:0" if GPU_op_device else "/CPU:0"):
        return tf.nn.conv2d_transpose(input, filter=weights, output_shape=output_shape, strides=[1, stride, stride, 1], padding="SAME", name=name)



def hinge_loss(pred, labels):
    true_classes = tf.argmax(labels, 1)
    idx_flattened = tf.range(0, tf.shape(pred)[0]) * tf.shape(pred)[1] + tf.cast(true_classes, dtype=tf.int32)

    true_scores = tf.cast(tf.gather(tf.reshape(pred, [-1]),
                            idx_flattened), dtype=tf.float32)

    L = tf.nn.relu((1 + tf.transpose(tf.nn.bias_add(tf.transpose(pred), tf.negative(true_scores)))) * (1 - labels))

    final_loss = tf.reduce_mean(tf.reduce_sum(L,axis=1))
    return final_loss


def get_bilinear_filter(filter_shape, upscale_factor, name):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name=name, initializer=init,
                                       shape=weights.shape)
    return bilinear_weights



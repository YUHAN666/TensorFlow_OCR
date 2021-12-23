import numpy as np
import tensorflow as tf


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """
    Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.initializers.variance_scaling uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def conv_stage(inputs, filters, kernel_size, strides, training, padding='same', data_format='channels_last',
               activation='relu', bn=True, momentum=0.99, pool=False, pool_size=(2, 2), pool_stride=(2, 2),
               use_bias=False, name=None):
    axis = 1 if data_format == 'channels_first' else -1

    x = tf.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, data_format=data_format,
                         name=name + '_Conv2D', padding=padding, use_bias=use_bias,
                         kernel_initializer=conv_kernel_initializer)(inputs)
    if bn:
        x = tf.layers.batch_normalization(x, axis=axis, training=training, momentum=momentum, name=name + '_bn')

    if activation:
        x = tf.nn.relu(x, name=name + '_relu')

    if pool:
        x = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=pool_stride, data_format=data_format, padding='same', name=name+'_maxpool')

    return x


def transpose_conv_stage(inputs, filters, kernel_size, strides, training, padding='valid', data_format='channels_last',
               activation='relu', bn=True, momentum=0.99, pool=False, pool_size=(2, 2), pool_stride=(2, 2),
               use_bias=False, name=None):

    axis = 1 if data_format == 'channels_first' else -1

    x = tf.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                  data_format=data_format, activation=None, use_bias=use_bias,
                                  kernel_initializer='he_normal', name=name+'_TransposeConv2D')(inputs)

    if bn:
        x = tf.layers.batch_normalization(x, axis=axis, training=training, momentum=momentum, name=name + '_bn')

    if activation == 'relu':
        x = tf.nn.relu(x, name=name + '_relu')
    elif activation == 'sigmoid':
        x = tf.nn.sigmoid(x, name=name+'_sigmoid')

    if pool:
        x = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=pool_stride, data_format=data_format, name=name+'_maxpool')

    return x

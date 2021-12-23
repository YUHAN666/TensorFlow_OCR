import tensorflow as tf
import numpy as np


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


def darknet_convblock(inputs, filters, kernel_size, strides, training, momentum, name, use_bias='False'):
	padding = 'valid' if strides == (2, 2) else 'same'
	with tf.variable_scope(name):
		x = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
		                     use_bias=use_bias, kernel_regularizer=tf.keras.regularizers.l2(5e-4),
		                     kernel_initializer=conv_kernel_initializer)(inputs)

		x = tf.layers.batch_normalization(x, training=training, momentum=momentum)

		x = tf.nn.leaky_relu(x, alpha=0.1)

	return x


def darknet_resblock(inputs, num_blocks, filters, training, momentum, name):
	with tf.variable_scope(name):
		x = tf.pad(inputs, [[0, 0], [1, 0], [1, 0], [0, 0]], name='pad')
		x = darknet_convblock(x, filters, (3, 3), (2, 2), training=training, momentum=momentum, name='downsample')
		for i in range(num_blocks):
			y = darknet_convblock(x, filters // 2, (1, 1), (1, 1), training=training, momentum=momentum,
			                      name='{}_1X1'.format(i))
			y = darknet_convblock(y, filters, (3, 3), (1, 1), training=training, momentum=momentum,
			                      name='{}_3X3'.format(i))
			x = tf.add(x, y, name='add{}'.format(i))

	return x


def darknet_body(inputs, training, momentum, name='darknet'):
	with tf.variable_scope(name):
		x = darknet_convblock(inputs, 32, (3, 3), (1, 1), training=training, momentum=momentum, name='input')
		x = darknet_resblock(x, 1, 64, training=training, momentum=momentum, name='resblock1')
		x = darknet_resblock(x, 2, 128, training=training, momentum=momentum, name='resblock2')
		x8 = darknet_resblock(x, 8, 256, training=training, momentum=momentum, name='resblock3')
		x16 = darknet_resblock(x8, 8, 512, training=training, momentum=momentum, name='resblock4')
		x = darknet_resblock(x16, 4, 1024, training=training, momentum=momentum, name='resblock5')

	return x, x8, x16


def darknet_head(inputs, filters, out_filters, training, momentum, name):
	x = darknet_convblock(inputs, filters, (1, 1), (1, 1), training=training, momentum=momentum, name=name + '1')
	x = darknet_convblock(x, filters * 2, (3, 3), (1, 1), training=training, momentum=momentum, name=name + '2')
	x = darknet_convblock(x, filters, (1, 1), (1, 1), training=training, momentum=momentum, name=name + '3')
	x = darknet_convblock(x, filters * 2, (3, 3), (1, 1), training=training, momentum=momentum, name=name + '4')
	x = darknet_convblock(x, filters, (1, 1), (1, 1), training=training, momentum=momentum, name=name + '5')
	# x = tf.layers.Conv2D(filters, (1,1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name=name + '5' )(x)

	y = darknet_convblock(x, filters * 2, (3, 3), (1, 1), training=training, momentum=momentum, name=name + '6')
	# y = darknet_convblock(y, out_filters, (1, 1), (1, 1), training=training, momentum=momentum, name=name + '7')
	y = tf.layers.Conv2D(out_filters, (1, 1), padding='same', use_bias=False,
	                     kernel_regularizer=tf.keras.regularizers.l2(5e-4), kernel_initializer=conv_kernel_initializer,
	                     name=name + '_out')(y)
	return x, y

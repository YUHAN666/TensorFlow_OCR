import tensorflow as tf
from mish import mish


def conv_bn_activation(inputs, filters, kernel_size, strides, padding, bn=True, training=False, momentum=0.99,
                       activation=None, use_bias=False, name=None):
	x = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias,
	                     name=name + '_conv2d')(inputs)

	if bn:
		x = tf.layers.batch_normalization(x, momentum=momentum, training=training, name=name + '_bn')

	if activation == 'mish':
		x = mish(x, name=name + '_mish')
	elif activation == 'leaky':
		x = tf.nn.leaky_relu(x, name=name + '_leaky_relu')
	elif activation == 'linear':
		pass

	return x


def res_block(inputs, filters, num_blocks, short_cut, bn, training, momentum, name):
	with tf.variable_scope(name):
		x = inputs
		for n in range(num_blocks):
			x = conv_bn_activation(x, filters, (1, 1), (1, 1), 'same', bn, training, momentum, 'mish', False,
			                       name='block{}1'.format(n + 1))
			x = conv_bn_activation(x, filters, (3, 3), (1, 1), 'same', bn, training, momentum, 'mish', False,
			                       name='block{}2'.format(n + 1))

		if short_cut:
			x = tf.add(inputs, x, name='add')

	return x

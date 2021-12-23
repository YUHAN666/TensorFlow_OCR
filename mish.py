import tensorflow as tf


def mish(inputs, name):

	return inputs * tf.nn.tanh(tf.nn.softplus(inputs), name=name)
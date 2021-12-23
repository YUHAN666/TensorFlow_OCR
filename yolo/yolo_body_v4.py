import tensorflow as tf
from yolo.yolo_block_v4 import conv_bn_activation, res_block


def down_sample_module1(inputs, training, momentum):
	with tf.variable_scope('DownSample1'):
		x1 = conv_bn_activation(inputs, 32, (3, 3), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block1')
		x2 = conv_bn_activation(x1, 64, (3, 3), (2, 2), 'same', True, training, momentum, activation='mish',
		                        name='block2')
		x3 = conv_bn_activation(x2, 64, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block3')
		x4 = conv_bn_activation(x2, 64, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block4')
		x5 = conv_bn_activation(x4, 32, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block5')
		x6 = conv_bn_activation(x5, 64, (3, 3), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block6')
		x6 = tf.add(x4, x6, name='add')

		x7 = conv_bn_activation(x6, 64, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block7')
		x7 = tf.concat([x7, x3], axis=-1, name='concat')

		x8 = conv_bn_activation(x7, 64, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block8')

	return x8


def down_sample_module2(inputs, training, momentum):
	with tf.variable_scope('DownSample2'):
		x1 = conv_bn_activation(inputs, 128, (3, 3), (2, 2), 'same', True, training, momentum, activation='mish',
		                        name='block1')
		x2 = conv_bn_activation(x1, 64, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block2')
		x3 = conv_bn_activation(x1, 64, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block3')
		r = res_block(x3, filters=64, num_blocks=2, short_cut=True, bn=True, training=training, momentum=momentum,
		                        name='ResModule')
		x4 = conv_bn_activation(r, 64, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block4')
		x4 = tf.concat([x2, x4], axis=-1, name='concat')
		x5 = conv_bn_activation(x4, 128, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block5')

		return x5


def down_sample_module3(inputs, training, momentum):
	with tf.variable_scope('DownSample3'):
		x1 = conv_bn_activation(inputs, 256, (3, 3), (2, 2), 'same', True, training, momentum, activation='mish',
		                        name='block1')
		x2 = conv_bn_activation(x1, 128, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block2')
		x3 = conv_bn_activation(x1, 128, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block3')
		r = res_block(x3, filters=128, num_blocks=8, short_cut=True, bn=True, training=training, momentum=momentum,
		                        name='ResModule')
		x4 = conv_bn_activation(r, 128, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block4')
		x4 = tf.concat([x2, x4], axis=-1, name='concat')
		x5 = conv_bn_activation(x4, 256, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block5')

		return x5


def down_sample_module4(inputs, training, momentum):
	with tf.variable_scope('DownSample4'):
		x1 = conv_bn_activation(inputs, 512, (3, 3), (2, 2), 'same', True, training, momentum, activation='mish',
		                        name='block1')
		x2 = conv_bn_activation(x1, 256, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block2')
		x3 = conv_bn_activation(x1, 256, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block3')

		r = res_block(x3, filters=256, num_blocks=8, short_cut=True, bn=True, training=training,
		              momentum=momentum, name='ResModule')
		x4 = conv_bn_activation(r, 256, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block4')
		x4 = tf.concat([x2, x4], axis=-1, name='concat')
		x5 = conv_bn_activation(x4, 512, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block5')

		return x5


def down_sample_module5(inputs, training, momentum):
	with tf.variable_scope('DownSample5'):
		x1 = conv_bn_activation(inputs, 1024, (3, 3), (2, 2), 'same', True, training, momentum, activation='mish',
		                        name='block1')
		x2 = conv_bn_activation(x1, 512, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block2')
		x3 = conv_bn_activation(x1, 512, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block3')
		r = res_block(x3, filters=512, num_blocks=4, short_cut=True, bn=True, training=training,
		              momentum=momentum, name='ResModule')
		x4 = conv_bn_activation(r, 512, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block4')
		x4 = tf.concat([x2, x4], axis=-1, name='concat')
		x5 = conv_bn_activation(x4, 1024, (1, 1), (1, 1), 'same', True, training, momentum, activation='mish',
		                        name='block5')

		return x5


def yolo_neck(inputs, downsample4, downsample3, training, momentum):
	with tf.variable_scope('Neck'):
		x1 = conv_bn_activation(inputs, 512, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                        activation='leaky', name='block1')
		x2 = conv_bn_activation(x1, 1024, (3, 3), (1, 1), 'same', training=training, momentum=momentum,
		                        activation='leaky', name='block2')
		x3 = conv_bn_activation(x2, 512, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                        activation='leaky', name='block3')
		m1 = tf.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same', name='maxpool5')(x3)
		m2 = tf.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same', name='maxpool9')(x3)
		m3 = tf.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same', name='maxpool13')(x3)
		spp = tf.concat([m3, m2, m1, x3], axis=-1, name='spp_concat')


		x4 = conv_bn_activation(spp, 512, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                        activation='leaky', name='block4')
		x5 = conv_bn_activation(x4, 1024, (3, 3), (1, 1), 'same', training=training, momentum=momentum,
		                        activation='leaky', name='block5')
		x6 = conv_bn_activation(x5, 512, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                        activation='leaky', name='block6')
		x7 = conv_bn_activation(x6, 256, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                        activation='leaky', name='block7')
		up32 = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample32')(x7)
		x8 = conv_bn_activation(downsample4, 256, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                        activation='leaky', name='block8')
		x8 = tf.concat([x8, up32], axis=-1, name='concat16')


		x9 = conv_bn_activation(x8, 256, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                        activation='leaky', name='block9')
		x10 = conv_bn_activation(x9, 512, (3, 3), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block10')
		x11 = conv_bn_activation(x10, 256, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block11')
		x12 = conv_bn_activation(x11, 512, (3, 3), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block12')
		x13 = conv_bn_activation(x12, 256, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block13')
		x14 = conv_bn_activation(x13, 128, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block14')
		up16 = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample16')(x14)
		x15 = conv_bn_activation(downsample3, 128, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block15')
		x15 = tf.concat([x15, up16], axis=-1, name='concat8')


		x16 = conv_bn_activation(x15, 128, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block16')
		x17 = conv_bn_activation(x16, 256, (3, 3), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block17')
		x18 = conv_bn_activation(x17, 128, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block18')
		x19 = conv_bn_activation(x18, 256, (3, 3), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block19')
		x20 = conv_bn_activation(x19, 128, (1, 1), (1, 1), 'same', training=training, momentum=momentum,
		                         activation='leaky', name='block20')

		return x20, x13, x6


def yolo_head(inputs8, inputs16, inputs32, num_anchors, class_num, training, momentum):
	with tf.variable_scope('YoloHead'):
		x1 = conv_bn_activation(inputs8, 256, (3,3), (1,1), 'same', True, training, momentum, 'leaky', name='block1')
		y8 =conv_bn_activation(x1, num_anchors//3*(5+class_num), (1,1), (1,1), 'same', bn=False, activation='linear', name='output8')

		x3 =conv_bn_activation(inputs8, 256, (3,3), (2,2), 'same', True, training, momentum, 'leaky', name='block3')
		x3 = tf.concat([x3, inputs16], axis=-1, name='concat16')
		x4 =conv_bn_activation(x3, 256, (1,1), (1,1), 'same', True, training, momentum, 'leaky', name='block4')
		x5 = conv_bn_activation(x4, 512, (3, 3), (1, 1), 'same', True, training, momentum, 'leaky', name='block5')
		x6 =  conv_bn_activation(x5, 256, (1, 1), (1, 1), 'same', True, training, momentum, 'leaky', name='block6')
		x7= conv_bn_activation(x6, 512, (3, 3), (1, 1), 'same', True, training, momentum, 'leaky', name='block7')
		x8 =  conv_bn_activation(x7, 256, (1, 1), (1, 1), 'same', True, training, momentum, 'leaky', name='block8')
		x9 = conv_bn_activation(x8, 512, (3, 3), (1, 1), 'same', True, training, momentum, 'leaky', name='block9')
		y16 = conv_bn_activation(x9, num_anchors//3*(5+class_num), (1, 1), (1, 1), 'same', bn=False, activation='linear', name='output16')

		x11 = conv_bn_activation(x8, 512, (3,3), (2,2), 'same', True, training, momentum, 'leaky', name='block11')
		x11 = tf.concat([x11, inputs32], axis=-1, name='concat32')
		x12 =conv_bn_activation(x11, 512, (1,1), (1,1), 'same', True, training, momentum, 'leaky', name='block12')
		x13 = conv_bn_activation(x12, 1024, (3, 3), (1, 1), 'same', True, training, momentum, 'leaky', name='block13')
		x14 = conv_bn_activation(x13, 512, (1,1), (1,1), 'same', True, training, momentum, 'leaky', name='block14')
		x15 = conv_bn_activation(x14, 1024, (3,3), (1,1), 'same', True, training, momentum, 'leaky', name='block15')
		x16 = conv_bn_activation(x15, 512, (1,1), (1,1), 'same', True, training, momentum, 'leaky', name='block16')
		x17 = conv_bn_activation(x16, 1024, (3,3), (1,1), 'same', True, training, momentum, 'leaky', name='block17')
		y32 = conv_bn_activation(x17, num_anchors//3*(5+class_num), (1,1), (1,1), 'same', bn=False, activation='linear', name='output32')

		return y32, y16, y8


def yolo_body(inputs, num_anchors, class_num, training, momentum):

	d1 = down_sample_module1(inputs, training, momentum)
	d2 = down_sample_module2(d1, training, momentum)
	d3 = down_sample_module3(d2, training, momentum)
	d4 = down_sample_module4(d3, training, momentum)
	d5 = down_sample_module5(d4, training, momentum)

	x8, x16, x32 = yolo_neck(d5, d4, d3, training, momentum)

	y32, y16, y8 = yolo_head(x8, x16, x32, num_anchors, class_num, training, momentum)

	return y32, y16, y8
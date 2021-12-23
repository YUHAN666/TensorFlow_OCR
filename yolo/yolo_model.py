import tensorflow as tf

from base.model import Model
from yolo.darknet import darknet_convblock, darknet_head, darknet_body, conv_kernel_initializer


# from yolo_body_v4 import yolo_body


def yolo_body(inputs, num_anchors, num_classes, training, momentum):

	x, x8, x16 = darknet_body(inputs, training=training, momentum=momentum)

	x, y32 = darknet_head(x, 512, num_anchors//3 * (num_classes + 5), training=training, momentum=momentum,
	                      name='yolo_dw32fea')

	x = darknet_convblock(x, 256, (1, 1), (1, 1), training=training, momentum=momentum, name='yolo_dw32fea8')
	x = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample32to16')(x)
	x = tf.concat([x, x16], axis=-1, name='concat16')
	x, y16 = darknet_head(x, 256, num_anchors//3 * (num_classes + 5), training=training, momentum=momentum,
	                      name='yolo_dw16fea')

	x = darknet_convblock(x, 128, (1, 1), (1, 1), training=training, momentum=momentum, name='yolo_dw16fea8')
	x = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample16to8')(x)
	x = tf.concat([x, x8], axis=-1, name='concat8')
	x, y8 = darknet_head(x, 128, num_anchors//3 * (num_classes + 5), training=training, momentum=momentum,
	                     name='yolo_dw8fea')

	return y32, y16, y8


def tiny_yolo_body(inputs, num_anchors, num_classes, training, momentum):
	x = darknet_convblock(inputs, 16, (3, 3), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv1')
	x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='tiny_yolo_pool1')(x)
	x = darknet_convblock(x, 32, (3, 3), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv2')
	x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='tiny_yolo_pool2')(x)
	x = darknet_convblock(x, 64, (3, 3), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv3')
	x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='tiny_yolo_pool3')(x)
	x = darknet_convblock(x, 128, (3, 3), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv4')
	x = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='tiny_yolo_pool4')(x)
	x = darknet_convblock(x, 256, (3, 3), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv5')

	x2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='tiny_yolo_pool5')(x)
	x2 = darknet_convblock(x2, 512, (3, 3), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv6')
	x2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', name='tiny_yolo_pool6')(x2)
	x2 = darknet_convblock(x2, 1024, (3, 3), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv7')
	x2 = darknet_convblock(x2, 256, (1, 1), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv8')

	y32 = darknet_convblock(x2, 512, (3, 3), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv9')
	# y32 = darknet_convblock(y32, num_anchors//2 * (num_classes + 5), (1, 1), (1, 1), training=training, momentum=momentum,
	#                         name='dw32output')
	y32 = tf.layers.Conv2D(filters=num_anchors//2 * (num_classes + 5), kernel_size=(1,1), padding='same',
	                       kernel_regularizer=tf.keras.regularizers.l2(5e-4),use_bias=False,kernel_initializer=conv_kernel_initializer , name='dw32output')(y32)


	x2 = darknet_convblock(x2, 128, (1, 1), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv10')
	x2 = tf.keras.layers.UpSampling2D(size=(2, 2), name='32to16upsample')(x2)

	y16 = tf.concat([x, x2], axis=-1, name='concat16')
	y16 = darknet_convblock(y16, 256, (3, 3), (1, 1), training=training, momentum=momentum, name='tiny_yolo_conv11')
	# y16 = darknet_convblock(y16, num_anchors//2 * (num_classes + 5), (1, 1), (1, 1), training=training, momentum=momentum,
	#                         name='dw16output')
	y16 = tf.layers.Conv2D(filters=num_anchors//2 * (num_classes + 5), kernel_size=(1,1), padding='same',
	                       kernel_regularizer=tf.keras.regularizers.l2(5e-4), kernel_initializer=conv_kernel_initializer,
	                       use_bias=False, name='dw16output')(y16)

	return y32, y16


def yolo_body_customized(inputs, num_classes, num_anchors, training, momentum):
	x, x8, x16 = darknet_body(inputs, training=training, momentum=momentum)

	x, y32 = darknet_head(x, 512, num_anchors // 2 * (num_classes + 5), training=training, momentum=momentum,
	                      name='yolo_dw32fea')

	x = darknet_convblock(x, 256, (1, 1), (1, 1), training=training, momentum=momentum, name='yolo_dw32fea8')
	x = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample32to16')(x)
	x = tf.concat([x, x16], axis=-1, name='concat16')
	x, y16 = darknet_head(x, 256, num_anchors // 2 * (num_classes + 5), training=training, momentum=momentum,
	                      name='yolo_dw16fea')

	x = darknet_convblock(x, 128, (1, 1), (1, 1), training=training, momentum=momentum, name='yolo_dw16fea8')
	x = tf.keras.layers.UpSampling2D(size=(2, 2), name='upsample16to8')(x)
	x = tf.concat([x, x8], axis=-1, name='concat8')
	x, y8 = darknet_head(x, 128, num_anchors // 2 * (num_classes + 5), training=training, momentum=momentum,
	                     name='yolo_dw8fea')

	return y16, y8


class ModelYOLO(Model):

	def __init__(self, sess, param, anchors, logger):
		self.step = 0
		self.session = sess
		self.logger = logger
		self.bn_momentum = param["momentum"]
		self.mode = param["mode"]
		self.input_shape = param["input_shape"]
		self.image_channel = param["image_channel"]
		self.checkPoint_dir = param["checkpoint_dir"]
		self.logger.info("Building model... backbone:{}, neck:{}".format(param["backbone"], param["neck"]))
		self.batch_size = param["batch_size"]
		self.batch_size_inference = param["batch_size_inference"]
		self.anchors = anchors
		self.num_anchors = len(self.anchors)
		self.num_classes = param["num_classes"]
		self.customized_downsample = param["down_scale"]

		with self.session.as_default():
			# Build placeholder to receive data

			if self.num_anchors == 9:

				if self.mode == 'train_yolo':
					self.is_training = tf.placeholder(tf.bool, name='is_training')

					self.image_input = tf.placeholder(tf.float32,
					                                  shape=(self.batch_size, self.input_shape[0], self.input_shape[1],
					                                         self.image_channel), name='image_input')

					self.gt_input32 = tf.placeholder(tf.float32, shape=(
					self.batch_size, self.input_shape[0] // 32, self.input_shape[1] // 32, 3, 5 + self.num_classes),
					                                 name='gt_input32')

					self.gt_input16 = tf.placeholder(tf.float32, shape=(
					self.batch_size, self.input_shape[0] // 16, self.input_shape[1] // 16, 3, 5 + self.num_classes),
					                                 name='gt_input16')

					self.gt_input8 = tf.placeholder(tf.float32, shape=(
					self.batch_size, self.input_shape[0] // 8, self.input_shape[1] // 8, 3, 5 + self.num_classes),
					                                 name='gt_input8')
					self.gt_input = [self.gt_input32, self.gt_input16, self.gt_input8]
					self.yolo_output = self.build_model()
					return

				else:
					self.is_training = False

					self.image_input = tf.placeholder(tf.float32, shape=(self.batch_size_inference, self.input_shape[0],
					                                                     self.input_shape[1], self.image_channel),
					                                  name='image_input')

					self.yolo_output = self.build_model()
					return

			elif self.num_anchors == 6:
				if self.mode == 'train_yolo':
					self.is_training = tf.placeholder(tf.bool, name='is_training')

					self.image_input = tf.placeholder(tf.float32,
					                                  shape=(self.batch_size, self.input_shape[0], self.input_shape[1],
					                                         self.image_channel), name='image_input')
					self.gt_input32 = tf.placeholder(tf.float32, shape=(
						self.batch_size, self.input_shape[0] // 32, self.input_shape[1] // 32, 3, 5 + self.num_classes),
					                                 name='gt_input32')

					self.gt_input16 = tf.placeholder(tf.float32, shape=(
						self.batch_size, self.input_shape[0] // 16, self.input_shape[1] // 16, 3, 5 + self.num_classes),
					                                 name='gt_input16')
					self.gt_input = [self.gt_input32, self.gt_input16]
					# Building model graph
					self.yolo_output = self.build_tiny_model()
					return

				else:
					self.is_training = False

					self.image_input = tf.placeholder(tf.float32, shape=(self.batch_size_inference, self.input_shape[0],
					                                                     self.input_shape[1], self.image_channel),
					                                  name='yolo_image_input')

					self.yolo_output = self.build_tiny_model()
					return
			else:

				if self.mode == 'train_yolo':
					self.is_training = tf.placeholder(tf.bool, name='is_training')

					self.image_input = tf.placeholder(tf.float32,
					                                  shape=(self.batch_size, self.input_shape[0], self.input_shape[1],
					                                         self.image_channel), name='image_input')
					self.gt_input32 = tf.placeholder(tf.float32, shape=(
						self.batch_size, self.input_shape[0] // self.customized_downsample,
						self.input_shape[1] // self.customized_downsample, self.num_anchors // 2, 5 + self.num_classes),
					                                 name='gt_input32')

					self.gt_input16 = tf.placeholder(tf.float32, shape=(
						self.batch_size, self.input_shape[0] * 2 // self.customized_downsample,
						self.input_shape[1] * 2 // self.customized_downsample, self.num_anchors // 2,
						5 + self.num_classes),
					                                 name='gt_input16')
					self.gt_input = [self.gt_input32, self.gt_input16]
					# Building model graph
					self.yolo_output = self.build_model_customized()
					print("Number of anchors: {}, use customized model".format(self.num_anchors))
					return

				else:
					self.is_training = False

					self.image_input = tf.placeholder(tf.float32, shape=(self.batch_size_inference, self.input_shape[0],
					                                                     self.input_shape[1], self.image_channel),
					                                  name='yolo_image_input')
					# self.image_input = tf.placeholder(tf.float32, shape=(self.batch_size_inference, 1200,
					# 1920, self.image_channel),  name='yolo_image_input')
					# self.resized_input = tf.image.crop_and_resize(self.image_input, [[-0.3, 0, 1.3, 1]], [0], [320, 320],
					#                                extrapolation_value=128.0, method="bilinear")
					# self.resized_input = self.resized_input / 255.0

					self.yolo_output = self.build_model_customized()

					print("Number of anchors: {}, use customized model".format(self.num_anchors))
					return

	def build_model_customized(self):
		"""
		Build model graph in session
		:return: segmentation_output: nodes for calculating segmentation loss
				 decision_output: nodes for calculating decision loss
				 mask_out: nodes for visualization output mask of the model
		"""

		y32, y16 = yolo_body_customized(self.image_input, self.num_classes, self.num_anchors, self.is_training,
		                                self.bn_momentum)

		return [y32, y16]





	def build_model(self):
		"""
		Build model graph in session
		:return: segmentation_output: nodes for calculating segmentation loss
				 decision_output: nodes for calculating decision loss
				 mask_out: nodes for visualization output mask of the model
		"""

		y32, y16, y8 = yolo_body(self.image_input, self.num_anchors, self.num_classes, self.is_training,
		                         self.bn_momentum)

		return [y32, y16, y8]

	def build_tiny_model(self):
		"""
		Build model graph in session
		:return: segmentation_output: nodes for calculating segmentation loss
				 decision_output: nodes for calculating decision loss
				 mask_out: nodes for visualization output mask of the model
		"""

		y32, y16 = tiny_yolo_body(self.image_input, self.num_anchors, self.num_classes, self.is_training,
		                         self.bn_momentum)

		return [y32, y16]

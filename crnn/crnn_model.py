import tensorflow as tf
import numpy as np
from base.model import Model
from crnn.crnn_config import DefaultParam as param
from crnn.shadowNet import shadow_net, rnn_head


class ModelCrnn(Model):

	def __init__(self, sess, param):
		self.step = 0
		self.session = sess
		self.bn_momentum = param["momentum"]
		self.image_height = param["image_height"]
		self.image_width = param["image_width"]
		self.max_text_len = param["max_text_len"]
		self.image_channel = param["image_channel"]
		self.hidden_units = param["hidden_units"]
		self.hidden_layers = param["hidden_layers"]
		self.class_num = param["class_num"]
		self.mode = param["mode"]
		self.tensorboard_logdir = param["tensorboard_dir"]
		self.checkPoint_dir = param["checkpoint_dir"]
		self.batch_size = param["batch_size"]
		self.batch_size_inference = param["batch_size_inference"]

		with self.session.as_default():
			if self.mode == 'train':
				self.is_training = tf.placeholder(tf.bool, name='is_training')
				self.keep_dropout = True
				self.image_input = tf.placeholder(tf.float32,
				                                  shape=(self.batch_size, self.image_height, self.image_width,
				                                         self.image_channel), name='image_input')
				self.label = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_text_len), name='label_input')
				self.label_length = tf.placeholder(tf.int32, shape=(self.batch_size), name='label_length_input')
				self.logit_length = tf.placeholder(tf.int32, shape=(self.batch_size), name='logit_length_input')

			else:
				self.is_training = False
				self.keep_dropout = False
				self.image_input = tf.placeholder(tf.float32,
				                                  shape=(self.batch_size_inference, self.image_height, self.image_width,
				                                         self.image_channel), name='image_input')
			self.rnn_pred, self.rnn_logits = self.build_model()
			# squence_length = self.image_width // 4 * np.ones(1)
			self.crnn_out = tf.nn.ctc_greedy_decoder(self.rnn_logits, sequence_length=25 * np.ones(1), merge_repeated=True)[0][0]

	def build_model(self):

		backbone_output = shadow_net(self.image_input, self.is_training, self.bn_momentum, 'ShadowNet')
		rnn_pred, rnn_logits = rnn_head(backbone_output, self.hidden_units, self.hidden_layers, self.class_num,
		                                'RnnHead')

		return rnn_pred, rnn_logits


if __name__ == "__main__":
	sess = tf.Session()
	model = ModelCrnn(sess, param)

	print('123')

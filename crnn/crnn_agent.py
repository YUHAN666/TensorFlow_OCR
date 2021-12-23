from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from base.agent import Agent
from crnn.crnn_datamanager import DataManagerCrnn
from crnn.crnn_model import ModelCrnn
from crnn.crnn_trainer import TrainerCrnn
from saver import Saver


class AgentCrnn(Agent):

	def __init__(self, param, logger):

		self.logger = logger
		logger.info("Start initializing AgentCrnn, mode is {}".format(param["mode"]))
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.param = param
		self.session = tf.Session(config=config)
		self.model = ModelCrnn(self.session, self.param)  # 建立将模型graph并写入session中
		# self.model.print_nodes(remove_training_nodes=True)
		# self.data_manager = DataManagerCrnn(self.param)  # 数据生成器
		# self.trainer = TrainerCrnn(self.session, self.model, self.param, self.logger)  # 损失函数，优化器，以及训练策略
		self.saver = Saver(self.session, self.param, self.model.checkPoint_dir,
		                   self.logger, self.model)  # 用于将session保存到checkpoint或pb文件中
		logger.info("Successfully initialized")

	def run(self):

		if not self.param["anew"]:
			self.saver.load_checkpoint()
		if self.param["mode"] == "train":  # 训练模型分割部分
			self.trainer.train(self.data_manager, self.saver)
		elif self.param["mode"] == "savePb":  # 保存模型到pb文件
			self.saver.save_pb()
		elif self.param["mode"] == "test":
			self.test_inference()

	def test_inference(self):

		sess = self.session
		# image_root = 'E:/CODES/TensorFlow_OCR/dataset/left_number2/test_images/'
		# image_paths = [i[2] for i in os.walk(image_root)][0]
		with sess.as_default():
			crnn_out = tf.nn.ctc_greedy_decoder(self.model.rnn_logits, 25 * np.ones(1), merge_repeated=True)
		false_account = 0
		for i in range(self.data_manager.data_num_test // self.data_manager.batch_size_inference):
			# image = cv2.imread(os.path.join(image_root, i), 1)
			with sess.as_default():
				image, label_string, image_path = sess.run(self.data_manager.next_batch_test)
				start = timer()
				decode, _ = sess.run(crnn_out, feed_dict={self.model.image_input: image})
				end = timer()
				string = ''
				for j in decode[0].values:
					if j == 10:
						string += '0'
					elif j == 11:
						string += '-'
					elif j == 0:
						continue
					else:
						string += str(j)
				if bytes.decode(label_string[0]) != string:
					print('{}:{}    {}'.format(bytes.decode(label_string[0]), string, bytes.decode(image_path[0])))
					false_account += 1
				print('{}s'.format(end - start))
		print(false_account)

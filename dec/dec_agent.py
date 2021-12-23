import numpy as np
import tensorflow as tf

from base.agent import Agent
from dec.dec_data_manager import DataManagerDec
from dec.dec_model import ModelDec
from dec.dec_trainer import TrainerDec
from saver import Saver


class AgentDec(Agent):

	def __init__(self, param, logger):

		self.logger = logger
		logger.info("Start initializing AgentDec, mode is {}".format(param["mode"]))
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.param = param
		self.session = tf.Session(config=config)
		self.model = ModelDec(self.session, self.param)    # 建立将模型graph并写入session中
		# self.model.print_nodes(remove_training_nodes=True)
		self.data_manager = DataManagerDec(self.param)     # 数据生成器
		self.trainer = TrainerDec(self.session, self.model, self.param, self.logger)    # 损失函数，优化器，以及训练策略
		self.saver = Saver(self.session, self.param, self.model.checkPoint_dir, self.logger,
		                   self.model)  # 用于将session保存到checkpoint或pb文件中
		if self.param["mode"] == "simple_save":
			self.saver.load_checkpoint()
			self.simple_save()
			return

		logger.info("Successfully initialized")

	def run(self):

		if not self.param["anew"] and self.param["mode"] != "simple_save":
			self.saver.load_checkpoint()
		if self.param["mode"] == "train_dec":      # 训练模型分割部分
			self.trainer.train(self.data_manager, self.saver)
		elif self.param["mode"] == "savePb":                # 保存模型到pb文件
			self.saver.save_pb()
		elif self.param["mode"] == "test":
			self.test()


	def test(self):

		dm = self.data_manager
		sess = self.session
		count = 0
		for i in range(dm.data_num_test):

			image_batch, label_batch, path_batch = sess.run(dm.next_batch_test)

			dec_out, prob = sess.run([self.model.decision_out, self.model.prob], feed_dict={self.model.image_input: image_batch})

			if dec_out[0] != np.argmax(label_batch):
				count += 1
				print(path_batch[0])
				print(dec_out)
				print(prob)
		print('False count: {}'.format(count))

	def simple_save(self):

		tf.saved_model.simple_save(self.session, './chip_pbmodel_dec', inputs={'my_input': self.model.image_input},
		                           outputs={'my_output': self.model.decision_out})  #dbnet/proba3_sigmoid

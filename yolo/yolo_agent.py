import os
import xml.dom.minidom

import numpy as np
import tensorflow as tf

from base.agent import Agent
from saver import Saver
from yolo.yolo_config import LABEL_DICT
from yolo.yolo_data_manager import YOLODataManager
from yolo.yolo_inference import YoloInference
from yolo.yolo_kmeans import YOLO_Kmeans
from yolo.yolo_model import ModelYOLO
from yolo.yolo_trainer import TrainerYOLO


class AgentYOLO(Agent):

	def __init__(self, param, logger):

		self.logger = logger
		logger.info("Start initializing AgentYOLO, mode is {}".format(param["mode"]))
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.param = param
		self.data_dir = param['data_dir']
		self.xml_root = os.path.join(self.data_dir, 'annotations')
		self.image_root = os.path.join(self.data_dir, 'JPEGImages')
		self.anchors_path = param['anchor_path']
		self.txt_path = param['txt_path']
		self.label_dict = LABEL_DICT
		self.clustter_num = param["clustter_num"]
		self.session = tf.Session(config=config)
		if not os.path.exists(self.param["txt_path"]):
			self.xml_paths = self.get_file_list()
			self.write_xml_to_txt()
		if not os.path.exists(self.anchors_path):
			self.kmeans_calculate_anchor()
		self.anchors = self.get_anchors()
		self.num_anchors = len(self.anchors)
		self.model = ModelYOLO(self.session, self.param, self.anchors, self.logger)    # 建立将模型graph并写入session中
		self.data_manager = YOLODataManager(self.param, self.anchors)
		self.trainer = TrainerYOLO(self.session, self.model, self.param, self.anchors, self.logger)    # 损失函数，优化器，以及训练策略
		self.inferencer = YoloInference(self.param, self.session, self.model, self.anchors, self.data_manager)
		self.saver = Saver(self.session, self.param, self.model.checkPoint_dir, self.logger)     # 用于将session保存到checkpoint或pb文件中

		logger.info("Successfully initialized")

	def run(self):

		if not self.param["anew"] and self.param["mode"] != "testPb":
			self.saver.load_checkpoint()
		if self.param["mode"] == "train_yolo":      # 训练模型分割部分
			if len(self.anchors) == 9:
				self.trainer.train(self.data_manager, self.saver)
			else:
				self.trainer.train_tiny(self.data_manager, self.saver)
		elif self.param["mode"] == "savePb":                # 保存模型到pb文件
			self.saver.save_pb()
		elif self.param["mode"] == "inference":
			# self.inferencer = YoloInference(self.param, self.session, self.model, self.anchors)
			# self.inferencer.inference_with_train_data()
			self.inferencer.inference()

		elif self.param["mode"] == "inspect_checkpoint":
			self.saver.inspect_checkpoint()

	def get_anchors(self):
		''' loads the anchors from a file '''
		with open(self.param["anchor_path"]) as f:
			anchors = f.readline()
		anchors = [float(x) for x in anchors.split(',')]
		return np.array(anchors).reshape(-1, 2)

	def kmeans_calculate_anchor(self):
		kmeans = YOLO_Kmeans(self.clustter_num, self.txt_path, self.anchors_path)
		kmeans.txt2clusters()

	def write_xml_to_txt(self):

		with open(self.txt_path, 'w') as f:
			for path in self.xml_paths:

				xml_path = os.path.join(self.xml_root, path)
				xml_data, image_name = self.read_xml(xml_path)
				f.write(os.path.join(self.image_root, image_name))
				for i in range(len(xml_data)):
					xmin, ymin, xmax, ymax, label = xml_data[i]
					f.write(' {},{},{},{},{}'.format(xmin, ymin, xmax, ymax, label))
				f.write('\n')

	def read_xml(self, xml_path):

		output = []
		document_tree = xml.dom.minidom.parse(xml_path)
		collection = document_tree.documentElement
		num_object = len(collection.getElementsByTagName('object'))
		image_name = collection.getElementsByTagName('filename')[0].childNodes[0].data
		for i in range(num_object):
			object_name = collection.getElementsByTagName('name')[i].childNodes[0].data
			xmin = collection.getElementsByTagName('xmin')[i].childNodes[0].data
			xmax = collection.getElementsByTagName('xmax')[i].childNodes[0].data
			ymin = collection.getElementsByTagName('ymin')[i].childNodes[0].data
			ymax = collection.getElementsByTagName('ymax')[i].childNodes[0].data
			output.append((int(xmin), int(ymin), int(xmax), int(ymax), self.label_dict[object_name]))

		return output, image_name

	def get_file_list(self):

		return [i[2] for i in os.walk(self.xml_root)][0]



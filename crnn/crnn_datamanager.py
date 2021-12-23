import os

import cv2
import numpy as np
import tensorflow as tf

from crnn.crnn_config import DefaultParam as param, LABEL_DICT


class DataManagerCrnn(object):

	def __init__(self, param):

		self.data_list_path = param["data_list_path"]
		self.train_list_path = param["train_list_path"]
		self.test_list_path = param["test_list_path"]
		self.image_height = param["image_height"]
		self.image_width = param["image_width"]
		self.batch_size = param["batch_size"]
		self.batch_size_inference = param["batch_size_inference"]
		self.max_text_len = param["max_text_len"]
		self.class_num = param["class_num"]
		self.logit_length = self.image_width // 4

		if not os.path.exists(self.train_list_path):
			self.data_list = self.get_file_list(self.data_list_path)
			self.train_list, self.test_list = self.split_data()
			with open(self.train_list_path, 'w') as f:
				f.writelines(self.train_list)
			with open(self.test_list_path, 'w') as f:
				f.writelines(self.test_list)
		else:
			self.train_list = self.get_file_list(self.train_list_path)
			self.test_list = self.get_file_list(self.test_list_path)

		self.data_num_train = len(self.train_list)
		self.num_batch_train = self.data_num_train // self.batch_size
		self.dataset_train = tf.data.Dataset.from_generator(self.generator_train,
		                                                    (tf.float32, tf.int32, tf.int32, tf.int32))
		self.dataset_train = self.dataset_train.batch(self.batch_size).repeat()
		self.iterator_train = self.dataset_train.make_one_shot_iterator()
		self.next_batch_train = self.iterator_train.get_next()

		self.data_num_test = len(self.test_list)
		self.num_batch_train = self.data_num_train // self.batch_size
		self.dataset_test = tf.data.Dataset.from_generator(self.generator_test, (tf.float32, tf.string, tf.string))
		self.dataset_test = self.dataset_test.batch(self.batch_size_inference).repeat()
		self.iterator_test = self.dataset_test.make_one_shot_iterator()
		self.next_batch_test = self.iterator_test.get_next()

	def generator_train(self):

		current_index = 0
		index = np.arange(self.data_num_train)
		np.random.shuffle(index)
		while True:
			if current_index >= self.data_num_train:
				np.random.shuffle(index)
				current_index = 0

			image_path, label_string = self.train_list[index[current_index]].split(' ')
			image = cv2.imread(image_path, 1)
			image = image / 255.0
			image = cv2.resize(image, (self.image_width, self.image_height))

			label = np.zeros((self.max_text_len))
			label_length = np.array([len(label_string)])
			logit_length = np.array([self.logit_length])
			for i in range(len(label_string)):
				label[i] = LABEL_DICT[label_string[i]]
			current_index += 1

			# if np.random.uniform() > 0.7:
			#     image_augment = iaa.Sequential([iaa.GammaContrast((0.8, 1.2)),
			#                                 iaa.Affine(scale=(0.6, 1.3), rotate=(-1, 1)),
			#                                 # iaa.Rotate(rotate=(-180, 180)),
			#                                 # iaa.MotionBlur(k=(3, 7), angle=90, direction=(0, 0)),
			#                                 # iaa.MotionBlur(k=(3, 7), angle=0, direction=(0, 0)),
			#                                 iaa.Crop(percent=(0, 0.2))], random_order=True)
			#     image = image_augment.augment_image(image)
			#
			#
			# print(label)
			# print(label_length)
			# cv2.imshow('image', image)
			# cv2.waitKey()
			# cv2.destroyAllWindows()

			yield image, label, label_length, logit_length

	def generator_test(self):

		for i in self.test_list:
			image_path, label_string = i.split(' ')
			image = cv2.imread(image_path, 1)
			image = image / 255.0
			image = cv2.resize(image, (self.image_width, self.image_height))

			label = np.zeros((self.max_text_len))
			for i in range(len(label_string)):
				label[i] = LABEL_DICT[label_string[i]]

			yield image, label_string, image_path

	def get_file_list(self, path):

		with open(path, 'r') as f:
			data_list = f.read().splitlines(keepends=False)
		return data_list

	def scalar2onehot(self, number):
		label = np.zeros((self.class_num,))
		label[number] = 1

		return label

	def split_data(self, split=0.2):
		length = len(self.data_list)
		index = np.arange(start=0, stop=length, dtype=np.int32)
		np.random.shuffle(index)
		data_list = np.array([i + '\n' for i in self.data_list])

		return data_list[index][0:int((1 - split) * length)], data_list[index][int((1 - split) * length):]


if __name__ == "__main__":
	dm = DataManagerCrnn(param)

	session = tf.Session()
	data = session.run(dm.next_batch_train)

	print(data)

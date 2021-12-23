import os

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from PIL import Image

from base.data_manager import DataManager
from yolo.yolo_config import DefaultParam as param
from yolo.yolo_config import LABEL_DICT


class YOLODataManager(DataManager):

	def __init__(self, param, anchors):

		self.data_dir = param['data_dir']
		self.xml_root = os.path.join(self.data_dir, 'annotations')
		self.image_root = os.path.join(self.data_dir, 'JPEGImages')
		self.anchors_path = param['anchor_path']
		self.txt_path = param['txt_path']
		self.label_dict = LABEL_DICT
		self.anchors = anchors
		self.num_anchors = len(self.anchors)
		self.batch_size = param['batch_size'] if param['mode'] == 'train_yolo' else param["batch_size_inference"]
		self.input_shape = param['input_shape']
		self.xml_paths = self.get_file_list()
		self.rotate_angle = param['augmentation_method']['Rotate']
		self.gaussian_sigma = param['augmentation_method']['GaussianBlur']
		self.resize_scale = param['augmentation_method']['Resize']
		self.down_scale = param['down_scale']

		self.class_num = len(self.label_dict.keys())

		self.feature_num = len(self.anchors) // 3

		if param["mosaic"]:
			self.data_generator = tf.data.Dataset.from_generator(self.generator_train_mosaic, (
				tf.float32, tf.float32, tf.float32,
				tf.float32)) if self.feature_num == 3 else tf.data.Dataset.from_generator(
				self.generator_train_mosaic, (tf.float32, tf.float32, tf.float32))
		else:
			self.data_generator = tf.data.Dataset.from_generator(self.generator_train, (
				tf.float32, tf.float32, tf.float32,
				tf.float32)) if self.feature_num == 3 else tf.data.Dataset.from_generator(
				self.generator_train, (tf.float32, tf.float32, tf.float32))

		self.num_batch_train = len(self.xml_paths) // self.batch_size
		self.iterator = self.data_generator.make_one_shot_iterator()
		self.get_next = self.iterator.get_next()

	def generator_train(self):
		"""
		:yield y_true[l]: [b, grid_x, grid_y, 3, 5+c]
		"""
		with open(self.txt_path, 'r') as f:
			annotations = f.read().splitlines(keepends=False)
		# ann = [i for i in f.readlines()]

		batch_num = len(annotations) // self.batch_size

		current_idx = 0
		while True:
			if current_idx + 1 > batch_num:
				np.random.shuffle(annotations)
				current_idx = 0
			annotation = annotations[current_idx * self.batch_size:(current_idx + 1) * self.batch_size]
			image_list = []
			box_list = []
			for b in range(self.batch_size):
				# image_path = annotation[b].split(' ')[0]
				# image = cv2.imread(os.path.join(self.image_root, image_path))
				# box_data = np.array([np.array(list(map(int, box.split(',')))) for box in annotation[b].split(' ')[1:]])
				image, box_data = self.get_random_data(annotation[b], self.input_shape, random=False,
				                                       jitter=self.resize_scale)

				# use imgaug to blur and rotate image
				augmenters = iaa.Sequential([iaa.Affine(rotate=self.rotate_angle), iaa.GammaContrast(),
				                             iaa.GaussianBlur(sigma=self.gaussian_sigma),
				                             iaa.Crop(percent=((0, 0.2), (0, 0.1), (0, 0.2), (0, 0.1))),
				                             iaa.Fliplr(0.5), iaa.Flipud(0.5)], random_order=False)
				# augmenters = iaa.Sequential([])
				augmenters = augmenters.to_deterministic()
				image_auged = augmenters.augment_image(image)

				for i in range(box_data.shape[0]):
					if np.max(box_data[i]) == 0:
						break
					key_point = [ia.Keypoint(x=box_data[i][0], y=box_data[i][1]),
					             ia.Keypoint(x=box_data[i][2], y=box_data[i][1]),
					             ia.Keypoint(x=box_data[i][2], y=box_data[i][3]),
					             ia.Keypoint(x=box_data[i][0], y=box_data[i][3])]
					key_point_on_image = ia.KeypointsOnImage(key_point, shape=image.shape)
					point_auged = augmenters.augment_keypoints(key_point_on_image)

					points_x = np.array([item.x_int for item in point_auged.items])
					points_y = np.array([item.y_int for item in point_auged.items])
					# points_x = points_x[points_x < image.shape[1]][0 < points_x]
					points_x = np.where(points_x < image_auged.shape[1], points_x, image_auged.shape[1]-1)
					points_x = np.where(points_x > 0, points_x, 0)
					# points_y = points_y[points_y < image.shape[0]][0 < points_y]
					points_y = np.where(points_y < image_auged.shape[0], points_y, image_auged.shape[0]-1)
					points_y = np.where(points_y > 0, points_y, 0)

					xmin = np.min(points_x)
					xmax = np.max(points_x)
					ymin = np.min(points_y)
					ymax = np.max(points_y)

					# new_key_points = [ia.Keypoint(x=xmin, y=ymin),
					#                   ia.Keypoint(x=xmax, y=ymax)]
					# new_key_point_on_image = ia.KeypointsOnImage(new_key_points, shape=image.shape)
					#
					# ia.imshow(
					# 	np.hstack([
					# 		# key_point_on_image.draw_on_image(image, size=7),
					# 		point_auged.draw_on_image(image_auged, size=7),
					# 		new_key_point_on_image.draw_on_image(image_auged, size=7)
					# 	])
					# )

					box_data[i][0] = float(xmin)
					box_data[i][1] = float(ymin)
					box_data[i][2] = float(xmax)
					box_data[i][3] = float(ymax)

				image_list.append(image_auged)
				box_list.append(box_data)

			# img = image_auged.copy()
				# for j in  range(box_data.shape[0]):
				# 	xmin = int(box_data[j][0])
				# 	ymin = int(box_data[j][1])
				# 	xmax = int(box_data[j][2])
				# 	ymax = int(box_data[j][3])
				# 	cv2.rectangle(img, (xmin, ymin), (xmax,ymax), (0,255,0),2)
			# cv2.imshow('img', image)
			# cv2.waitKey()
				# cv2.imshow('img', img)
				# cv2.waitKey()
			# cv2.destroyAllWindows()

			if self.num_anchors not in [6, 9]:
				batch_box = self.preprocess_true_boxes_customized(np.array(box_list))
			else:
				batch_box = self.preprocess_true_boxes(np.array(box_list))
			# print(np.max(batch_box[0]))
			# print(np.max(batch_box[1]))
			batch_image = np.array(image_list)
			current_idx += 1
			#
			yield (batch_image, batch_box[0], batch_box[1], batch_box[2]) if self.feature_num == 3 else (
				batch_image, batch_box[0], batch_box[1])

	def generator_train_mosaic(self):
		"""
		implemented mosaic augmentation
		:yield y_true[l]: [b, grid_x, grid_y, 3, 5+c]

		"""
		with open(self.txt_path, 'r') as f:
			annotations = f.read().splitlines(keepends=False)
		# ann = [i for i in f.readlines()]

		batch_num = len(annotations) // (self.batch_size * 4)
		np.random.shuffle(annotations) 

		current_idx = 0
		while True:
			if current_idx + 1 > batch_num:
				np.random.shuffle(annotations)
				current_idx = 0

			image_list = []
			box_list = []
			for b in range(self.batch_size):
				annotation = annotations[(current_idx*self.batch_size*4 + b * 4): (current_idx*self.batch_size*4 + (b+1)*4)]

				image, box_data = self.get_random_data_mosaic(annotation, self.input_shape, jitter=self.resize_scale)

				image_list.append(image)
				box_list.append(box_data)

				# img = image.copy()
				# for j in  range(box_data.shape[0]):
				# 	xmin = int(box_data[j][0])
				# 	ymin = int(box_data[j][1])
				# 	xmax = int(box_data[j][2])
				# 	ymax = int(box_data[j][3])
				# 	cv2.rectangle(img, (xmin, ymin), (xmax,ymax), (0,255,0),2)
				# cv2.imshow('img', img)
				# cv2.waitKey()
				# cv2.destroyAllWindows()

			batch_box = self.preprocess_true_boxes(np.array(box_list))
			# print(np.max(batch_box[0]))
			# print(np.max(batch_box[1]))
			batch_image = np.array(image_list)
			current_idx += 1

			yield (batch_image, batch_box[0], batch_box[1], batch_box[2]) if self.feature_num == 3 else (
				batch_image, batch_box[0], batch_box[1])
	def preprocess_true_boxes(self, true_boxes):
		'''Preprocess true boxes to training input format

		Parameters
		----------
		true_boxes: array, shape=(m, T, 5)
			Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
		input_shape: array-like, hw, multiples of 32
		anchors: array, shape=(N, 2), wh  N=9
		num_classes: integer

		Returns
		-------
		y_true: list of array, shape like yolo_outputs, xywh are relative value
				len(y_true) = num_layers
				y_true[L] = [b, grid_shapes[L][0], grid_shapes[L][1], 3, 5+c]
				L: feature map layer
				grid_shapes[L][0]: width of feature map L
				grid_shapes[L][1]: height of feature map L
				3: number of anchor per layertrue_boxes
				5: x_center, y_center, w, h, prob of true_box， 有几个true_box就有几个点的prob和坐标不为0，其余均为0
				c: number of classes
				有几个gt_box就有几个true_box,

		y_true 是一个有3个元素的列表，每个元素对应一个尺度的feature map，因为yolo_body输出有三个不同尺度的feature map
		y_true[l] 对应第l个feature map，是一个 [m, grid_shape[l][0], grid_shape[l][1], 3, 5+c]的矩阵，
				  其中 m 为 batch_size；grid_shape是该层feature_map的尺寸；3对应3种大小的anchor；
				  5对应gt_box的x_center,y_center,w,h,prob；c对应类别数
		共3个尺度的feature map，每种feature map有三个大小的anchor预测层，所以对应9种大小的gt_box预测
		anchor的作用是让batch中每个gt_box(即b)找到y_true上9种尺度的预测层中与其大小最相似的一层(即l,k)，
		然后在该层上按缩放比例找到其x_center,y_center（即i,j），并将其的y_true[l][b,j,i,k]的[0:5+c]设置成gt_box的x_center,y_center,w,h,prob；c对应类别数
		注意：整个y_true上只有gt_box对应找到的点有值，其余地方均为0，有几个gt_box则几个点不为0

		'''
		anchors = self.anchors
		input_shape = self.input_shape
		assert (true_boxes[..., 4] < self.class_num).all(), 'class id must be less than num_classes'
		num_layers = len(anchors) // 3  # default setting
		anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]
		# 用于标明哪个anchor用于那一层feature_map，如标号为[6,7,8]的anchor用于第一层feature_map（input_shape//32）
		# 大anchor用于下采样大(size小)的feature_map

		true_boxes = np.array(true_boxes, dtype='float32')
		input_shape = np.array(input_shape, dtype='int32')
		boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # center of true box
		boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # w,h of true box
		true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # 转换为比例，方便后续在feature map上找对应点
		true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

		m = true_boxes.shape[0]  # batch number
		grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in
		               range(num_layers)]  # [input_shape//32, input_shape//16, input_shape//8]
		y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.class_num),
		                   dtype='float32') for l in range(num_layers)]

		# Expand dim to apply broadcasting.
		anchors = np.expand_dims(anchors, 0)
		anchor_maxes = anchors / 2.  # anchor 右上坐标
		anchor_mins = -anchor_maxes  # anchor 左下点坐标
		valid_mask = boxes_wh[..., 0] > 0  # 长宽非零的gt_box(之前用0填充置max_box个gt_box)

		for b in range(m):
			# Discard zero rows.
			wh = boxes_wh[b, valid_mask[b]]
			if len(wh) == 0:
				continue
			# Expand dim to apply broadcasting.
			wh = np.expand_dims(wh, -2)
			box_maxes = wh / 2.  # gt_box 右上点坐标
			box_mins = -box_maxes  # gt_box 左下点坐标

			intersect_mins = np.maximum(box_mins, anchor_mins)
			intersect_maxes = np.minimum(box_maxes, anchor_maxes)
			intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
			intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
			box_area = wh[..., 0] * wh[..., 1]
			anchor_area = anchors[..., 0] * anchors[..., 1]
			iou = intersect_area / (box_area + anchor_area - intersect_area)  # (T, N)
			# (T, N) T:box个数；N:anchor个数

			# Find best anchor for each true box
			best_anchor = np.argmax(iou, axis=-1)  # 每个gt_box对应的最大IOU anchor的编号（共9个anchor） (T,1)

			for t, n in enumerate(best_anchor):  # t：true_box index；n：与第t个true_box对应的iou最大的anchor index
				for l in range(num_layers):  # 最佳anchor所在层数
					if n in anchor_mask[l]:
						i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype(
							'int32')  # 最佳anchor在所在featuremap层的横坐标
						j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype(
							'int32')  # 最佳anchor在所在featuremap层的纵坐标
						k = anchor_mask[l].index(n)
						c = true_boxes[b, t, 4].astype('int32')  # 类别
						y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
						y_true[l][b, j, i, k, 4] = 1
						y_true[l][b, j, i, k, 5 + c] = 1

		return y_true

	def preprocess_true_boxes_customized(self, true_boxes):
		anchors = self.anchors
		input_shape = self.input_shape
		assert (true_boxes[..., 4] < self.class_num).all(), 'class id must be less than num_classes'
		num_layers = 2
		anchor_mask = [[2, 3], [0, 1]]
		# 用于标明哪个anchor用于那一层feature_map，如标号为[6,7,8]的anchor用于第一层feature_map（input_shape//32）
		# 大anchor用于下采样大(size小)的feature_map

		true_boxes = np.array(true_boxes, dtype='float32')
		input_shape = np.array(input_shape, dtype='int32')
		boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # center of true box
		boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # w,h of true box
		true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # 转换为比例，方便后续在feature map上找对应点
		true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

		m = true_boxes.shape[0]  # batch number
		grid_shapes = [input_shape // {0: self.down_scale, 1: self.down_scale // 2, 2: self.down_scale // 4}[l] for l in
		               range(num_layers)]  # [input_shape//32, input_shape//16, input_shape//8]
		y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.class_num),
		                   dtype='float32') for l in range(num_layers)]

		# Expand dim to apply broadcasting.
		anchors = np.expand_dims(anchors, 0)
		anchor_maxes = anchors / 2.  # anchor 右上坐标
		anchor_mins = -anchor_maxes  # anchor 左下点坐标
		valid_mask = boxes_wh[..., 0] > 0  # 长宽非零的gt_box(之前用0填充置max_box个gt_box)

		for b in range(m):
			# Discard zero rows.
			wh = boxes_wh[b, valid_mask[b]]
			if len(wh) == 0:
				continue
			# Expand dim to apply broadcasting.
			wh = np.expand_dims(wh, -2)
			box_maxes = wh / 2.  # gt_box 右上点坐标
			box_mins = -box_maxes  # gt_box 左下点坐标

			intersect_mins = np.maximum(box_mins, anchor_mins)
			intersect_maxes = np.minimum(box_maxes, anchor_maxes)
			intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
			intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
			box_area = wh[..., 0] * wh[..., 1]
			anchor_area = anchors[..., 0] * anchors[..., 1]
			iou = intersect_area / (box_area + anchor_area - intersect_area)  # (T, N)
			# (T, N) T:box个数；N:anchor个数

			# Find best anchor for each true box
			best_anchor = np.argmax(iou, axis=-1)  # 每个gt_box对应的最大IOU anchor的编号（共9个anchor） (T,1)

			for t, n in enumerate(best_anchor):  # t：true_box index；n：与第t个true_box对应的iou最大的anchor index
				for l in range(num_layers):  # 最佳anchor所在层数
					if n in anchor_mask[l]:
						i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype(
							'int32')  # 最佳anchor在所在featuremap层的横坐标
						j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype(
							'int32')  # 最佳anchor在所在featuremap层的纵坐标
						k = anchor_mask[l].index(n)
						c = true_boxes[b, t, 4].astype('int32')  # 类别
						y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
						y_true[l][b, j, i, k, 4] = 1
						y_true[l][b, j, i, k, 5 + c] = 1

		return y_true

	def gt_anchor_iou(self, gt_box, anchor):

		gt_max_point = gt_box // 2
		gt_min_point = -gt_box // 2
		anchor_max_point = anchor // 2
		anchor_min_point = -anchor // 2

		gt_max_point = np.expand_dims(gt_max_point, -2)
		gt_min_point = np.expand_dims(gt_min_point, -2)
		anchor_max_point = np.expand_dims(anchor_max_point, 0)
		anchor_min_point = np.expand_dims(anchor_min_point, 0)

		iou_box_min_point = np.maximum(gt_min_point, anchor_min_point)
		iou_box_max_point = np.minimum(gt_max_point, anchor_max_point)
		iou_wh = np.maximum(iou_box_max_point - iou_box_min_point, 0)
		iou = iou_wh[..., 0] * iou_wh[..., 1]

		iou_index = np.argmax(iou, axis=1)

		return iou_index

	def get_file_list(self):

		return [i[2] for i in os.walk(self.xml_root)][0]

	def get_random_data(self, annotation_line, input_shape, random=True, max_boxes=20,
	                    jitter=.2, hue=.1, sat=1.5, val=1.5, proc_img=True):
		""" 图像增强
		if ramdom == True将原图随机resize，并随机粘贴到input_shape大小的zero_pad上，既resize又crop，并添加了hsv增强和filp
		:param annotation_line: [image_path xmin, ymin, xmax, ymax, class ...]
		:param input_shape:     input_shape of model
		:param random:          if need augment
		:param max_boxes:       max number of boxes to keep, if not enough, then pad with zeros
		:param jitter:          缩放比例
		:param hue:             HSV增强
		:param sat:             HSV增强
		:param val:             HSV增强
		:param proc_img:        random为False时是否需要resize图片
		:return:
		"""

		def rand(a=0., b=1.):
			return np.random.rand() * (b - a) + a

		line = annotation_line.split()
		image = Image.open(line[0])
		iw, ih = image.size
		h, w = input_shape
		box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

		if not random:
			# resize image
			scale = min(w / iw, h / ih)
			nw = int(iw * scale)
			nh = int(ih * scale)
			dx = (w - nw) // 2
			dy = (h - nh) // 2
			image_data = 0
			if proc_img:
				image = image.resize((nw, nh), Image.BICUBIC)
				new_image = Image.new('RGB', (w, h), (128, 128, 128))
				new_image.paste(image, (dx, dy))
				image_data = np.array(new_image) / 255.

			# correct boxes
			box_data = np.zeros((max_boxes, 5))
			if len(box) > 0:
				np.random.shuffle(box)
				if len(box) > max_boxes: box = box[:max_boxes]
				box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
				box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
				box_data[:len(box)] = box

			return image_data, box_data

		# resize image
		new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
		scale = rand(.5, 2)
		if new_ar < 1:
			nh = int(scale * h)
			nw = int(nh * new_ar)
		else:
			nw = int(scale * w)
			nh = int(nw / new_ar)
		image = image.resize((nw, nh), Image.BICUBIC)

		# place image
		dx = int(rand(0, w - nw))
		dy = int(rand(0, h - nh))
		new_image = Image.new('RGB', (w, h), (128, 128, 128))
		new_image.paste(image, (dx, dy))
		image = new_image

		# flip image or not
		flip = rand() < .5
		if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

		# # distort image hsv
		# hue = rand(-hue, hue)
		# sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
		# val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
		# x = rgb_to_hsv(np.array(image) / 255.)
		# x[..., 0] += hue
		# x[..., 0][x[..., 0] > 1] -= 1
		# x[..., 0][x[..., 0] < 0] += 1
		# x[..., 1] *= sat
		# x[..., 2] *= val
		# x[x > 1] = 1
		# x[x < 0] = 0
		# image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
		image_data = np.array(image) / 255.

		# correct boxes
		box_data = np.zeros((max_boxes, 5))
		if len(box) > 0:
			np.random.shuffle(box)
			old_w = (box[:, 2] - box[:, 0]) * nw / iw
			old_h = (box[:, 3] - box[:, 1]) * nh / ih
			box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
			box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
			if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
			box[:, 0:2][box[:, 0:2] < 0] = 0
			box[:, 2][box[:, 2] > w] = w
			box[:, 3][box[:, 3] > h] = h
			box_w = box[:, 2] - box[:, 0]
			box_h = box[:, 3] - box[:, 1]
			# box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
			box = box[np.logical_and(box_w > old_w * 1 // 5, box_h > old_h * 1 // 3)]  # discard invalid box
			if len(box) > max_boxes: box = box[:max_boxes]
			box_data[:len(box)] = box

		return image_data, box_data


	def get_random_data_mosaic(self, annotation_lines, input_shape, max_boxes=20, jitter=.2, mosaic_scale=.3):


		h, w = input_shape
		new_image = Image.new('RGB', (w, h), (128, 128, 128))
		cutx = np.random.randint(int(w * mosaic_scale), int(w * (1 - mosaic_scale)))
		cuty = np.random.randint(int(h * mosaic_scale), int(h * (1 - mosaic_scale)))

		image0, box0 = self.get_resized_image_mosaic(annotation_lines[0], jitter, (cutx, cuty), 0, 0, max_boxes)
		image1, box1 = self.get_resized_image_mosaic(annotation_lines[1], jitter, (w-cutx, cuty), cutx, 0, max_boxes)
		image2, box2 = self.get_resized_image_mosaic(annotation_lines[2], jitter, (w-cutx, h-cuty), cutx, cuty, max_boxes)
		image3, box3 = self.get_resized_image_mosaic(annotation_lines[3], jitter, (cutx, h-cuty), 0, cuty, max_boxes)

		new_image = np.array(new_image) /255.0
		new_image[ :cuty, :cutx, ...] = image0
		new_image[ :cuty, cutx:, ...] = image1
		new_image[cuty:, cutx:, ...] = image2
		new_image[cuty:, :cutx, ...] = image3
		box = np.concatenate([box0, box1, box2, box3], axis=0)

		return new_image, box

	def get_resized_image_mosaic(self, annotation_line, jitter, output_shape, offset_x, offset_y, max_boxes):

		def rand(a=0., b=1.):
			return np.random.rand() * (b - a) + a

		w, h = output_shape
		line = annotation_line.split()
		image = Image.open(line[0])
		iw, ih = image.size

		image_array = np.array(image)
		box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
		# use imgaug to blur and rotate image
		augmenters = iaa.Sequential(
			[iaa.Affine(rotate=self.rotate_angle), iaa.GaussianBlur(sigma=self.gaussian_sigma),
			 iaa.GammaContrast()], random_order=True)
		augmenters = augmenters.to_deterministic()
		image_auged = augmenters.augment_image(image_array)

		for i in range(box.shape[0]):
			if np.max(box[i]) == 0:
				break
			key_point = [ia.Keypoint(x=box[i][0], y=box[i][1]),
			             ia.Keypoint(x=box[i][2], y=box[i][1]),
			             ia.Keypoint(x=box[i][2], y=box[i][3]),
			             ia.Keypoint(x=box[i][0], y=box[i][3])]
			key_point_on_image = ia.KeypointsOnImage(key_point, shape=image_array.shape)
			point_auged = augmenters.augment_keypoints(key_point_on_image)

			points_x = np.array([item.x_int for item in point_auged.items])
			points_y = np.array([item.y_int for item in point_auged.items])
			# points_x = points_x[points_x < image.shape[1]][0 < points_x]
			points_x = np.where(points_x < iw, points_x, iw)
			points_x = np.where(points_x > 0, points_x, 0)
			# points_y = points_y[points_y < image.shape[0]][0 < points_y]
			points_y = np.where(points_y < ih, points_y, ih)
			points_y = np.where(points_y > 0, points_y, 0)

			xmin = np.min(points_x)
			xmax = np.max(points_x)
			ymin = np.min(points_y)
			ymax = np.max(points_y)

			box[i][0] = float(xmin)
			box[i][1] = float(ymin)
			box[i][2] = float(xmax)
			box[i][3] = float(ymax)

		image = Image.fromarray(image_auged)

		# resize image
		new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
		scale = rand(.75, 2)
		if new_ar < 1:
			nh = int(scale * h)
			nw = int(nh * new_ar)
		else:
			nw = int(scale * w)
			nh = int(nw / new_ar)
		image = image.resize((nw, nh), Image.BICUBIC)

		dx = int(rand(0, w - nw))
		dy = int(rand(0, h - nh))
		new_image = Image.new('RGB', (w, h), (128, 128, 128))
		new_image.paste(image, (dx, dy))
		#
		# flip = rand() < .5
		# if flip: new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
		image_data = np.array(new_image) / 255.


		# correct boxes
		box_data = np.zeros((max_boxes, 5))
		if len(box) > 0:
			np.random.shuffle(box)
			old_w = (box[:, 2] - box[:, 0]) * nw / iw
			old_h = (box[:, 3] - box[:, 1]) * nh / ih
			box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx + offset_x
			box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy + offset_y
			# if flip: box[:, [0, 2]] = w + offset_x - box[:, [2, 0]]
			box[:, 0][box[:, 0] < offset_x] = offset_x
			box[:, 2][box[:, 2] < offset_x] = offset_x
			box[:, 2][box[:, 2] > w + offset_x] = w + offset_x
			box[:, 0][box[:, 0] > w + offset_x] = w + offset_x
			box[:, 1][box[:, 1] < offset_y] = offset_y
			box[:, 3][box[:, 3] < offset_y] = offset_y
			box[:, 1][box[:, 1] > h + offset_y] = h + offset_y
			box[:, 3][box[:, 3] > h + offset_y] = h + offset_y
			box_w = box[:, 2] - box[:, 0]
			box_h = box[:, 3] - box[:, 1]
			# box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
			box = box[np.logical_and(box_w > old_w * 1 // 5, box_h > old_h * 1 // 3)]  # discard invalid box
			if len(box) > max_boxes: box = box[:max_boxes]
			box_data[:len(box)] = box

		return image_data, box_data


if __name__ == '__main__':
	with open(param["anchor_path"]) as f:
		anchors = f.readline()
	anchors = [float(x) for x in anchors.split(',')]
	anchors =  np.array(anchors).reshape(-1, 2)

	dm = YOLODataManager(param, anchors)
	sess = tf.Session()
	with sess.as_default():
		for i in range(10000):
			image_batch,_,_,_ = sess.run(dm.get_next)
			# image = image_batch[0,...]
			# cv2.imshow('image', image)
			# cv2.waitKey()

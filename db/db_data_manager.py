import os.path as osp

import cv2
import imgaug.augmenters as iaa
import math
import numpy as np
import pyclipper
import tensorflow as tf
from shapely.geometry import Polygon

from base.data_manager import DataManager
from utiles.transform import transform, crop, resize, resize_image


# mean = [103.939, 103.939, 103.939]


def load_all_anns(gt_paths, dataset_type='total_text'):
	res = []
	for gt in gt_paths:
		lines = []
		reader = open(gt, 'r').readlines()
		for line in reader:
			item = {}
			parts = line.strip().split(',')
			label = parts[-1]
			# if label == '1':
			#     label = '###'
			line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
			if 'icdar' == dataset_type:
				poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
			else:
				num_points = math.floor((len(line) - 1) / 2) * 2
				poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
			if len(poly) < 3:
				continue
			item['poly'] = poly
			item['text'] = label
			lines.append(item)
		res.append(lines)
	return res


def show_polys(image, anns, window_name):
	for ann in anns:
		poly = np.array(ann['poly']).astype(np.int32)
		cv2.drawContours(image, np.expand_dims(poly, axis=0), -1, (0, 255, 0), 2)

	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.imshow(window_name, image)


def draw_thresh_map(polygon, canvas, mask, shrink_ratio=0.4):
	polygon = np.array(polygon)
	assert polygon.ndim == 2
	assert polygon.shape[1] == 2

	polygon_shape = Polygon(polygon)
	distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
	subject = [tuple(l) for l in polygon]
	padding = pyclipper.PyclipperOffset()
	padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
	padded_polygon = np.array(padding.Execute(distance)[0])
	cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

	xmin = padded_polygon[:, 0].min()
	xmax = padded_polygon[:, 0].max()
	ymin = padded_polygon[:, 1].min()
	ymax = padded_polygon[:, 1].max()
	width = xmax - xmin + 1
	height = ymax - ymin + 1

	polygon[:, 0] = polygon[:, 0] - xmin
	polygon[:, 1] = polygon[:, 1] - ymin

	xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
	ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

	distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
	for i in range(polygon.shape[0]):
		j = (i + 1) % polygon.shape[0]
		absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
		distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
	distance_map = np.min(distance_map, axis=0)

	xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
	xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
	ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
	ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
	canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
		1 - distance_map[ymin_valid - ymin:ymax_valid - ymin + 1,
						 xmin_valid - xmin:xmax_valid - xmin + 1],
		canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


def compute_distance(xs, ys, point_1, point_2):
	square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
	square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
	square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

	cosin = (square_distance - square_distance_1 - square_distance_2) / \
			(2 * np.sqrt(square_distance_1 * square_distance_2))
	square_sin = 1 - np.square(cosin)
	square_sin = np.nan_to_num(square_sin)
	result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

	result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
	return result


class DataManagerDB(DataManager):

	def __init__(self, param):
		self.data_dir = param['data_dir']
		self.image_size = param["image_size"]
		self.min_text_size = 1
		self.shrink_ratio = 0.3     #越接近0，gt收缩至越小
		self.thresh_min = 0.3
		self.thresh_max = 0.7
		self.batch_size = param['batch_size']
		self.batch_size_inference = param['batch_size_inference']

		dataset_train = tf.data.Dataset.from_generator(self.generator_train, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
		dataset_train = dataset_train.repeat()
		self.iterator_train = dataset_train.make_one_shot_iterator()
		self.next_batch_train = self.iterator_train.get_next()

		dataset_test = tf.data.Dataset.from_generator(self.generator_test, (tf.float32, tf.float32))
		dataset_test = dataset_test.repeat()
		self.iterator_test = dataset_test.make_one_shot_iterator()
		self.next_batch_test = self.iterator_test.get_next()
		self.train_image_paths, self.test_image_paths = self.get_file_list()
		self.num_batch_train = len(self.train_image_paths) // self.batch_size
		self.num_batch_test = len(self.test_image_paths) // self.batch_size_inference

	def generator_train(self):

		with open(osp.join(self.data_dir, 'train_list.txt')) as f:
			image_fnames = f.readlines()
			image_paths = [osp.join(self.data_dir, 'train_images', image_fname.strip()) for image_fname in image_fnames]
			gt_paths = [osp.join(self.data_dir, 'train_gts', image_fname.strip() + '.txt') for image_fname in image_fnames]
			all_anns = load_all_anns(gt_paths)

		transform_aug = iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-0, 0)), iaa.Crop(percent=((0,0.1), (0,0.1), (0,0.1), (0,0.1)), sample_independently=True)])
		dataset_size = len(image_paths)
		indices = np.arange(dataset_size)
		np.random.shuffle(indices)
		current_idx = 0
		b = 0
		while True:
			if current_idx >= dataset_size:
				np.random.shuffle(indices)
				current_idx = 0
			if b == 0:
				# Init batch arrays
				batch_images = np.zeros([self.batch_size, self.image_size, self.image_size, 3], dtype=np.float32)
				batch_gts = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
				batch_masks = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
				batch_thresh_maps = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
				batch_thresh_masks = np.zeros([self.batch_size, self.image_size, self.image_size], dtype=np.float32)
			i = indices[current_idx]
			image_path = image_paths[i]
			anns = all_anns[i]
			image = cv2.imread(image_path)
			transform_aug = transform_aug.to_deterministic()
			image, anns = transform(transform_aug, image, anns)
			image, anns = crop(image, anns)
			image, anns = resize(self.image_size, image, anns)
			anns = [ann for ann in anns if Polygon(ann['poly']).is_valid]
			gt = np.zeros((self.image_size, self.image_size), dtype=np.float32)
			mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
			thresh_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
			thresh_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
			# ori = np.zeros((self.image_size, self.image_size), dtype=np.float32)
			for ann in anns:
				poly = np.array(ann['poly'])
				height = max(poly[:, 1]) - min(poly[:, 1])
				width = max(poly[:, 0]) - min(poly[:, 0])
				polygon = Polygon(poly)
				# generate gt and mask
				# if polygon.area < 1 or min(height, width) < self.min_text_size or ann['text'] == '###':
				if polygon.area < 1 or min(height, width) < self.min_text_size:
					cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
					continue
				else:
					distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
					subject = [tuple(l) for l in ann['poly']]
					padding = pyclipper.PyclipperOffset()
					padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
					shrinked = padding.Execute(-distance)
					if len(shrinked) == 0:
						cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
						continue
					else:
						shrinked = np.array(shrinked[0]).reshape(-1, 2)
						# pl = np.array([(polygon.exterior.coords.xy[0][i], polygon.exterior.coords.xy[1][i]) for i in range(len(polygon.exterior.coords.xy[0]))]).reshape(-1,2)
						if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
							# cv2.fillPoly(ori, [pl.astype(np.int32)], 1)
							cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
						else:
							cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
							continue
				# generate thresh map and thresh mask
				draw_thresh_map(ann['poly'], thresh_map, thresh_mask, shrink_ratio=self.shrink_ratio)
			thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min

			# image =  np.where(np.tile(np.expand_dims(thresh_map, -1), [1,1,3])>0.3, 255, image)
			image = image.astype(np.float32)
			image = image / 255.0

			# t = np.expand_dims(gt, axis=-1)
			# img = np.where(t>0.3, np.ones_like(image), image)
			# # m = np.expand_dims(mask, -1)
			# # img = np.where(m == 0, np.zeros_like(img), img)
			# img = cv2.resize(img, (320,160))
			# cv2.imshow('image', img)
			# cv2.waitKey()

			batch_images[b] = image
			batch_gts[b] = gt
			batch_masks[b] = mask
			batch_thresh_maps[b] = thresh_map
			batch_thresh_masks[b] = thresh_mask

			b += 1
			current_idx += 1
			if b == self.batch_size:
				cv2.destroyAllWindows()
				yield batch_images, batch_gts, batch_masks, batch_thresh_maps, batch_thresh_masks
				b = 0

	def generator_test(self):

		with open(osp.join(self.data_dir, 'train_list.txt')) as f:
			image_fnames = f.readlines()
			image_paths = [osp.join(self.data_dir, 'train_images', image_fname.strip()) for image_fname in image_fnames]

		dataset_size = len(image_paths)
		indices = np.arange(dataset_size)
		current_idx = 0
		b = 0
		while True:
			if current_idx >= dataset_size:
				current_idx = 0
			if b == 0:
				# Init batch arrays
				batch_images = np.zeros([self.batch_size_inference, self.image_size, self.image_size, 3], dtype=np.float32)
			i = indices[current_idx]
			image_path = image_paths[i]
			image = cv2.imread(image_path)
			original_image  = image.copy()
			image = resize_image(self.image_size, image)

			image = image.astype(np.float32)
			# image[..., 0] -= mean[0]
			# image[..., 1] -= mean[1]
			# image[..., 2] -= mean[2]
			image = image / 255.0
			batch_images[b] = image
			b += 1
			current_idx += 1
			if b == self.batch_size_inference:
				yield batch_images, original_image
				b = 0

	def get_file_list(self):

		with open(osp.join(self.data_dir, 'train_list.txt')) as f:
			image_fnames = f.readlines()
			train_image_paths = [osp.join(self.data_dir, 'train_images', image_fname.strip()) for image_fname in image_fnames]
		with open(osp.join(self.data_dir, 'train_list.txt')) as f:
			image_fnames = f.readlines()
			test_image_paths = [osp.join(self.data_dir, 'train_images ', image_fname.strip()) for image_fname in image_fnames]

		return train_image_paths, test_image_paths


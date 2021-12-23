import os
from timeit import default_timer as timer

import cv2
import numpy as np
import tensorflow as tf

from yolo.nms import nms
from yolo.yolo_loss import feat2gt


class YoloInference(object):

	def __init__(self, param, sess, model, anchors, data_manager):

		self.anchors = anchors
		self.image_size = param["image_size"]
		self.image_channel = param["image_channel"]
		self.input_shape = param["input_shape"]
		self.image_shape = param["image_size"]
		self.num_classes = param["num_classes"]
		self.iou_thres = param["iou_thres"]
		self.score_thres = param["score_thres"]
		self.num_classes = param["num_classes"]
		self.max_boxes = param["max_boxes"]
		self.inference_dir = param["inference_dir"]
		self.model = model
		self.session = sess
		self.input_shape_placeholder = tf.placeholder(tf.float32, (2,), name='yolo_input_shape')
		self.image_shape_placeholder = tf.placeholder(tf.float32, (2,), name='yolo_image_shape')
		self.data_manager = data_manager
		if len(self.anchors) not in [6, 9]:
			self.boxes, self.scores, self.classes = self.yolo_eval_customized(num_classes=self.num_classes,
			                                                                  max_boxes=self.max_boxes,
			                                                                  score_threshold=self.score_thres,
			                                                                  iou_threshold=self.iou_thres)
		else:
			self.boxes, self.scores, self.classes = self.yolo_eval(num_classes=self.num_classes,
			                                                       max_boxes=self.max_boxes,
			                                                       score_threshold=self.score_thres,
			                                                       iou_threshold=self.iou_thres)

		# self.init_op = tf.global_variables_initializer()

	def inference(self):

		image_paths = [i[2] for i in os.walk(self.inference_dir)][0]
		for i in image_paths:
			with self.session.as_default():
				# self.init_op.run()

				image = cv2.imread(os.path.join(self.inference_dir, i))
				img = image.copy()
				image = self.resize_image_with_pad(image, self.input_shape)
				image = image / 255.0
				batch_image = np.expand_dims(image, axis=0)
				input_shape = np.array(self.input_shape)
				image_shape = np.array(self.image_shape)
				start = timer()
				boxes, scores, classes = self.session.run([self.boxes, self.scores, self.classes],
				                                          feed_dict={self.model.image_input: batch_image,
					                                                 self.input_shape_placeholder: input_shape,
				                                                     self.image_shape_placeholder: image_shape})
				end = timer()
				print('time: {}s'.format(end-start))
				print(boxes)
				for j in range(boxes.shape[0]):
					xmin = int(boxes[j][0])
					xmax = int(boxes[j][2])
					ymin = int(boxes[j][1])
					ymax = int(boxes[j][3])
					pred_classes = classes[j]
					if xmin < 0 or ymin < 0 or xmax > image_shape[1] or ymax > image_shape[1]:
						continue

					cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
					cv2.putText(img, str(int(scores[j][0] * 1000) / 1000.0), (xmin, ymax), 5, 1.0, (0, 0, 255))

				img = cv2.resize(img, (1024, 768))
				cv2.imshow(i, img)
				cv2.waitKey()
				cv2.destroyWindow(i)

	def resize_image_with_pad(self, image, input_shape):

		if len(input_shape) < 3:
			input_shape = (input_shape[0], input_shape[1], self.image_channel)

		scale = min(input_shape[0] / image.shape[0], input_shape[1] / image.shape[1])
		img = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale) + 1))

		pad_image = np.ones(input_shape, np.uint8)*128.0
		pad_image[(input_shape[0] - img.shape[0]) // 2:(input_shape[0] - (input_shape[0] - img.shape[0]) // 2)-1,
		(input_shape[1] - img.shape[1]) // 2:(input_shape[1] - (input_shape[1] - img.shape[1]) // 2), :] = img

		return pad_image

	def yolo_eval(self, num_classes=1, max_boxes=20, score_threshold=0.7, iou_threshold=0.45):

		"""
		:param num_classes:       BOX有多少分类
		:param max_boxes:        NMS最多取多少个BOX
		:param score_threshold:  过滤掉 prob过低的box
		:param iou_threshold:   for nms 每次删除与上次prob最大的box的iou超过iou_threshold的box
		:return:
				boxes_:   (N,4)     N:经过NMS后得到的BOX个数 4：xmin, ymin, xmax, ymax
				scores_:  (N,1)     box概率，(0,1) 之间
				classes_: (N,1)     box类别，(0,num_classes-1)中的一个整数

		"""

		anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if len(self.anchors) == 9 else [[3, 4, 5], [0, 1, 2]]
		with self.session.as_default():

			yolo_output = self.model.yolo_output
			boxes = []
			scores = []
			for l in range(len(yolo_output)):
				gt_xy, gt_wh, gt_confidence, gt_classes = feat2gt(yolo_output[l], self.anchors[anchor_mask[l]],
				                                                  self.input_shape, self.num_classes)
				# gt_confidence = tf.Print(gt_confidence, [tf.reduce_max(gt_confidence)], message="debug gt_confidence")
				# gt_classes = tf.Print(gt_classes, [tf.reduce_max(gt_classes)], message="debug gt_classes")

				box_coordinates = self.get_box_coordinates(gt_xy, gt_wh, self.input_shape_placeholder, self.image_shape_placeholder)
				box_probs = gt_confidence * gt_classes
				box_coordinates = tf.reshape(box_coordinates, (-1, 4))
				box_probs = tf.reshape(box_probs, (-1, num_classes))
				boxes.append(box_coordinates)
				scores.append(box_probs)

			boxes = tf.concat(boxes, axis=0)
			scores = tf.concat(scores, axis=0)

			def loop_cond(scores, score_threshold):
				return tf.logical_and(tf.reduce_max(scores) < score_threshold, score_threshold > 0)

			def loop_body(scores, score_threshold):

				score_threshold -= 0.1

				return scores, score_threshold

			# 確保至少有一個box
			scores, score_threshold = tf.while_loop(loop_cond, loop_body, [scores, score_threshold])
			mask = scores >= score_threshold
			max_boxes_tensor = tf.constant(max_boxes, dtype='int32')

			boxes_ = []
			scores_ = []
			classes_ = []
			for c in range(num_classes):

				# class_boxes = tf.boolean_mask(boxes, mask[:, c])
				# class_box_scores = tf.boolean_mask(scores[:, c], mask[:, c])
				# nms_index = tf.image.non_max_suppression(
				# 	class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
				# class_boxes = tf.gather(class_boxes, nms_index)
				# class_box_scores = tf.gather(class_box_scores, nms_index)
				# classes = tf.ones_like(class_box_scores, 'int32') * c
				#
				# boxes_.append(class_boxes)
				# scores_.append(class_box_scores)
				# classes_.append(classes)

				class_boxes = tf.boolean_mask(boxes, mask[:, c])
				class_box_scores = tf.boolean_mask(scores[:, c], mask[:, c])
				class_box_scores = tf.expand_dims(class_box_scores, axis=-1)
				nms_boxes, nms_scores = nms(class_boxes, class_box_scores, iou_threshold, max_boxes_tensor)
				classes = tf.ones_like(nms_scores, 'int32') * c

				boxes_.append(nms_boxes)
				scores_.append(nms_scores)
				classes_.append(classes)
			boxes_ = tf.concat(boxes_, axis=0, name='yolo_output_boxes')
			scores_ = tf.concat(scores_, axis=0, name='yolo_output_scores')
			classes_ = tf.concat(classes_, axis=0, name='yolo_output_classes')

		return boxes_, scores_, classes_

	def yolo_eval_customized(self, num_classes=1, max_boxes=20, score_threshold=0.7, iou_threshold=0.45):

		anchor_mask = [[2, 3], [0, 1]]
		with self.session.as_default():

			yolo_output = self.model.yolo_output
			boxes = []
			scores = []
			for l in range(len(yolo_output)):
				gt_xy, gt_wh, gt_confidence, gt_classes = feat2gt(yolo_output[l], self.anchors[anchor_mask[l]],
				                                                  self.input_shape, self.num_classes)
				# gt_confidence = tf.Print(gt_confidence, [tf.reduce_max(gt_confidence)], message="debug gt_confidence")
				# gt_classes = tf.Print(gt_classes, [tf.reduce_max(gt_classes)], message="debug gt_classes")

				box_coordinates = self.get_box_coordinates(gt_xy, gt_wh, self.input_shape_placeholder,
				                                           self.image_shape_placeholder)
				box_probs = gt_confidence * gt_classes
				box_coordinates = tf.reshape(box_coordinates, (-1, 4))
				box_probs = tf.reshape(box_probs, (-1, num_classes))
				boxes.append(box_coordinates)
				scores.append(box_probs)

			boxes = tf.concat(boxes, axis=0)
			scores = tf.concat(scores, axis=0)

			def loop_cond(scores, score_threshold):
				return tf.logical_and(tf.reduce_max(scores) < score_threshold, score_threshold > 0)

			def loop_body(scores, score_threshold):

				score_threshold -= 0.1

				return scores, score_threshold

			# 確保至少有一個box
			scores, score_threshold = tf.while_loop(loop_cond, loop_body, [scores, score_threshold])
			mask = scores >= score_threshold
			max_boxes_tensor = tf.constant(max_boxes, dtype='int32')

			boxes_ = []
			scores_ = []
			classes_ = []
			for c in range(num_classes):
				# class_boxes = tf.boolean_mask(boxes, mask[:, c])
				# class_boxes = tf.Print(class_boxes, [tf.shape(class_boxes)], 'debug class_boxes:', summarize=1000)
				# class_box_scores = tf.boolean_mask(scores[:, c], mask[:, c])
				# nms_index = tf.image.non_max_suppression(
				# 	class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
				# class_boxes = tf.gather(class_boxes, nms_index)
				# class_box_scores = tf.gather(class_box_scores, nms_index)
				# classes = tf.ones_like(class_box_scores, 'int32') * c
				#
				# boxes_.append(class_boxes)
				# scores_.append(class_box_scores)
				# classes_.append(classes)

				class_boxes = tf.boolean_mask(boxes, mask[:, c])
				# class_boxes = tf.Print(class_boxes, [tf.shape(class_boxes)], 'debug class_boxes:', summarize=1000)
				class_box_scores = tf.boolean_mask(scores[:, c], mask[:, c])
				class_box_scores = tf.expand_dims(class_box_scores, axis=-1)
				nms_boxes, nms_scores = nms(class_boxes, class_box_scores, iou_threshold, max_boxes)
				classes = tf.ones_like(nms_scores, 'int32') * c

				boxes_.append(nms_boxes)
				scores_.append(nms_scores)
				classes_.append(classes)
			boxes_ = tf.concat(boxes_, axis=0, name='yolo_output_boxes')
			scores_ = tf.concat(scores_, axis=0, name='yolo_output_scores')
			classes_ = tf.concat(classes_, axis=0, name='yolo_output_classes')


		return boxes_, scores_, classes_

	def get_box_coordinates(self, box_xy, box_wh, input_shape, image_shape):

		image_shape = image_shape[::-1]
		input_shape = input_shape[::-1]

		resized_scale = tf.minimum(input_shape[0] / image_shape[0], input_shape[1] / image_shape[1])
		new_shape = image_shape * resized_scale
		offset = (input_shape - new_shape) / 2

		box_xy = (box_xy * input_shape - offset) / new_shape
		box_wh = box_wh * input_shape / new_shape

		box_xmin_ymin = (box_xy - box_wh / 2) * image_shape
		box_xmax_ymax = (box_xy + box_wh / 2) * image_shape

		# box_xmin = box_xmin_ymin[...,0:1]
		# box_xmin = tf.where(box_xmin > 0, box_xmin, tf.zeros_like(box_xmin))
		# box_xmin = tf.where(box_xmin < image_shape[0], box_xmin, tf.ones_like(box_xmin)*image_shape[0])
		# box_xmax = box_xmax_ymax[...,0:1]
		# box_xmax = tf.where(box_xmax > 0, box_xmax, tf.zeros_like(box_xmax))
		# box_xmax = tf.where(box_xmax < image_shape[0], box_xmax, tf.ones_like(box_xmax)*image_shape[0])
		#
		# box_ymin = box_xmin_ymin[...,1:2]
		# box_ymin = tf.where(box_ymin > 0, box_ymin, tf.zeros_like(box_ymin))
		# box_ymin = tf.where(box_ymin < image_shape[1], box_ymin, tf.ones_like(box_ymin) * image_shape[1])
		#
		# box_ymax = box_xmax_ymax[...,1:2]
		# box_ymax = tf.where(box_ymax > 0, box_ymax, tf.zeros_like(box_ymax))
		# box_ymax = tf.where(box_ymax < image_shape[1], box_ymax, tf.ones_like(box_ymax) * image_shape[1])
		#
		# box_coordinates = tf.concat([box_xmin, box_ymin, box_xmax, box_ymax], axis=-1)
		box_coordinates = tf.concat([box_xmin_ymin,box_xmax_ymax], axis=-1)


		return box_coordinates

	def inference_with_train_data(self):

		for i in range(100):
			with self.session.as_default():
				batch_image, _, _, _ = self.session.run(self.data_manager.get_next)
				input_shape = np.array(self.input_shape)
				image_shape = np.array(self.image_shape)
				start = timer()
				boxes, scores, classes = self.session.run([self.boxes, self.scores, self.classes],
				                                          feed_dict={self.model.image_input: batch_image,
					                                                 self.input_shape_placeholder: input_shape,
				                                                     self.image_shape_placeholder: image_shape})
				end = timer()
				print('time: {}s'.format(end-start))
				print(boxes)
				img = np.squeeze(batch_image, 0)
				for j in range(boxes.shape[0]):
					xmin = int(boxes[j][0])
					xmax = int(boxes[j][2])
					ymin = int(boxes[j][1])
					ymax = int(boxes[j][3])
					pred_classes = classes[j][0]

					cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
					cv2.putText(img, str(pred_classes), (xmin, ymax), 1, 10.0, (0,0,255))

				img = cv2.resize(img, (1024, 768))
				cv2.imshow('img', img)
				cv2.waitKey()
				cv2.destroyWindow('img')

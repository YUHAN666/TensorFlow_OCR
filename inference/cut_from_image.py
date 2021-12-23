import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from utiles.box_utiles import polygons_from_bitmap
from utiles.transform import resize_image

IMAGE_SIZE = [160, 160]
ORIGINAL_IMAGE_SIZE = [1200, 1920]
pb_file_path = '../pbMode/pzt_db_model.pb'
# image_root = 'E:/CODES/TensorFlow_OCR/dataset/chip_ocr_6_2/train_images'
image_root = '../test/'

sess = tf.Session()
with gfile.FastGFile(pb_file_path, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')
sess.run(tf.global_variables_initializer())

# original_image_input = tf.placeholder(tf.float32, (1, ORIGINAL_IMAGE_SIZE[0], ORIGINAL_IMAGE_SIZE[1], 3))
decision_out = sess.graph.get_tensor_by_name('dbnet/proba3_sigmoid:0')
input_image = sess.graph.get_tensor_by_name('image_input:0')
# mask = tf.where(decision_out>0.5, tf.ones_like(decision_out), tf.zeros_like(decision_out))
# mask = tf.squeeze(mask, -1)
# grid_x = tf.tile(tf.reshape(tf.range(0, tf.shape(mask)[2]), (1, 1, -1, 1)), (1, tf.shape(mask)[1], 1, 1))
# grid_y = tf.tile(tf.reshape(tf.range(0,tf.shape(mask)[1]), (1, -1, 1, 1)), (1, 1, tf.shape(mask)[2], 1))
# grid = tf.concat([grid_x, grid_y], -1)
# coord = tf.boolean_mask(grid, mask)
# xmax = tf.reduce_max(coord[:,0], 0)
# xmin = tf.reduce_min(coord[:,0], 0)
# ymax = tf.reduce_max(coord[:,1], 0)
# ymin = tf.reduce_min(coord[:,1], 0)
# xcenter = (xmax + xmin) / 2
# ycenter = (ymax + ymin) / 2
# x_scale = ORIGINAL_IMAGE_SIZE[1] / IMAGE_SIZE[1]
# y_scale = ORIGINAL_IMAGE_SIZE[0] / IMAGE_SIZE[0]
# xc = tf.to_int32(xcenter * x_scale)
# yc = tf.to_int32(ycenter * x_scale)
# out_image = original_image_input[0, yc - 80:yc + 80, xc - 160:xc + 160, :]



image_names = [i[2] for i in os.walk(image_root)][0]
name_dic = {}
count = 0
for i in image_names:
	image_path = os.path.join(image_root, i)
	image = cv2.imread(image_path)
	img = image.copy()
	image = image.astype(np.float32)

	resized_image = resize_image(IMAGE_SIZE[0], image)
	resized_image = np.array(resized_image[np.newaxis,:, :, :])
	resized_image /= 255.0

	proba_map = sess.run(decision_out, feed_dict={input_image: resized_image})
	proba_map = np.squeeze(proba_map, axis=0)
	# proba_map = sess.run(decision_out, feed_dict={input_image: image})
	# proba_map = np.squeeze(proba_map, 0)
	# image = np.squeeze(image, 0)
	# image *= 255.0
	# img = image.copy()
	bitmap = proba_map > 0.3
	contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# boxes, scores = polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)
	resized_image = np.squeeze(resized_image, 0)
	scale = image.shape[1]/IMAGE_SIZE[0]
	xwidth = 180
	ywidth = 220
	min_distance = image.shape[1]
	cut_center_x = 0
	cut_center_y = 0
	for contour in contours:
		contour = np.reshape(contour, (-1, 2))
		# xmin = int(sorted(list(contour), key=lambda x: x[0])[0][0]*scale)
		# xmax = int(sorted(list(contour), key=lambda x: x[0])[-1][0]*scale)
		# ymin = int(sorted(list(contour), key=lambda x: x[1])[0][1]*scale)
		# ymax = int(sorted(list(contour), key=lambda x: x[1])[-1][1]*scale)
		xcenter = int(np.mean(contour, axis=0)[0]*scale)
		ycenter = int(np.mean(contour, axis=0)[1]*scale)
		# xcenter = int((xmin+xmax)//2)
		# ycenter = int((ymin+ymax)//2)
		distance = abs(xcenter - image.shape[1]//2)
		if distance < min_distance:
			cut_center_x = xcenter
			cut_center_y = ycenter
			min_distance = distance

		# xcenter = (xmin+xmax)//2
		# ycenter = (ymin+ymax)//2
		cut_image = img[cut_center_y-ywidth:cut_center_y+ywidth, cut_center_x-xwidth:cut_center_x+xwidth, :]
		# cv2.drawContours(resized_image, [np.array(contour)], -1, (0, 255, 0), 2)
		# cv2.imshow('resized_image', cut_image/255.0)
		# cv2.waitKey()
		try:
			cv2.imwrite(os.path.join('../cut/', str(count)+'.jpg'), cut_image)
			count+=1
		except:
			continue
	# # cv2.imwrite(os.path.join('./test/', i), image)
	# for contour in boxes:
	# 	contour = np.reshape(contour, (-1, 2))
	# 	xmin = sorted(list(contour), key=lambda x: x[0])[0][0]
	# 	xmax = sorted(list(contour), key=lambda x: x[0])[-1][0]
	# 	ymin = sorted(list(contour), key=lambda x: x[1])[0][1]
	# 	ymax = sorted(list(contour), key=lambda x: x[1])[-1][1]
	# 	cut_image = img[ymin:ymax, xmin:xmax, :]
	#
	# cv2.imwrite(os.path.join('../cut/', str(count) + '-' + str(count) + '.jpg'), o*255)
	# count += 1







import os
from timeit import default_timer as timer
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from dec.label_dict import num2label_dic
from utiles.box_utiles import polygons_from_bitmap
from utiles.transform import resize_image

IMAGE_SIZE = [320, 320]
CUT_SIZE_1 = (320, 320)
CUT_SIZE_2 = (64, 64)
db_model_path_1 = '../pbMode/db_model_1.pb'
db_model_path_2 = '../pbMode/db_model_2.pb'
dec_model_path = '../pbMode/dec_model.pb'
image_root = 'E:/CODES/TensorFlow_OCR/dataset/chip_number2/train_images'

sess = tf.Session()
graph_def = tf.GraphDef()
with gfile.FastGFile(db_model_path_1, 'rb') as f:

	graph_def.ParseFromString(f.read())
	with sess.graph.as_default():
		tf.import_graph_def(graph_def, name='')
with gfile.FastGFile(db_model_path_2, 'rb') as f:

	graph_def.ParseFromString(f.read())
	with sess.graph.as_default():
		tf.import_graph_def(graph_def, name='')
with gfile.FastGFile(dec_model_path, 'rb') as f:

	graph_def.ParseFromString(f.read())
	with sess.graph.as_default():
		tf.import_graph_def(graph_def, name='')


sess.run(tf.global_variables_initializer())

db_input_image_1 = sess.graph.get_tensor_by_name('image_input:0')
db_out_1 = sess.graph.get_tensor_by_name('dbnet/proba3_sigmoid:0')
db_input_image_2 = sess.graph.get_tensor_by_name('step2/step2_image_input:0')
db_out_2 = sess.graph.get_tensor_by_name('step2/dbnet/proba3_sigmoid:0')
dec_input_image = sess.graph.get_tensor_by_name('cut_image_input:0')
decision_out = sess.graph.get_tensor_by_name('decision_out:0')

image_names = [i[2] for i in os.walk(image_root)][0]
name_dic = {}

for i in image_names:
	start = timer()
	image_path = os.path.join(image_root, i)
	image = cv2.imread(image_path)
	image = image.astype(np.float32)
	img = image.copy()
	image = resize_image(IMAGE_SIZE[0], image)
	image = image / 255.0
	image = np.array(image[np.newaxis, :, :, :])

	proba_map = sess.run(db_out_1, feed_dict={db_input_image_1: image})
	proba_map = np.squeeze(proba_map, 0)
	image = np.squeeze(image, axis=0)
	bitmap = proba_map > 0.3
	boxes, scores = polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)

	w_scale = 2048/320
	for contour in boxes:
		contour = np.reshape(contour, (-1, 2))
		xmin = int(sorted(list(contour), key=lambda x: x[0])[0][0]*w_scale)
		xmax = int(sorted(list(contour), key=lambda x: x[0])[-1][0]*w_scale)
		ymin = int(sorted(list(contour), key=lambda x: x[1])[0][1]*w_scale)
		ymax = int(sorted(list(contour), key=lambda x: x[1])[-1][1]*w_scale)

		cut_image = img[ymin:ymax, xmin:xmax, :]
		cv2.imshow('cut_image_1', cut_image/255.0)
		cv2.waitKey()
		cv2.destroyAllWindows()

	cut_image = resize_image(IMAGE_SIZE[0], cut_image)
	cut_image = cut_image / 255.0
	cut_image = np.array(cut_image[np.newaxis, :, :, :])
	proba_map = sess.run(db_out_2, feed_dict={db_input_image_2: cut_image})
	proba_map = np.squeeze(proba_map, 0)
	bitmap = proba_map > 0.3
	cut_image = np.squeeze(cut_image, axis=0)
	boxes, scores = polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)
	for contour in boxes:
		contour = np.reshape(contour, (-1, 2))
		xmin = sorted(list(contour), key=lambda x: x[0])[0][0]
		xmax = sorted(list(contour), key=lambda x: x[0])[-1][0]
		ymin = sorted(list(contour), key=lambda x: x[1])[0][1]
		ymax = sorted(list(contour), key=lambda x: x[1])[-1][1]
		cut_image_2 = cut_image[ymin:ymax, xmin:xmax, :]
		cv2.imshow('cut_image__2', cut_image_2)
		cv2.waitKey()
		cv2.destroyAllWindows()
		cut_image_2 = cv2.resize(cut_image_2, CUT_SIZE_2)
		cut_image_2 = np.array(cut_image_2[np.newaxis, :, :, :])

		decision = sess.run(decision_out, feed_dict={dec_input_image: cut_image_2})
		print(decision)
	# 	cut_image = np.array(cut_image[np.newaxis, :, :, :])
	# 	decision = sess.run(decision_out, feed_dict={dec_input_image: cut_image / 255.0})[0]
	# 	label = num2label_dic[str(decision)]
	# 	name_str.append((xmin, label))
	# name_str = sorted(name_str, key=lambda x: x[0])
	# name = [i[1] for i in name_str]
	# string = ''.join(name)
	# with open('./123.txt', 'a') as f:
	# 	f.write('{} {}\n'.format(image_path, string))
	# end = timer()
	# print(end - start)

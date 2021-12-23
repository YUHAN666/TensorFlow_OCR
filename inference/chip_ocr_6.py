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
CUT_SIZE = (64, 64)
db_model_path_1 = '../pbMode/chip_ocr_6_db_model_1.pb'
db_model_path_2 = '../pbMode/chip_ocr_6_db_model_2.pb'
dec_model_path = '../pbMode/dec_model.pb'
image_root = 'E:/CODES/TensorFlow_OCR/dataset/chip_ocr_6_1/test_images/'
# image_root = 'E:/CODES/FAST-SCNN/DATA/carrier2/'
sess = tf.Session()
with gfile.FastGFile(db_model_path_1, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')

with gfile.FastGFile(db_model_path_2, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')

with gfile.FastGFile(dec_model_path, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())

db_input_image_1 = sess.graph.get_tensor_by_name('image_input:0')
db_out_1 = sess.graph.get_tensor_by_name('dbnet/proba3_sigmoid:0')
db_input_image_2 = sess.graph.get_tensor_by_name('step2/image_input:0')
db_out_2 = sess.graph.get_tensor_by_name('step2/dbnet/proba3_sigmoid:0')
dec_input_image = sess.graph.get_tensor_by_name('cut_image_input:0')
decision_out = sess.graph.get_tensor_by_name('decision_out:0')

image_names = [i[2] for i in os.walk(image_root)][0]
name_dic = {}


def inference_model_1(image, session, input_tensor, output_tensor):
	img = image.copy()
	image = image.astype(np.float32)

	resized_image = resize_image(IMAGE_SIZE[0], image)
	resized_image = np.array(resized_image[np.newaxis, :, :, :])
	resized_image /= 255.0

	proba_map = session.run(output_tensor, feed_dict={input_tensor: resized_image})
	proba_map = np.squeeze(proba_map, axis=0)
	bitmap = proba_map > 0.3
	contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# boxes, scores = polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)
	scale = image.shape[1] / 320
	xwidth = 96
	ywidth = 60
	min_distance = image.shape[1]
	cut_center_x = 0
	cut_center_y = 0
	for contour in contours:
		contour = np.reshape(contour, (-1, 2))
		xmin = int(sorted(list(contour), key=lambda x: x[0])[0][0] * scale)
		xmax = int(sorted(list(contour), key=lambda x: x[0])[-1][0] * scale)
		ymin = int(sorted(list(contour), key=lambda x: x[1])[0][1] * scale)
		ymax = int(sorted(list(contour), key=lambda x: x[1])[-1][1] * scale)

		area = (xmax - xmin) * (ymax - ymin)
		print("面积：{}".format(area))
		if area < 6000:
			continue
		xcenter = int((xmin + xmax) // 2)
		ycenter = int((ymin + ymax) // 2)
		distance = abs(xcenter - image.shape[1] // 2)
		if distance < min_distance:
			cut_center_x = xcenter
			cut_center_y = ycenter
			min_distance = distance

	cut_image = img[cut_center_y - ywidth:cut_center_y + ywidth, cut_center_x - xwidth:cut_center_x + xwidth, :]
	# cv2.imshow('cut_image_1', cut_image)
	# cv2.waitKey()
	# cv2.destroyAllWindows()

	return cut_image


def inference_model_2(image, session, input_tensor, output_tensor, input_tensor_2, output_tensor_2):
	img = image.copy()
	image = image.astype(np.float32)

	resized_image = resize_image(IMAGE_SIZE[0], image)
	resized_image = np.array(resized_image[np.newaxis, :, :, :])
	resized_image /= 255.0

	proba_map = session.run(output_tensor, feed_dict={input_tensor: resized_image})
	proba_map = np.squeeze(proba_map, axis=0)
	bitmap = proba_map > 0.5
	contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# boxes, scores = polygons_from_bitmap(proba_map, bitmap, image.shape[0], image.shape[1], box_thresh=0.5)
	scale = image.shape[1] / 320
	xwidth = 15
	ywidth = 20
	cut_center_numbers = []
	for contour in contours:
		contour = np.reshape(contour, (-1, 2))
		xmin = int(sorted(list(contour), key=lambda x: x[0])[0][0] * scale)
		xmax = int(sorted(list(contour), key=lambda x: x[0])[-1][0] * scale)
		ymin = int(sorted(list(contour), key=lambda x: x[1])[0][1] * scale)
		ymax = int(sorted(list(contour), key=lambda x: x[1])[-1][1] * scale)

		area = (xmax - xmin) * (ymax - ymin)
		print("面积：{}".format(area))
		if area<150:
			continue
		xcenter = int((xmin + xmax) // 2)
		ycenter = int((ymin + ymax) // 2)
		cut_image = img[ycenter - ywidth:ycenter + ywidth, xcenter - xwidth:xcenter + xwidth, :]

		decision = inference_model_3(cut_image, session, input_tensor_2, output_tensor_2)

		cut_center_numbers.append((xcenter, ycenter, decision))
	up_line = ''.join([num2label_dic[str(i[2])] for i in list(sorted(list(sorted(cut_center_numbers, key=lambda x:x[1]))[:5], key=lambda x:x[0]))])
	down_line = ''.join([num2label_dic[str(i[2])] for i in list(sorted(list(sorted(cut_center_numbers, key=lambda x:x[1]))[5:], key=lambda x:x[0]))])


	return up_line, down_line


def inference_model_3(image, session, input_tensor, output_tensor):

	image = image.astype(np.float32)
	# cv2.imshow('cut_image_2', image/255.0)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	resized_image = cv2.resize(image, CUT_SIZE)
	resized_image = np.array(resized_image[np.newaxis, :, :, :])
	resized_image /= 255.0
	decision = session.run(output_tensor, feed_dict={input_tensor: resized_image})

	return decision[0]


for i in image_names:
	image_path = os.path.join(image_root, i)
	image = cv2.imread(image_path)
	try:
		cut_image_1 = inference_model_1(image, sess, db_input_image_1, db_out_1)
		up_line, down_line = inference_model_2(cut_image_1, sess, db_input_image_2, db_out_2, dec_input_image, decision_out)

		cv2.putText(image, up_line, (100, 200), 1, 10, (0,0,255))
		cv2.putText(image, down_line, (100, 400), 1, 10, (0,0,255))
	except:
		print('error')
		continue

	image = cv2.resize(image, (1024, 928))
	cv2.imshow('image', image)
	cv2.waitKey()
	cv2.destroyAllWindows()


# todo: 1.exception 2. bad numbers probability


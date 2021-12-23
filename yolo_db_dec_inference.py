import os
from timeit import default_timer as timer

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

IMAGE_SIZE = [320, 320]
INPUT_SHAPE = [416, 416]
CUT_SIZE = (96, 96)


def rand(a=0., b=1.):
	return np.random.rand() * (b - a) + a


def resize_image_with_pad(image, input_shape):
	if len(input_shape) < 3:
		input_shape = (input_shape[0], input_shape[1], 3)

	scale = min(input_shape[0] / image.shape[0], input_shape[1] / image.shape[1])
	img = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale) + 1))

	pad_image = np.ones(input_shape, np.uint8) * 128.0
	pad_image[(input_shape[0] - img.shape[0]) // 2:(input_shape[0] - (input_shape[0] - img.shape[0]) // 2),
	(input_shape[1] - img.shape[1]) // 2:(input_shape[1] - (input_shape[1] - img.shape[1]) // 2), :] = img

	return pad_image


yolo_model_path = './pbMode/yolo_model.pb'
db_model_path = './pbMode/db_model.pb'
dec_model_path = './pbMode/dec_model.pb'
image_root = './dataset/selfmade9/jpegimages/'

sess = tf.Session()
with gfile.FastGFile(yolo_model_path, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')

with gfile.FastGFile(db_model_path, 'rb') as f:
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

yolo_input_image = sess.graph.get_tensor_by_name('yolo_image_input:0')
yolo_input_shape = sess.graph.get_tensor_by_name('yolo_input_shape:0')
yolo_image_shape = sess.graph.get_tensor_by_name('yolo_image_shape:0')
yolo_output_boxes = sess.graph.get_tensor_by_name('yolo_output_boxes:0')
yolo_output_scores = sess.graph.get_tensor_by_name('yolo_output_scores:0')
yolo_output_classes = sess.graph.get_tensor_by_name('yolo_output_classes:0')

db_input_image = sess.graph.get_tensor_by_name('image_input:0')
db_out = sess.graph.get_tensor_by_name('dbnet/proba3_sigmoid:0')
dec_input_image = sess.graph.get_tensor_by_name('cut_image_input:0')
decision_out = sess.graph.get_tensor_by_name('decision_out:0')
image_names = [i[2] for i in os.walk(image_root)][0]
name_dic = {}

for i in image_names:
	image_path = os.path.join(image_root, i)
	image = cv2.imread(image_path, 1)

	image_data = resize_image_with_pad(image, INPUT_SHAPE)
	image_data = image_data / 255.0
	cv2.imshow('image_data', image_data)
	cv2.waitKey()
	cv2.destroyAllWindows()
	batch_image_data = np.expand_dims(image_data, 0)
	start = timer()

	boxes, scores, classes = sess.run([yolo_output_boxes, yolo_output_scores, yolo_output_scores],
	                                  feed_dict={yolo_input_image: batch_image_data,
	                                             yolo_input_shape: np.array((416, 416)),
	                                             yolo_image_shape: np.array((211, 310))})

	end = timer()
	print('time: {}s'.format(end - start))
	print(boxes)
	for j in range(boxes.shape[0]):
		xmin = int(boxes[j][0])
		xmax = int(boxes[j][2])
		ymin = int(boxes[j][1])
		ymax = int(boxes[j][3])
		pred_classes = classes[j]
		if xmin < 0 or ymin < 0 or xmax > image.shape[1] or ymax > image.shape[0]:
			continue

		cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
	# cv2.putText(img, str(pred_classes), (xmin, ymax), 1, 10.0, (0,0,255))

	img = cv2.resize(image, (1024, 768))

	cv2.imshow(i, img / 255.0)
	cv2.waitKey()
	cv2.destroyAllWindows()

import os
from timeit import default_timer as timer

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

INPUT_SHAPE = (320, 320)
IMAGE_SHAPE = (1200, 1920)
model_path = '../pbMode/123.pb'
image_root = '../dataset/yolo_pzt/JPEGImages/'


def resize_image_with_pad(image, input_shape):

	if len(input_shape) < 3:
		input_shape = (input_shape[0], input_shape[1], 3)

	scale = min(input_shape[0] / image.shape[0], input_shape[1] / image.shape[1])
	img = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

	pad_image = np.ones(input_shape, np.uint8)*128.0
	pad_image[(input_shape[0] - img.shape[0]) // 2:(input_shape[0] - (input_shape[0] - img.shape[0]) // 2),
	(input_shape[1] - img.shape[1]) // 2:(input_shape[1] - (input_shape[1] - img.shape[1]) // 2), :] = img

	return pad_image

sess = tf.Session()
with gfile.FastGFile(model_path, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')

input_image = sess.graph.get_tensor_by_name('yolo_image_input:0')
input_shape_placeholder = sess.graph.get_tensor_by_name('yolo_input_shape:0')
image_shape_placeholder = sess.graph.get_tensor_by_name('yolo_image_shape:0')
boxes = sess.graph.get_tensor_by_name('yolo_output_boxes:0')
scores = sess.graph.get_tensor_by_name('yolo_output_scores:0')
classes = sess.graph.get_tensor_by_name('yolo_output_classes:0')
image_names = [i[2] for i in os.walk(image_root)][0]


with sess.as_default():
	for path in image_names:

		image_path = os.path.join(image_root, path)
		image = cv2.imread(image_path)
		img = image.copy()
		# image = resize_image_with_pad(image, INPUT_SHAPE)
		# image = image/255.0
		batch_image = np.expand_dims(image, axis=0)
		input_shape = np.array(INPUT_SHAPE)
		image_shape = np.array(IMAGE_SHAPE)
		start = timer()

		box, score, cla = sess.run([boxes, scores, classes], feed_dict={input_image: batch_image,
		                                                                input_shape_placeholder: input_shape,
		                                                                image_shape_placeholder: image_shape})
		end = timer()
		print('{}s'.format(end-start))

		for j in range(box.shape[0]):
			xmin = int(box[j][0])
			xmax = int(box[j][2])
			ymin = int(box[j][1])
			ymax = int(box[j][3])
			pred_classes = cla[j][0]

			cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
			cv2.putText(img, str(pred_classes), (xmin, ymax), 1, 10.0, (0, 0, 255))

		img = cv2.resize(img, (1024, 768))
		cv2.imshow(image_path, img)
		cv2.waitKey()
		cv2.destroyWindow(image_path)










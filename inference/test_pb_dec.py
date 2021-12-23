from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import cv2
import numpy as np


IMAGE_SIZE = [96, 96]    # pad
from timeit import default_timer as timer


pb_file_path = '../pbMode/pzt_dec_model.pb'
image_root = 'E:/DATA/CUT/'

sess = tf.Session()
with gfile.FastGFile(pb_file_path, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	sess.graph.as_default()
	tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())
decision_out = sess.graph.get_tensor_by_name('decision_out:0')
input_image = sess.graph.get_tensor_by_name('cut_image_input:0')
image_names = [i[2] for i in os.walk(image_root)][0]

count = 0
for i in image_names:
	image_path = os.path.join(image_root, i)
	image = cv2.imread(image_path, 1)
	image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
	image = image/255
	image = np.array(image[np.newaxis,:,:,:])

	start=timer()
	decision = sess.run([decision_out], feed_dict={input_image: image})
	end = timer()
	os.rename(image_path, os.path.join(image_root, str(decision[0][0])+'-'+str(count)+'.jpg'))
	count+=1
	print(end-start)
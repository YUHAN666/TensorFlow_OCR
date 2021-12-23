import os
from timeit import default_timer as timer
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from dec.label_dict import num2label_dic
from utiles.transform import resize_image

CUT_SIZE = (96, 96)
dec_model_path = '../pbMode/pzt_dec_model.pb'
image_root = '../cut/'
save_root = '../CUT/'
sess = tf.Session()

with gfile.FastGFile(dec_model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())

dec_input_image = sess.graph.get_tensor_by_name('cut_image_input:0')
decision_out = sess.graph.get_tensor_by_name('decision_out:0')
image_names = [i[2] for i in os.walk(image_root)][0]
name_dic = {}

for i in range(len(image_names)):
    start = timer()
    image_path = os.path.join(image_root, image_names[i])

    image = cv2.imread(image_path)
    img = image.copy()
    image = image.astype(np.float32)
    image = cv2.resize(image, CUT_SIZE)
    image = image / 255.0
    # cv2.imshow('image', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    image = np.array(image[np.newaxis, :, :, :])
    decision = sess.run(decision_out, feed_dict={dec_input_image: image})
    if str(decision[0]) != image_names[i].split("-")[0]:
        os.rename(image_path, os.path.join(save_root, str(decision[0])+'-'+str(i)+'.jpg'))
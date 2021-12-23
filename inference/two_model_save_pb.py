import tensorflow as tf
from db.db_config import DefaultParam as dbParam
from dec.dec_config import DefaultParam as decParam
from db.db_model import ModelDB
from dec.dec_model import ModelDec
from logger import Logger
from saver import Saver
import os
from tensorflow.python.platform import gfile

# IMAGE_SIZE = [160, 160]
# CUT_SIZE = (64, 64)
db_model_path = '../pbMode/pzt_db_model.pb'
dec_model_path = '../pbMode/pzt_dec_model.pb'
sess = tf.Session()
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


pb_save_path = "../pbMode"
pbModel_name = "20211007.pb"

if __name__ == '__main__':

	output_node_names = ["image_input", "dbnet/proba3_sigmoid", "cut_image_input", "decision_out"]
	output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
	                                                                sess.graph_def,
	                                                                output_node_names)
	output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

	if not os.path.exists(pb_save_path):
		os.makedirs(pb_save_path)
	pbpath = os.path.join(pb_save_path, pbModel_name)
	print(" Saved to {}".format(pbpath))
	with tf.gfile.GFile(pbpath, mode='wb') as f:
		f.write(output_graph_def.SerializeToString())


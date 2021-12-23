import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from yolo.yolo_config import DefaultParam as YoloParam
from yolo.yolo_agent import AgentYOLO as AgentYOLO
from crnn.crnn_agent import AgentCrnn
from crnn.crnn_config import DefaultParam as CrnnParam
from db.db_agent import AgentDB
from chip_ocr.chip_config import DefaultParam as ChipParam
from chip_ocr.chip_agent import AgentChip
from crnn.crnn_config import DefaultParam as CrnnParam
from dec.dec_agent import AgentDec
from dec.dec_config import DefaultParam as DecParam
from db.db_config import DefaultParam as DBParam
from dec.label_dict import *
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    logger = Logger(DBParam)
    agent = AgentDec(DecParam, logger)
    # agent = AgentDB(DBParam, logger)
    # agent = AgentYOLO(YoloPx  aram, logger)
    # agent = AgentCrnn(CrnnParam, logger)
    # agent = AgentChip(ChipParam, logger)
    agent.run()


def rename_inference(param, image_dir):
    session = tf.Session()

    with gfile.GFile(os.path.join(param["pb_save_path"], param["pb_name"]), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        session.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    image_input = session.graph.get_tensor_by_name("cut_image_input:0")
    label = session.graph.get_tensor_by_name("decision_out:0")

    image_names = [i[2] for i in os.walk(image_dir)][0]
    for image_name in image_names:
        image = cv2.imread(os.path.join(image_dir, image_name))
        image = cv2.resize(image, (param["cut_image_size"], param["cut_image_size"]))
        image = image/255.0
        image_batch = image[np.newaxis, :, :, :]
        dec = session.run(label, feed_dict={image_input: image_batch})
        # print(os.path.join(image_dir, image_name))
        if num2label_dic[str(dec[0])] != image_name.split('-')[0]:
            print(image_name)
            print('{}: {}'.format(num2label_dic[str(dec[0])], image_name.split('-')[0]))
        # else:
            # print('{}: {}'.format(num2label_dic[str(dec[0])], image_name.split('-')[0]))


if __name__ == '__main__':
    main()
    # rename_inference(DecParam, './dataset/selfmade3/cut/train/')
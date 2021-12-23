import tensorflow as tf
from db.db_config import DATA_FORMAT, ACTIVATION
from models.decmodel_components.decision_head import decision_head
from base.model import Model
from models.decmodel_components.ghostnet import ghostnet_base
from dec.label_dict import *
slim=tf.contrib.slim


class ModelDec(Model):

    def __init__(self, sess, param):
        self.step = 0
        self.session = sess
        self.bn_momentum = param["momentum"]
        self.cut_image_size = param["cut_image_size"]
        self.cut_image_channel = param["cut_image_channel"]
        self.mode = param["mode"]
        self.backbone = param["backbone"]
        self.tensorboard_logdir = param["tensorboard_dir"]
        self.checkPoint_dir = param["checkpoint_dir_dec"]
        self.batch_size = param["batch_size_cut"]
        self.batch_size_inference = param["batch_size_inference"]
        self.class_num = len(label2num_dic)

        with self.session.as_default():
            if self.mode == 'train_dec':
                self.is_training = tf.placeholder(tf.bool, name='is_training')
                self.keep_dropout = True
                self.image_input = tf.placeholder(tf.float32,
                                                  shape=(self.batch_size, self.cut_image_size, self.cut_image_size,
                                                         self.cut_image_channel), name='cut_image_input')
                self.label = tf.placeholder(tf.float32, shape=(self.batch_size, self.class_num), name='label_input')

            else:
                self.is_training = False
                self.keep_dropout = False
                self.image_input = tf.placeholder(tf.float32, shape=(self.batch_size_inference, self.cut_image_size,
                                                  self.cut_image_size, self.cut_image_channel), name='cut_image_input')
                self.label = tf.placeholder(tf.float32, shape=(self.batch_size_inference, self.class_num), name='label_input')
            self.logits, self.decision_out, self.prob = self.build_model()

    def build_model(self):

        # Set depth_multiplier to change the depth of GhostNet
        backbone_output = ghostnet_base(self.image_input, mode=self.mode, data_format=DATA_FORMAT, scope='ghostnet',
                                        dw_code=None, ratio_code=None,
                                        se=1, min_depth=8, depth=1, depth_multiplier=0.5, conv_defs=None,
                                        is_training=self.is_training, momentum=self.bn_momentum)
        # Create decision head
        dec_out = decision_head(backbone_output[-1], backbone_output[1], class_num=512, scope='decision',
                                keep_dropout_head=self.keep_dropout,
                                training=self.is_training, data_format=DATA_FORMAT, momentum=self.bn_momentum,
                                mode=self.mode, activation=ACTIVATION)
        logits = slim.fully_connected(dec_out, self.class_num, activation_fn=None)
        decision_out = tf.nn.softmax(logits)
        prob = tf.reduce_max(decision_out, axis=1, keepdims=False, name='decision_prob')
        decision_out = tf.argmax(decision_out, axis=1, name='decision_out')

        return logits, decision_out, prob

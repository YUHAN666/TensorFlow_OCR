import tensorflow as tf
from models.dbmodel_components.resnet50 import resnet_50
from models.dbmodel_components.dbnet import dbnet
from db.db_config import DATA_FORMAT
from base.model import Model


class ModelDB(Model):

    def __init__(self, sess, param, logger):
        self.step = 0
        self.session = sess
        self.logger = logger
        self.bn_momentum = param["momentum"]
        self.mode = param["mode"]
        self.image_size = param["image_size"]
        self.image_channel = param["image_channel"]
        self.checkPoint_dir = param["checkpoint_dir_db"]
        self.logger.info("Building model... backbone:{}, neck:{}".format(param["backbone"], param["neck"]))
        self.batch_size = param["batch_size"]
        self.batch_size_inference = param["batch_size_inference"]

        # with tf.variable_scope('step2'):
        with self.session.as_default():
            # Build placeholder to receive data
            if self.mode == 'train_db':

                self.is_training = tf.placeholder(tf.bool, name='is_training')

                self.image_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size,
                                                                     self.image_channel), name='image_input')
                self.gt_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size), name='gt_input')

                self.mask_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size), name='mask_input')

                self.thresh_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size), name='mask_input')

                self.thresh_mask_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size), name='thresh_mask_input')

            else:
                self.is_training = False

                self.image_input = tf.placeholder(tf.float32, shape=(self.batch_size_inference, self.image_size,
                                                                     self.image_size, self.image_channel), name='image_input')

                self.p, self.t, self.b_hat = self.build_model()
                return

            # Building model graph
            self.p, self.t, self.b_hat = self.build_model()

            self.inputs = [self.gt_input, self.mask_input, self.thresh_input, self.thresh_mask_input]
            self.outputs = [self.p, self.t, self.b_hat]

    def build_model(self):
        """
        Build model graph in session
        :return: segmentation_output: nodes for calculating segmentation loss
                 decision_output: nodes for calculating decision loss
                 mask_out: nodes for visualization output mask of the model
        """

        feature_list = resnet_50(self.image_input, self.is_training, DATA_FORMAT, self.bn_momentum, 'resnet50')
        p, t, b_hat = dbnet(feature_list, 50, self.is_training, DATA_FORMAT, self.bn_momentum, 'dbnet')

        return p, t, b_hat






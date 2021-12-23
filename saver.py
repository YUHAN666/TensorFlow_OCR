# coding=utf-8
import tensorflow as tf
import os
import numpy as np
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools import inspect_checkpoint as chkp


class Saver(object):

    def __init__(self, sess, param, checkpoint_dir, logger, model=None):
        """
        :param sess: session with model graph built
        :param param: checkpoint_dir: the path of checkpoint file
                      max_to_keep: max number of checkpoint to keep during training
                      input_list: input nodes of model when save to pb model
                      output_list: output nodes of model when save to pb model
                      pb_save_path: path to save pb model
                      pb_name: name to save pb model
                      mode: whether CBR module is used or not in model

        """
        self.session = sess
        self.model=model
        self.logger = logger
        self.checkPoint_dir = checkpoint_dir
        self.max_to_keep = param["max_to_keep"]
        self.input_nodes = param["input_list"]
        self.output_nodes = param["output_list"]
        self.pb_save_path = param["pb_save_path"]
        self.pbModel_name = param["pb_name"]
        self.saving_mode = param["saving_mode"]
        self.mode = param["mode"]

        self.ckpt = tf.train.latest_checkpoint(self.checkPoint_dir)
        self.init_op = tf.global_variables_initializer()
        self.step = 0

        with self.session.as_default():
            self.init_op.run()
            self.var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            self.var_list += bn_moving_vars

            if self.saving_mode == "CBR" and self.mode == 'savePb':
                # CBR模块在训练时不使用bias，部署时将BN层融合到conv层的bias中
                var_list2 = [v for v in self.var_list if "bias" not in v.name or "CBR" not in v.name]

                self.saver = tf.train.Saver(var_list2, max_to_keep=self.max_to_keep)
            else:
                self.saver = tf.train.Saver(self.var_list, max_to_keep=self.max_to_keep)

    def load_checkpoint(self):
        """
        Restore session from checkpoint
        Used when continue to train

        """
        if self.ckpt:
            self.step = int(self.ckpt.split('-')[1])
            print('Restoring from epoch:{}'.format(self.step))
            self.logger.info("Restoring from {}".format(self.ckpt))
            self.saver.restore(self.session, self.ckpt)
        else:
            print('Cannot find checkpoint, start with new model')
            self.logger.info('Cannot find checkpoint, start with new model')

    def save_checkpoint(self, step):
        """
        Save session to checkpoint during training
        """
        if not os.path.exists(self.checkPoint_dir):
            os.makedirs(self.checkPoint_dir)

        self.saver.save(self.session, os.path.join(self.checkPoint_dir, 'ckp'), global_step=+step, write_meta_graph=False)
        self.logger.info("Saved to {}".format(self.checkPoint_dir+'/ckp-{}'.format(step)))

    def save_pb(self):
        """
        Restore session from checkpoint and save to .pb model
        the saving_mode should be set to CBR if the model used CBR module(combine BatchNorm layer with Conv layer in order to speed up inference),

        """
        if self.saving_mode == "CBR":
            reader = pywrap_tensorflow.NewCheckpointReader(self.ckpt)
            var_to_shape_map = reader.get_variable_to_shape_map()
            source_list = [key for key in var_to_shape_map if "CBR" in key]
            epsilon = 0.001
            for key in source_list:
                if "moving_mean" in key:
                    mean = np.array(reader.get_tensor(key))

                    key_var = key[0:-11] + "moving_variance"
                    var = np.array(reader.get_tensor(key_var))

                    key_gamma = key[0:-11] + "gamma"
                    gamma = np.array(reader.get_tensor(key_gamma))

                    key_beta = key[0:-11] + "beta"
                    beta = np.array(reader.get_tensor(key_beta))

                    key_W = key[0:-14] + "Conv2D/kernel"
                    W = np.array(reader.get_tensor(key_W))

                    key_B = key_W[0:-6] + "bias"

                    alpha = gamma / ((var + epsilon) ** 0.5)

                    try:
                        B = np.array(reader.get_tensor((key_B)))
                        B_new = beta - mean * alpha + B * alpha
                    except:
                        B_new = beta - mean * alpha

                    W_new = W * alpha

                    weight = tf.get_default_graph().get_tensor_by_name(key_W + ':0')

                    update_weight = tf.assign(weight, W_new)

                    # bias_name = key_W[0:-6] + 'bias:0'

                    bias = tf.get_default_graph().get_tensor_by_name(key_B + ':0')

                    update_bias = tf.assign(bias, B_new)

                    self.session.run(update_weight)
                    self.session.run(update_bias)

        else:
            self.saver.restore(self.session, self.ckpt)

        # tf.saved_model.simple_save(self.session, './pb_model_dec', inputs={'my_input': self.model.image_input},
        #                            outputs={'my_output': self.model.decision_out})  #dbnet/proba3_sigmoid


        output_node_names = self.input_nodes + self.output_nodes
        output_graph_def = tf.graph_util.convert_variables_to_constants(self.session,
                                                                        self.session.graph_def,
                                                                        output_node_names)
        output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

        if not os.path.exists(self.pb_save_path):
            os.makedirs(self.pb_save_path)
        pbpath = os.path.join(self.pb_save_path, self.pbModel_name)
        print(" Saved to {}".format(pbpath))
        print("Model inputs: {}".format(self.input_nodes))
        print("Model outputs: {}".format(self.output_nodes))
        with tf.gfile.GFile(pbpath, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())

    def inspect_checkpoint(self):
        """ Print nodes in checkpoint"""
        # graph = tf.get_default_graph()
        # print(graph._nodes_by_name)
        chkp.print_tensors_in_checkpoint_file(file_name=self.ckpt,
                                              tensor_name=None,  # 如果为None,则默认为ckpt里的所有变量
                                              all_tensors=False,  # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
                                              all_tensor_names=False)  # bool 是否console.WriteLine(files1[i]);打印所有的tensor的name

    @staticmethod
    def count_checkpoint_parameter(inspect_checkpoint_path):
        """ Count the parameter number of model """
        ckpt = tf.train.get_checkpoint_state(inspect_checkpoint_path).model_checkpoint_path
        saver = tf.train.import_meta_graph(ckpt + '.meta')
        variables = tf.trainable_variables()
        total_parameters = 0
        for variable in variables:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        print("the total number of parameter is {}".format(total_parameters))


import tensorflow as tf
from models.dbmodel_components.conv_block import conv_stage


def res_block(res_input, inter_channel, out_channel, stride, change_dim, is_training, data_format, momentum, use_bias, name):

    if change_dim:
        res_branch = conv_stage(res_input, filters=out_channel, kernel_size=1, strides=stride, training=is_training,
                                data_format=data_format, activation=None, momentum=momentum, use_bias=use_bias, name=name+'_shortcut')
    else:
        res_branch = res_input
    x = conv_stage(res_input, filters=inter_channel, kernel_size=1, strides=stride, training=is_training,
                   data_format=data_format, activation='relu', momentum=momentum, use_bias=use_bias, name=name+'_inconv')
    x = conv_stage(x, filters=inter_channel, kernel_size=3, strides=1, training=is_training,
                   data_format=data_format, activation='relu', momentum=momentum, use_bias=use_bias, name=name+'_conv')
    x = conv_stage(x, filters=out_channel, kernel_size=1, strides=1, training=is_training,
                   data_format=data_format, activation=None, momentum=momentum, use_bias=use_bias, name=name+'_outconv')

    x = tf.add(x, res_branch, name=name + '_add')
    res_output = tf.nn.relu(x, name=name + 'relu')

    return res_output

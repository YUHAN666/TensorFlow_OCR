import tensorflow as tf
from models.dbmodel_components.conv_block import conv_stage
from models.dbmodel_components.res_block import res_block


# def resnet_50(image_input, is_training, data_format, momentum, name):
#
#     with tf.variable_scope(name):
#
#         x = conv_stage(image_input, filters=64, kernel_size=7, strides=2, training=is_training, data_format=data_format,
#                        activation='relu', momentum=momentum, pool=True, pool_size=(3, 3), pool_stride=(2, 2), name='conv1')
#
#         x = res_block(x, inter_channel=64, out_channel=256, stride=1, change_dim=True, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_2a')
#         x = res_block(x, inter_channel=64, out_channel=256, stride=1, change_dim=False, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_2b')
#         res_2c = res_block(x, inter_channel=64, out_channel=256, stride=1, change_dim=False, is_training=is_training,
#                            data_format=data_format, momentum=momentum, use_bias=True, name='res_2c')
#
#         x = res_block(res_2c, inter_channel=128, out_channel=512, stride=2, change_dim=True, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_3a')
#         x = res_block(x, inter_channel=128, out_channel=512, stride=1, change_dim=False, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_3b')
#         x = res_block(x, inter_channel=128, out_channel=512, stride=1, change_dim=False, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_3c')
#         res_3d = res_block(x, inter_channel=128, out_channel=512, stride=1, change_dim=False, is_training=is_training,
#                            data_format=data_format, momentum=momentum, use_bias=True, name='res_3d')
#
#         x = res_block(res_3d, inter_channel=256, out_channel=1024, stride=2, change_dim=True, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4a')
#         x = res_block(x, inter_channel=256, out_channel=1024, stride=1, change_dim=False, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4b')
#         x = res_block(x, inter_channel=256, out_channel=1024, stride=1, change_dim=False, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4c')
#         x = res_block(x, inter_channel=256, out_channel=1024, stride=1, change_dim=False, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4d')
#         x = res_block(x, inter_channel=256, out_channel=1024, stride=1, change_dim=False, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4e')
#         res_4f = res_block(x, inter_channel=256, out_channel=1024, stride=1, change_dim=False, is_training=is_training,
#                            data_format=data_format, momentum=momentum, use_bias=True, name='res_4f')
#
#         x = res_block(res_4f, inter_channel=512, out_channel=2048, stride=2, change_dim=True, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_5a')
#         x = res_block(x, inter_channel=512, out_channel=2048, stride=1, change_dim=False, is_training=is_training,
#                       data_format=data_format, momentum=momentum, use_bias=True, name='res_block_5b')
#         res_5c = res_block(x, inter_channel=512, out_channel=2048, stride=1, change_dim=False, is_training=is_training,
#                            data_format=data_format, momentum=momentum, use_bias=True, name='res_5c')
#
#     return res_2c, res_3d, res_4f, res_5c

def resnet_50(image_input, is_training, data_format, momentum, name):

    with tf.variable_scope(name):

        x = conv_stage(image_input, filters=64, kernel_size=7, strides=2, training=is_training, data_format=data_format,
                       activation='relu', momentum=momentum, pool=True, pool_size=(3, 3), pool_stride=(2, 2), name='conv1')

        x = res_block(x, inter_channel=64, out_channel=128, stride=1, change_dim=True, is_training=is_training,
                      data_format=data_format, momentum=momentum, use_bias=True, name='res_block_2a')
        x = res_block(x, inter_channel=64, out_channel=128, stride=1, change_dim=False, is_training=is_training,
                      data_format=data_format, momentum=momentum, use_bias=True, name='res_block_2b')
        res_2c = res_block(x, inter_channel=64, out_channel=128, stride=1, change_dim=False, is_training=is_training,
                           data_format=data_format, momentum=momentum, use_bias=True, name='res_2c')

        x = res_block(res_2c, inter_channel=128, out_channel=256, stride=2, change_dim=True, is_training=is_training,
                      data_format=data_format, momentum=momentum, use_bias=True, name='res_block_3a')
        x = res_block(x, inter_channel=128, out_channel=256, stride=1, change_dim=False, is_training=is_training,
                      data_format=data_format, momentum=momentum, use_bias=True, name='res_block_3b')
        x = res_block(x, inter_channel=128, out_channel=256, stride=1, change_dim=False, is_training=is_training,
                      data_format=data_format, momentum=momentum, use_bias=True, name='res_block_3c')
        res_3d = res_block(x, inter_channel=128, out_channel=256, stride=1, change_dim=False, is_training=is_training,
                           data_format=data_format, momentum=momentum, use_bias=True, name='res_3d')

        # x = res_block(res_3d, inter_channel=256, out_channel=512, stride=2, change_dim=True, is_training=is_training,
        #               data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4a')
        # # x = res_block(x, inter_channel=256, out_channel=1024, stride=1, change_dim=False, is_training=is_training,
        # #               data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4b')
        # # x = res_block(x, inter_channel=256, out_channel=1024, stride=1, change_dim=False, is_training=is_training,
        # #               data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4c')
        # # x = res_block(x, inter_channel=256, out_channel=1024, stride=1, change_dim=False, is_training=is_training,
        # #               data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4d')
        # x = res_block(x, inter_channel=256, out_channel=512, stride=1, change_dim=False, is_training=is_training,
        #               data_format=data_format, momentum=momentum, use_bias=True, name='res_block_4e')
        # res_4f = res_block(x, inter_channel=256, out_channel=512, stride=1, change_dim=False, is_training=is_training,
        #                    data_format=data_format, momentum=momentum, use_bias=True, name='res_4f')

        # x = res_block(res_4f, inter_channel=512, out_channel=2048, stride=2, change_dim=True, is_training=is_training,
        #               data_format=data_format, momentum=momentum, use_bias=True, name='res_block_5a')
        # x = res_block(x, inter_channel=512, out_channel=2048, stride=1, change_dim=False, is_training=is_training,
        #               data_format=data_format, momentum=momentum, use_bias=True, name='res_block_5b')
        # res_5c = res_block(x, inter_channel=512, out_channel=2048, stride=1, change_dim=False, is_training=is_training,
        #                    data_format=data_format, momentum=momentum, use_bias=True, name='res_5c')

    return res_2c, res_3d

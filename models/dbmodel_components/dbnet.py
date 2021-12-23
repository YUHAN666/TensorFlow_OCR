import tensorflow as tf
from models.dbmodel_components.conv_block import conv_stage, transpose_conv_stage
from tensorflow import keras


# def dbnet(feature_list, k, is_training, data_format, momentum, name):
#
#     C2, C3, C4, C5 = feature_list
#     with tf.variable_scope(name):
#         in2 = conv_stage(C2, 256, 1, 1, is_training, activation=None, bn=False, use_bias=True, name='in2')
#         in3 = conv_stage(C3, 256, 1, 1, is_training, activation=None, bn=False, use_bias=True, name='in3')
#         in4 = conv_stage(C4, 256, 1, 1, is_training, activation=None, bn=False, use_bias=True, name='in4')
#         in5 = conv_stage(C5, 256, 1, 1, is_training, activation=None, bn=False, use_bias=True, name='in5')
#
#         P5 = keras.layers.UpSampling2D(size=(8, 8))(
#             conv_stage(in5, 64, 3, 1, is_training, activation=None, bn=False, use_bias=True, name='P5'))
#         # 1 / 16 * 4 = 1 / 4
#         out4 = tf.add(in4, keras.layers.UpSampling2D(size=(2, 2))(in5), name='out4_add')
#         P4 = keras.layers.UpSampling2D(size=(4, 4))(
#             conv_stage(out4, 64, 3, 1, is_training, activation=None, bn=False, use_bias=True, name='P4'))
#         # 1 / 8 * 2 = 1 / 4
#         out3 = tf.add(in3, keras.layers.UpSampling2D(size=(2, 2))(out4), name='out3_add')
#         P3 = keras.layers.UpSampling2D(size=(2, 2))(
#             conv_stage(out3, 64, 3, 1, is_training, activation=None, bn=False, use_bias=True, name='P3'))
#         # 1 / 4
#         out2 = tf.add(in2, keras.layers.UpSampling2D(size=(2, 2))(out3), name='out2_add')
#         P2 = conv_stage(out2, 64, 3, 1, is_training, activation=None, bn=False, use_bias=True, name='P2')
#         # (b, /4, /4, 256)
#         fuse = tf.concat([P2, P3, P4, P5], axis=-1)
#
#         # probability map
#         p = conv_stage(fuse, filters=64, kernel_size=3, strides=1, training=is_training,
#                        activation='relu', momentum=momentum, use_bias=False, name='proba1')
#         p = transpose_conv_stage(p, filters=64, kernel_size=2, strides=2, training=is_training,
#                                  data_format=data_format, activation='relu', momentum=momentum, name='proba2')
#         p = transpose_conv_stage(p, filters=1, kernel_size=2, strides=2, training=is_training, use_bias=True,
#                                  data_format=data_format, activation='sigmoid', momentum=momentum, name='proba3')
#
#         # threshold map
#         t = conv_stage(fuse, filters=64, kernel_size=3, strides=1, training=is_training,
#                        activation='relu', momentum=momentum, use_bias=False, name='thres1')
#         t = transpose_conv_stage(t, filters=64, kernel_size=2, strides=2, training=is_training,
#                                  data_format=data_format, activation='relu', momentum=momentum, name='thres2')
#         t = transpose_conv_stage(t, filters=1, kernel_size=2, strides=2, training=is_training, use_bias=True,
#                                  data_format=data_format, activation='sigmoid', momentum=momentum, name='thres3')
#
#         # approximate binary map
#         b_hat = keras.layers.Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([p, t])
#
#     return p, t, b_hat



# def dbnet(feature_list, k, is_training, data_format, momentum, name):
#
#     C2, C3, C4 = feature_list
#     with tf.variable_scope(name):
#         in2 = conv_stage(C2, 256, 1, 1, is_training, activation=None, bn=False, use_bias=True, name='in2')
#         in3 = conv_stage(C3, 256, 1, 1, is_training, activation=None, bn=False, use_bias=True, name='in3')
#         in4 = conv_stage(C4, 256, 1, 1, is_training, activation=None, bn=False, use_bias=True, name='in4')
#
#         P4 = keras.layers.UpSampling2D(size=(4, 4))(
#             conv_stage(in4, 64, 3, 1, is_training, activation=None, bn=False, use_bias=True, name='P4'))
# #         # 1 / 8 * 2 = 1 / 4
#         out3 = tf.add(in3, keras.layers.UpSampling2D(size=(2, 2))(in4), name='out3_add')
#         P3 = keras.layers.UpSampling2D(size=(2, 2))(
#             conv_stage(out3, 64, 3, 1, is_training, activation=None, bn=False, use_bias=True, name='P3'))
#         # 1 / 4
#         out2 = tf.add(in2, keras.layers.UpSampling2D(size=(2, 2))(in3), name='out2_add')
#         P2 = conv_stage(out2, 64, 3, 1, is_training, activation=None, bn=False, use_bias=True, name='P2')
#         # (b, /4, /4, 256)
#         fuse = tf.concat([P2, P3], axis=-1)
#
#         # probability map
#         p = conv_stage(fuse, filters=64, kernel_size=3, strides=1, training=is_training,
#                        activation='relu', momentum=momentum, use_bias=False, name='proba1')
#         p = transpose_conv_stage(p, filters=64, kernel_size=2, strides=2, training=is_training,
#                                  data_format=data_format, activation='relu', momentum=momentum, name='proba2')
#         p = transpose_conv_stage(p, filters=1, kernel_size=2, strides=2, training=is_training, use_bias=True,
#                                  data_format=data_format, activation='sigmoid', momentum=momentum, name='proba3')
#
#         # threshold map
#         t = conv_stage(fuse, filters=64, kernel_size=3, strides=1, training=is_training,
#                        activation='relu', momentum=momentum, use_bias=False, name='thres1')
#         t = transpose_conv_stage(t, filters=64, kernel_size=2, strides=2, training=is_training,
#                                  data_format=data_format, activation='relu', momentum=momentum, name='thres2')
#         t = transpose_conv_stage(t, filters=1, kernel_size=2, strides=2, training=is_training, use_bias=True,
#                                  data_format=data_format, activation='sigmoid', momentum=momentum, name='thres3')
#
#         # approximate binary map
#         b_hat = keras.layers.Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([p, t])
#
#     return p, t, b_hat


def dbnet(feature_list, k, is_training, data_format, momentum, name):

    C2, C3 = feature_list
    with tf.variable_scope(name):
        in2 = conv_stage(C2, 256, 1, 1, is_training, activation=None, bn=False, use_bias=True, name='in2')
        in3 = conv_stage(C3, 256, 1, 1, is_training, activation=None, bn=False, use_bias=True, name='in3')

        P3 = keras.layers.UpSampling2D(size=(2, 2))(
            conv_stage(in3, 64, 3, 1, is_training, activation=None, bn=False, use_bias=True, name='P3'))
        # 1 / 4
        out2 = tf.add(in2, keras.layers.UpSampling2D(size=(2, 2))(in3), name='out2_add')
        P2 = conv_stage(out2, 64, 3, 1, is_training, activation=None, bn=False, use_bias=True, name='P2')
        # (b, /4, /4, 256)
        fuse = tf.concat([P2, P3], axis=-1)

        # probability map
        p = conv_stage(fuse, filters=64, kernel_size=3, strides=1, training=is_training,
                       activation='relu', momentum=momentum, use_bias=False, name='proba1')
        p = transpose_conv_stage(p, filters=64, kernel_size=2, strides=2, training=is_training,
                                 data_format=data_format, activation='relu', momentum=momentum, name='proba2')
        p = transpose_conv_stage(p, filters=1, kernel_size=2, strides=2, training=is_training, use_bias=True,
                                 data_format=data_format, activation='sigmoid', momentum=momentum, name='proba3')

        # threshold map
        t = conv_stage(fuse, filters=64, kernel_size=3, strides=1, training=is_training,
                       activation='relu', momentum=momentum, use_bias=False, name='thres1')
        t = transpose_conv_stage(t, filters=64, kernel_size=2, strides=2, training=is_training,
                                 data_format=data_format, activation='relu', momentum=momentum, name='thres2')
        t = transpose_conv_stage(t, filters=1, kernel_size=2, strides=2, training=is_training, use_bias=True,
                                 data_format=data_format, activation='sigmoid', momentum=momentum, name='thres3')

        # approximate binary map
        b_hat = keras.layers.Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([p, t])

    return p, t, b_hat

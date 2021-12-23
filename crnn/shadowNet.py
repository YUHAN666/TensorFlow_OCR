import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from models.dbmodel_components.conv_block import conv_stage


def shadow_net(image_input, is_training, momentum, name):

    with tf.variable_scope(name):
        x = conv_stage(image_input, 64, 3, 1, training=is_training, momentum=momentum, activation='relu',
                       pool=True, use_bias=True, name='conv1')
        x = conv_stage(x, 128, 3, 1, bn=True, training=is_training, momentum=momentum, activation='relu',
                       pool=True, use_bias=True, name='conv2')
        x = conv_stage(x, 256, 3, 1, bn=True, training=is_training, momentum=momentum, activation='relu',
                       pool=False, use_bias=False, name='conv3')
        x = conv_stage(x, 256, 3, 1, bn=True, training=is_training, momentum=momentum, activation='relu',
                       pool=True, pool_size=(2, 1), pool_stride=(2, 1), use_bias=False, name='conv4')
        x = conv_stage(x, 512, 3, 1, bn=True, training=is_training, momentum=momentum, activation='relu',
                       pool=False, use_bias=False, name='conv5')
        x = conv_stage(x, 512, 3, 1, bn=True, training=is_training, momentum=momentum, activation='relu',
                       pool=True, pool_size=(2, 1), pool_stride=(2, 1), use_bias=False, name='conv6')
        x = conv_stage(x, 512, 3, (2, 1), bn=True, training=is_training, momentum=momentum, activation='relu',
                       pool=False, use_bias=False, name='conv7')
    return x


def rnn_head(feature_input, hidden_size, layer_num, num_classes, name):

    shape = feature_input.get_shape().as_list()
    assert shape[1] == 1

    with tf.variable_scope(name):
        feature_input = tf.squeeze(feature_input, axis=1)
        fw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                        nh in [hidden_size] * layer_num]
        # Backward direction cells
        bw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                        nh in [hidden_size] * layer_num]

        stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
            fw_cell_list, bw_cell_list, feature_input,
            dtype=tf.float32
        )

        [_, _, hidden_nums] = feature_input.get_shape().as_list()  # [batch, width, 2*n_hidden]

        shape = tf.shape(stack_lstm_layer)
        rnn_reshaped = tf.reshape(stack_lstm_layer, [shape[0] * shape[1], shape[2]])

        w = tf.get_variable(
            name='w',
            shape=[hidden_nums, num_classes],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
            trainable=True
        )

        # Doing the affine projection
        logits = tf.matmul(rnn_reshaped, w, name='logits')

        logits = tf.reshape(logits, [shape[0], shape[1], num_classes], name='logits_reshape')

        raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2, name='raw_prediction')

        # Swap batch and batch axis
        rnn_out = tf.transpose(logits, [1, 0, 2], name='transpose_time_major')

    return raw_pred, rnn_out


if __name__ == '__main__':
    # image_input = tf.placeholder(tf.float32, shape=(4, 32, 100, 3))
    #
    # x = shadow_net(image_input, True, 0.99, 'shadow_net')
    #
    # y = rnn_head(x, 256, 2, 10, 'rnn_head')

    max_text_len = 6
    num_label = 10
    batch_size = 2
    logits_placeholder = tf.placeholder(tf.float32, shape=(25, batch_size, num_label))
    label_placeholder = tf.placeholder(tf.int32, shape=(batch_size, max_text_len))
    input_length_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    label_length_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    logits = np.zeros((25, batch_size, num_label), np.float32)
    label = np.array([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 0, 0]], np.int32)
    input_length = np.array([25, 25], np.int32)
    label_length = np.array([6, 6], np.int32)
    ctc_loss = tf.nn.ctc_loss_v2(label_placeholder, logits_placeholder, label_length_placeholder,
                                 input_length_placeholder)
    tf.nn.ctc_loss_v2()
    sess = tf.Session()
    loss = sess.run(ctc_loss, feed_dict={logits_placeholder: logits,
                                         label_placeholder: label,
                                         label_length_placeholder: label_length,
                                         input_length_placeholder: input_length})
    print('loss: {}'.format(loss))

    max_text_len = 4
    num_label = 10
    batch_size = 2
    logits_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 25, num_label))
    label_placeholder = tf.placeholder(tf.int32, shape=(batch_size, max_text_len))
    input_length_placeholder = tf.placeholder(tf.int32, shape=(batch_size, 1))
    label_length_placeholder = tf.placeholder(tf.int32, shape=(batch_size, 1))

    logits = np.zeros((batch_size, 25, num_label), np.float32)
    label = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.int32)
    input_length = np.array([[25], [25]], np.int32)
    label_length = np.array([[4], [4]], np.int32)
    ctc_loss2 = tf.keras.backend.ctc_batch_cost(y_true=label_placeholder, y_pred=logits_placeholder,
                                                input_length=input_length_placeholder,
                                                label_length=label_length_placeholder)
    loss2 = sess.run(ctc_loss2, feed_dict={logits_placeholder: logits,
                                           label_placeholder: label,
                                           label_length_placeholder: label_length,
                                           input_length_placeholder: input_length})

    print('loss2: {}'.format(loss2))

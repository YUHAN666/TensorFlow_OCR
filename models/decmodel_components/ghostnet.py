""" Implementation of Ghostnet """
from collections import namedtuple
import tensorflow as tf
# from GhostNetModule import ConvBlock
from models.decmodel_components.GhostNetModule import GhostModule as MyConv
from models.decmodel_components.GhostNetModule import MyDepthConv as DepthConv
from dec.dec_config import ACTIVATION, ATTENTION
from models.decmodel_components.convblock import ConvBatchNormRelu as CBR
from models.decmodel_components.squeeze_layers import Squeeze_excitation_layer as SElayer

kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'factor', 'se'])
Bottleneck = namedtuple('Bottleneck', ['kernel', 'stride', 'depth', 'factor', 'se'])

_CONV_DEFS_0 = [
    Conv(kernel=[3, 3], stride=2, depth=16, factor=1, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=16, factor=1, se=0),

    Bottleneck(kernel=[3, 3], stride=2, depth=24, factor=48 / 16, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=24, factor=72 / 24, se=0),

    Bottleneck(kernel=[5, 5], stride=2, depth=40, factor=72 / 24, se=1),
    Bottleneck(kernel=[5, 5], stride=1, depth=40, factor=120 / 40, se=1),

    Bottleneck(kernel=[3, 3], stride=2, depth=80, factor=240 / 40, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=200 / 80, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184 / 80, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184 / 80, se=0),

    Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=480 / 80, se=1),
    Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=672 / 112, se=1),
    Bottleneck(kernel=[5, 5], stride=2, depth=160, factor=672 / 112, se=1),

    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=0),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=1),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=0),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=1),

]


def shape2d(a):
    """
    Ensure a 2D shape.

    Args:
        a: a int or tuple/list of length 2

    Returns:
        list: of length 2. if ``a`` is a int, return ``[a, a]``.
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))


def ghostnet_base(inputs,
                  mode,
                  data_format,
                  min_depth=8,
                  depth_multiplier=1.0,
                  depth=1.0,
                  conv_defs=None,
                  output_stride=None,
                  dw_code=None,
                  ratio_code=None,
                  se=1,
                  scope=None,
                  is_training=False,
                  momentum=0.9):

    """ By adjusting depth_multiplier can change the depth of network """
    if data_format == 'channels_first':
        axis = 1
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
    else:
        axis = -1

    output_layers = []

    def depth(d):
        d = max(int(d * depth_multiplier), min_depth)
        d = round(d / 4) * 4
        return d

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS_0

    if dw_code is None or len(dw_code) < len(conv_defs):
        dw_code = [3] * len(conv_defs)
    print('dw_code', dw_code)

    if ratio_code is None or len(ratio_code) < len(conv_defs):
        ratio_code = [2] * len(conv_defs)
    print('ratio_code', ratio_code)

    se_code = [x.se for x in conv_defs]
    print('se_code', se_code)

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    with tf.variable_scope(scope, 'MobilenetV2', [inputs]):

        """
        The current_stride variable keeps track of the output stride of the
        activations, i.e., the running product of convolution strides up to the
        current network layer. This allows us to invoke atrous convolution
        whenever applying the next convolution would result in the activations
        having output stride larger than the target output_stride.
        """
        current_stride = 1
        # The atrous convolution rate parameter.
        rate = 1
        net = inputs
        in_depth = 3
        gi = 0

        for i, conv_def in enumerate(conv_defs):
            layer_stride = conv_def.stride
            current_stride *= conv_def.stride
            if layer_stride != 1:
                output_layers.append(net)

            if isinstance(conv_def, Conv):
                # net = ConvBlock(net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride,
                #                 name='ConvBlock_{}'.format(i), is_training=is_training, activation=ACTIVATION)
                net = CBR(net, depth(conv_def.depth), conv_def.kernel[0], strides=conv_def.stride, training=is_training,
                          momentum=momentum, mode=mode, name='ConvBlock_{}'.format(i), padding='same', data_format=data_format,
                          activation=ACTIVATION, bn=True, use_bias=False)

            # Bottleneck block.
            elif isinstance(conv_def, Bottleneck):
                # Stride > 1 or different depth: no residual part.
                if layer_stride == 1 and in_depth == conv_def.depth:
                    res = net
                else:
                    res = DepthConv(net, conv_def.kernel, stride=layer_stride, data_format=data_format, name='Bottleneck_block_{}_shortcut_dw'.format(i))
                    res = tf.layers.batch_normalization(res, training=is_training, name='Bottleneck_block_{}_shortcut_dw_BN'.format(i), axis=axis)

                    res = CBR(res, depth(conv_def.depth), 1, 1, training=is_training, momentum=momentum, mode=mode,
                              name='Bottleneck_block_{}_shortcut_1x1'.format(i), padding='same',
                              data_format=data_format, activation=ACTIVATION, bn=True, use_bias=False)

                # Increase depth with 1x1 conv.
                net = MyConv('Bottleneck_block_{}_up_pointwise'.format(i), net, depth(in_depth * conv_def.factor), 1,
                             dw_code[gi], ratio_code[gi], mode=mode, strides=1, data_format=data_format, use_bias=False, is_training=is_training, activation=ACTIVATION, momentum=momentum)

                # Depthwise conv2d.
                if layer_stride > 1:
                    net = DepthConv(net, conv_def.kernel, stride=layer_stride, data_format=data_format, name='Bottleneck_block_{}_depthwise'.format(i))
                    net = tf.layers.batch_normalization(net, training=is_training, name='Bottleneck_block_{}_depthwise_BN'.format(i), axis=axis)

                # SE
                if se_code[i] > 0 and se > 0:
                    if ATTENTION == 'se':
                        # net = SELayer(net, depth(in_depth * conv_def.factor), 4)
                        net = SElayer(net, depth(in_depth * conv_def.factor), depth(in_depth * conv_def.factor)//4, "se_{}".format(i), data_format=data_format)
                # Downscale 1x1 conv.
                net = MyConv('Bottleneck_block_{}_down_pointwise'.format(i), net, depth(conv_def.depth), 1, dw_code[gi],
                             ratio_code[gi], mode=mode, strides=1, data_format=data_format, use_bias=False, is_training=is_training, activation=ACTIVATION, momentum=momentum)

                gi += 1

                # Residual connection?
                net = tf.add(res, net, name='Bottleneck_block_{}_Add'.format(i)) if res is not None else net

            in_depth = conv_def.depth
            # Final end point?
        output_layers.pop(0)
        output_layers.append(net)
        return output_layers

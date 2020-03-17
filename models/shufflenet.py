import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, SmartInit, model_utils
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils import logger

# Shuffle net


@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
              W_init=None, activation=tf.identity):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1]
    assert out_channel % in_channel == 0, (out_channel, in_channel)
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = tf.variance_scaling_initializer(2.0)
    kernel_shape = [kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, 1, stride, stride], padding=padding, data_format='NCHW')
    return activation(conv, name='output')


@under_name_scope()
def channel_shuffle(l, group):
    in_shape = l.get_shape().as_list()
    in_channel = in_shape[1]
    assert in_channel % group == 0, in_channel
    l = tf.reshape(l, [-1, in_channel // group, group] + in_shape[-2:])
    l = tf.transpose(l, [0, 2, 1, 3, 4])
    l = tf.reshape(l, [-1, in_channel] + in_shape[-2:])
    return l


@layer_register()
def shufflenet_unit(l, out_channel, group, stride):
    in_shape = l.get_shape().as_list()
    in_channel = in_shape[1]
    shortcut = l

    # "We do not apply group convolution on the first pointwise layer
    #  because the number of input channels is relatively small."
    first_split = group if in_channel > 24 else 1
    l = Conv2D('conv1', l, out_channel // 4, 1, split=first_split, activation=BNReLU)
    l = channel_shuffle(l, group)
    l = DepthConv('dconv', l, out_channel // 4, 3, stride=stride)
    l = BatchNorm('dconv_bn', l)

    l = Conv2D('conv2', l,
               out_channel if stride == 1 else out_channel - in_channel,
               1, split=group)
    l = BatchNorm('conv2_bn', l)
    if stride == 1:     # unit (b)
        output = tf.nn.relu(shortcut + l)
    else:   # unit (c)
        shortcut = AvgPooling('avgpool', shortcut, 3, 2, padding='SAME')
        output = tf.concat([shortcut, tf.nn.relu(l)], axis=1)
    return output


@layer_register()
def shufflenet_unit_v2(l, out_channel, stride):
    if stride == 1:
        shortcut, l = tf.split(l, 2, axis=1)
    else:
        shortcut, l = l, l
    shortcut_channel = int(shortcut.shape[1])

    l = Conv2D('conv1', l, out_channel // 2, 1, activation=BNReLU)
    l = DepthConv('dconv', l, out_channel // 2, 3, stride=stride)
    l = BatchNorm('dconv_bn', l)
    l = Conv2D('conv2', l, out_channel - shortcut_channel, 1, activation=BNReLU)

    if stride == 2:
        shortcut = DepthConv('shortcut_dconv', shortcut, shortcut_channel, 3, stride=2)
        shortcut = BatchNorm('shortcut_dconv_bn', shortcut)
        shortcut = Conv2D('shortcut_conv', shortcut, shortcut_channel, 1, activation=BNReLU)
    output = tf.concat([shortcut, l], axis=1)
    output = channel_shuffle(output, 2)
    return output


@layer_register(log_shape=True)
def shufflenet_stage(input, channel, num_blocks, group):
    l = input
    for i in range(num_blocks):
        name = 'block{}'.format(i)
        if True: #args.v2:
            l = shufflenet_unit_v2(name, l, channel, 2 if i == 0 else 1)
        else:
            l = shufflenet_unit(name, l, channel, group, 2 if i == 0 else 1)
    return l


def ShuffleNet(image, classes=5):
    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='channels_first'), \
        argscope(Conv2D, use_bias=False):
        image_channel_first = tf.transpose(image, [0, 3, 1, 2])
        group = 8 #args.group
        ratio = 0.5
        if not True: #args.v2:
            # Copied from the paper
            channels = {
                3: [240, 480, 960],
                4: [272, 544, 1088],
                8: [384, 768, 1536]
            }
            mul = group * 4  # #chan has to be a multiple of this number
            channels = [int(math.ceil(x * ratio / mul) * mul)
                        for x in channels[group]]
            # The first channel must be a multiple of group
            first_chan = int(math.ceil(24 * ratio / group) * group)
        else:
            # Copied from the paper
            channels = {
                0.5: [48, 96, 192],
                1.: [116, 232, 464]
            }[ratio]
            first_chan = 24

        logger.info("#Channels: " + str([first_chan] + channels))

        l = Conv2D('conv1', image_channel_first, first_chan, 3, strides=2, activation=BNReLU)
        l = MaxPooling('pool1', l, 3, 2, padding='SAME')

        l = shufflenet_stage('stage2', l, channels[0], 4, group)
        l = shufflenet_stage('stage3', l, channels[1], 8, group)
        l = shufflenet_stage('stage4', l, channels[2], 4, group)

        if True: #args.v2:
            l = Conv2D('conv5', l, 1024, 1, activation=BNReLU)

        l = GlobalAvgPooling('gap', l)
        output = FullyConnected('linear', l, classes)
        return output

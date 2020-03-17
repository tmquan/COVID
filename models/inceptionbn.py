import tensorflow as tf
tf = tf.compat.v1
from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, SmartInit, model_utils
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils import logger

# InceptionBN net


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


def InceptionBN(image, classes=5):
    def inception(name, x, nr1x1, nr3x3r, nr3x3, nr233r, nr233, nrpool, pooltype):
        stride = 2 if nr1x1 == 0 else 1
        with tf.variable_scope(name):
            outs = []
            if nr1x1 != 0:
                outs.append(Conv2D('conv1x1', x, nr1x1, 1))
            x2 = Conv2D('conv3x3r', x, nr3x3r, 1)
            outs.append(Conv2D('conv3x3', x2, nr3x3, 3, strides=stride))

            x3 = Conv2D('conv233r', x, nr233r, 1)
            x3 = Conv2D('conv233a', x3, nr233, 3)
            outs.append(Conv2D('conv233b', x3, nr233, 3, strides=stride))

            if pooltype == 'max':
                x4 = MaxPooling('mpool', x, 3, stride, padding='SAME')
            else:
                assert pooltype == 'avg'
                x4 = AvgPooling('apool', x, 3, stride, padding='SAME')
            if nrpool != 0:  # pool + passthrough if nrpool == 0
                x4 = Conv2D('poolproj', x4, nrpool, 1)
            outs.append(x4)
            return tf.concat(outs, 3, name='concat')

    with argscope(Conv2D, activation=BNReLU, use_bias=False):
        l = (LinearWrap(image)
             .Conv2D('conv0', 64, 7, strides=2)
             .MaxPooling('pool0', 3, 2, padding='SAME')
             .Conv2D('conv1', 64, 1)
             .Conv2D('conv2', 192, 3)
             .MaxPooling('pool2', 3, 2, padding='SAME')())
        # 28
        l = inception('incep3a', l, 64, 64, 64, 64, 96, 32, 'avg')
        l = inception('incep3b', l, 64, 64, 96, 64, 96, 64, 'avg')
        l = inception('incep3c', l, 0, 128, 160, 64, 96, 0, 'max')

        br1 = (LinearWrap(l)
               .Conv2D('loss1conv', 128, 1)
               .FullyConnected('loss1fc', 1024, activation=tf.nn.relu)
               .FullyConnected('loss1logit', 1000, activation=tf.identity)())
        # loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=br1, labels=label)
        # loss1 = tf.reduce_mean(loss1, name='loss1')

        # 14
        l = inception('incep4a', l, 224, 64, 96, 96, 128, 128, 'avg')
        l = inception('incep4b', l, 192, 96, 128, 96, 128, 128, 'avg')
        l = inception('incep4c', l, 160, 128, 160, 128, 160, 128, 'avg')
        l = inception('incep4d', l, 96, 128, 192, 160, 192, 128, 'avg')
        l = inception('incep4e', l, 0, 128, 192, 192, 256, 0, 'max')

        br2 = Conv2D('loss2conv', l, 128, 1)
        br2 = FullyConnected('loss2fc', br2, 1024, activation=tf.nn.relu)
        br2 = FullyConnected('loss2logit', br2, 1000, activation=tf.identity)
        # loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=br2, labels=label)
        # loss2 = tf.reduce_mean(loss2, name='loss2')

        # 7
        l = inception('incep5a', l, 352, 192, 320, 160, 224, 128, 'avg')
        l = inception('incep5b', l, 352, 192, 320, 192, 224, 128, 'max')
        l = GlobalAvgPooling('gap', l)

        output = FullyConnected('linear', l, classes, activation=tf.identity)
        return output

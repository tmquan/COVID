import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, SmartInit, model_utils
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils import logger

import numpy as np


def GroupNorm(x, group, gamma_initializer=tf.constant_initializer(1.)):
    """
    https://arxiv.org/abs/1803.08494
    More code that reproduces the paper can be found at https://github.com/ppwwyyxx/GroupNorm-reproduce/.
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')


def convnormrelu(x, name, chan, norm='gn'):
    x = Conv2D(name, x, chan, 3)
    if norm == 'bn':
        x = BatchNorm(name + '_bn', x)
    elif norm == 'gn':
        with tf.variable_scope(name + '_gn'):
            x = GroupNorm(x, 32)
    x = tf.nn.relu(x, name=name + '_relu')
    return x


def VGG16(image, classes=5):
    with argscope(Conv2D, kernel_initializer=tf.variance_scaling_initializer(scale=2.)), \
        argscope([Conv2D, MaxPooling, BatchNorm], data_format='channels_first'):
        image_channel_first = tf.transpose(image, [0, 3, 1, 2])
        latent = (LinearWrap(image_channel_first)
                  .apply(convnormrelu, 'conv1_1', 64)
                  .apply(convnormrelu, 'conv1_2', 64)
                  .MaxPooling('pool1', 2)
                  # 112
                  .apply(convnormrelu, 'conv2_1', 128)
                  .apply(convnormrelu, 'conv2_2', 128)
                  .MaxPooling('pool2', 2)
                  # 56
                  .apply(convnormrelu, 'conv3_1', 256)
                  .apply(convnormrelu, 'conv3_2', 256)
                  .apply(convnormrelu, 'conv3_3', 256)
                  .MaxPooling('pool3', 2)
                  # 28
                  .apply(convnormrelu, 'conv4_1', 512)
                  .apply(convnormrelu, 'conv4_2', 512)
                  .apply(convnormrelu, 'conv4_3', 512)
                  .MaxPooling('pool4', 2)
                  # 14
                  .apply(convnormrelu, 'conv5_1', 512)
                  .apply(convnormrelu, 'conv5_2', 512)
                  .apply(convnormrelu, 'conv5_3', 512)
                  .MaxPooling('pool5', 2)
                  ())
                  # 7
        logits = (LinearWrap(latent)
                  .FullyConnected('fc6', 4096,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.001))
                  .tf.nn.relu(name='fc6_relu')
                  .Dropout('drop0', rate=0.5)
                  .FullyConnected('fc7', 4096,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.001))
                  .tf.nn.relu(name='fc7_relu')
                  .Dropout('drop1', rate=0.5)
                  .FullyConnected('fc8', classes,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01))())
        return logits, latent

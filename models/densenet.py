import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, SmartInit, model_utils
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils import logger

import numpy as np


# DenseNet net

def composite_function(_input, out_features, kernel_size=3):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    """
    # BN
    output = BNReLU('bn_compos', _input)
    # convolution
    output = Conv2D('conv_compos', output, out_features, kernel_size)
    return output


def densenet_bottleneck(_input, out_features):
    output = BNReLU('bn_bottleneck', _input)
    inter_features = out_features * 4
    output = Conv2D('conv_bottleneck', output, inter_features, 1)
    return output


def add_layer(_input, growth_rate, bc_mode):
    """Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    if not bc_mode:
        comp_out = composite_function(
            _input, out_features=growth_rate, kernel_size=3)
    elif bc_mode:
        bottleneck_out = densenet_bottleneck(_input, out_features=growth_rate)
        comp_out = composite_function(
            bottleneck_out, out_features=growth_rate, kernel_size=3)
    # concatenate _input with out from composite function
    output = tf.concat([comp_out, _input], 3, name='concat')
    return output


def add_transition(_input, name, bc_mode, theta):
    shape = _input.get_shape().as_list()
    out_features = shape[3]
    with tf.variable_scope(name) as scope:
        if bc_mode:
            out_features = int(out_features * theta)
            print(out_features, theta)
        output = composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = AvgPooling('pool', output, 2)
    return output


def densenet_block(_input, name, growth_rate, bc_mode, count):
    output = _input
    with tf.variable_scope(name):
        for i in range(count):
            with tf.variable_scope('block{}'.format(i)):
                output = add_layer(output, growth_rate, bc_mode)
        return output


def densenet_backbone(image, num_blocks, classes=1000, growth_rate=32, bc_mode=False, theta=0.5):
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=tf.contrib.layers.variance_scaling_initializer(mode='FAN_OUT')):
        print('backbone:', bc_mode, theta)
        latent = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(densenet_block, 'dense_group0', growth_rate, bc_mode, num_blocks[0])
                  .apply(add_transition, 'trans_group0', bc_mode, theta)
                  .apply(densenet_block, 'dense_group1', growth_rate, bc_mode, num_blocks[1])
                  .apply(add_transition, 'trans_group1', bc_mode, theta)
                  .apply(densenet_block, 'dense_group2', growth_rate, bc_mode, num_blocks[2])
                  .apply(add_transition, 'trans_group2', bc_mode, theta)
                  .apply(densenet_block, 'dense_group3', growth_rate, bc_mode, num_blocks[3])
                  .BNReLU('bnlast')
                  # .GlobalAvgPooling('gap')
                  #. .FullyConnected('linear', classes, nl=tf.identity)
                  ())
        logits =(LinearWrap(latent)
                    .GlobalAvgPooling('gap')
                    .Dropout('dropout', 0.5)
                    .FullyConnected('linear', classes, nl=tf.identity)
                    ())
        latent = tf.transpose(latent, [0, 3, 1, 2])
    return logits, latent

DENSENET_CONFIG = {
  121: [6, 12, 24, 16],
  169: [6, 12, 32, 32],
  201: [6, 12, 48, 32],
  265: [6, 12, 64, 48]
}

def DenseNet121(image, classes=5):
	return densenet_backbone(image, num_blocks=[6, 12, 24, 16], classes=classes, \
							 growth_rate=32, bc_mode=False, theta=0.5)

def DenseNet169(image, classes=5):
    return densenet_backbone(image, num_blocks=[6, 12, 32, 32], classes=classes, \
                             growth_rate=32, bc_mode=False, theta=0.5)

def DenseNet201(image, classes=5):
    return densenet_backbone(image, num_blocks=[6, 12, 48, 32], classes=classes, \
                             growth_rate=32, bc_mode=False, theta=0.5)
    # def DenseNet(image, classes=5):
    #     depth = 40
    #     N = int((depth - 4)  / 3)
    #     growthRate = 12
    #     l = conv('conv0', image, 16, 1)
    #     with tf.variable_scope('block1') as scope:

    #         for i in range(N):
    #             l = add_layer('dense_layer.{}'.format(i), l)
    #         l = add_transition('transition1', l)

    #     with tf.variable_scope('block2') as scope:

    #         for i in range(N):
    #             l = add_layer('dense_layer.{}'.format(i), l)
    #         l = add_transition('transition2', l)

    #     with tf.variable_scope('block3') as scope:

    #         for i in range(N):
    #             l = add_layer('dense_layer.{}'.format(i), l)
    #     l = BatchNorm('bnlast', l)
    #     l = tf.nn.relu(l)
    #     l = GlobalAvgPooling('gap', l)
    #     output = FullyConnected('linear', l, out_dim=classes, nl=tf.identity)

    #     return output

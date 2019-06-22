# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf


class FCN_VGG(object):

    def __init__(self, train_config):

        self.batch_size = train_config['batch_size']
        self.height = train_config['crop_h']
        self.width = train_config['crop_w']


    def vgg_model(self, x, is_train):
        """

        build a vgg16 net
        :param x: the input image,tensor shape are [batch_size, height, width, channels]
        :return: the dict of pools feature map
        """
        output = {}
        x = conv2d('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = conv2d('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = pool2d('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        output['x1'] = x

        x = conv2d('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = conv2d('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = pool2d('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        output['x2'] = x

        x = conv2d('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = conv2d('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = conv2d('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = pool2d('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        output['x3'] = x

        x = conv2d('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = conv2d('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = conv2d('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = pool2d('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        output['x4'] = x

        x = conv2d('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = conv2d('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = conv2d('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=is_train)
        x = pool2d('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
        output['x5'] = x
        return output

    def fcn_model(self, x, n_class, is_training=True):
        """

        :param x: the vgg model output
        :param n_class: the number od class
        :param is_training: for batch_normalization
        :return: the feature map
        """
        output = x
        pool5 = output['x5']  # size=(N, x.H/32, x.W/32, 512)
        pool4 = output['x4']  # size=(N, x.H/16, x.W/16, 512)
        pool3 = output['x3']  # size=(N, x.H/8,  x.W/8, 256)
        pool2 = output['x2']  # size=(N, x.H/4,  x.W/4, 128)
        pool1 = output['x1']  # size=(N, x.H/2,  x.W/2, 64)
        score = conv2d_transpose('deconv1', pool5, 512,
                                 output_shape=[self.batch_size, self.height//16, self.width//16, 512],
                                 is_training=is_training)
        score += pool4
        score = conv2d_transpose('deconv2', score, 256,
                                 output_shape=[self.batch_size, self.height//8, self.width//8, 256],
                                 is_training=is_training)
        score += pool3
        score = conv2d_transpose('deconv3', score, 128,
                                 output_shape=[self.batch_size, self.height//4, self.width//4, 128],
                                 is_training=is_training)
        score += pool2
        score = conv2d_transpose('deconv4', score, 64,
                                 output_shape=[self.batch_size, self.height//2, self.width//2, 64],
                                 is_training=is_training)
        score += pool1
        score = conv2d_transpose('deconv5', score, 32,
                                 output_shape=[self.batch_size, self.height, self.width, 32],
                                 is_training=is_training)
        score = conv2d('conv', score, n_class, kernel_size=[1, 1], is_train=True)
        return score


def conv2d(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1], is_train=True):
    """Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_train: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    """

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_train,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=is_train,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


def pool2d(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
    """Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding: SAME
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    """
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x


def conv2d_transpose(layer_name, x, out_channels, output_shape,
                     kernel_size=[3, 3], stride=[1, 2, 2, 1], is_training=True):
    """

    :param layer_name:
    :param x:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param is_pretrain:
    :return:
    """
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=True,
                            shape=[kernel_size[0], kernel_size[1], out_channels, in_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=True,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=stride, padding='SAME', name='deconv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.layers.batch_normalization(x, training=is_training, name='batch_norm')
        x = tf.nn.relu(x, name='relu')
    return x

# 不使用该batch_norm, 使用自带的tf.layers.batch_normalization()
def batch_norm(x, name=None):
    """Batch normlization(I didn't include the offset and scale)
    """
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])     # Calculate the mean and variance of `x`
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon,
                                  name=name)
    return x

# tf.layers.batch_normalization()
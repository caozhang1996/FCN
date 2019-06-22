#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:18:21 2019

@author: caozhang
"""
from __future__ import print_function

from math import ceil
import os
import numpy as np
import tensorflow as tf
from torch import nn

from datasets.CamVId_loader import CamVidLoader
from net.fcnnet import FCN_VGG, conv2d_transpose
from configuration import TRAIN_CONFIG, MODEL_CONFIG


if __name__ == "__main__":
    # data_config = TRAIN_CONFIG['train_data_config']
    #
    # dataset = CamVidLoader(data_config, class_dict='CamVid/class_dict.csv')
    #
    # images, labels = dataset.get_one_batch()
    #
    # labels = tf.one_hot(labels, 32)
    # images_shape, labels_shape = images.shape, labels.shape
    # with tf.Session() as sess:
    #     print(images_shape)
    #     print(labels_shape)

    # x = tf.constant([[[[0, 1, 2],
    #                    [1, 2, 3],
    #                    [1, 2, 3],
    #                    [5, 6, 9]],
    #                   [[1, 4, 0],
    #                    [9, 10, 4],
    #                    [1, 0, 6],
    #                    [0, 3, 5]],
    #                   [[1, 11, 4],
    #                    [2, 9, 2],
    #                    [2, 3, 5],
    #                    [0, 2, 1]],
    #                   [[0, 11, 2],
    #                    [0, 11, 2],
    #                    [1, 3, 4],
    #                    [1, 4, 5]]]])
    #
    # x = tf.argmax(x, -1)
    #
    # with tf.Session() as sess:
    #     print(sess.run(x))

    x = tf.boolean_mask([0, 1, 2], [False, False, False])
    with tf.Session() as sess:
        print(sess.run(x))

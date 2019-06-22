#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:11:18 2019

@author: caozhang
"""

from __future__ import print_function

import os
import tensorflow as tf


LOG_DIR = 'Logs/fcn'  # where checkpoints, logs are saved
RUN_NAME = 'fcn-vgg16'  # identifier of the experiment

MODEL_CONFIG = {

  'frontend_config': {'frontend': 'Xception39',
                      'pretrained_dir': 'pretrain',  # path of the pretrained frontend model.
                      'train_frontend': True,
                      'use_bn': True,
                      'bn_scale': True,
                      'bn_momentum': 0.05,
                      'bn_epsilon': 1e-6,
                      'weight_decay': 5e-4,
                      'stride': 8, },
  'conv_config': {"init_method": "kaiming_normal",
                  },
  'batch_norm_params': {"scale": True,
                        # Decay for the moving averages.
                        "decay": 0.9,
                        # Epsilon to prevent 0s in variance.
                        "epsilon": 1e-5,
                        'updates_collections': tf.GraphKeys.UPDATE_OPS,  # Ensure that updates are done within a frame
                        },

}

TRAIN_CONFIG = {
  'DataSet': 'CamVid',
  'class_dict': 'CamVid/class_dict.csv',
  'train_dir': os.path.join(LOG_DIR,  RUN_NAME),
  'pre_trained_params': 'vgg_pretrain/vgg16.npy',

  'seed': 123,  # fix seed for reproducing experiments

  'train_data_config': {'preprocessing_name': 'augment',
                        'input_dir': 'train',
                        'output_dir': 'train_labels',
                        'crop_h': 800,
                        'crop_w': 800,
                        'random_scale': True,
                        'random_mirror': True,
                        'num_examples_per_epoch': 421,
                        'epoch': 2000,
                        'batch_size': 8,
                        'img_mean': tf.constant([123.68, 103.939, 116.779]),    # mean of three channels in the order of RGB
                        'prefetch_threads': 8, },

  'validation_data_config': {'preprocessing_name': 'None',
                             'input_dir': 'val',
                             'output_dir': 'val_labels',
                             'crop_h': 736,
                             'crop_w': 960,
                             'batch_size': 2,
                             'prefetch_threads': 4, },

  'test_data_config': {'preprocessing_name': 'None',
                       'input_dir': 'test',
                       'output_dir': 'test_labels',
                       'crop_h': 736,
                       'crop_w': 960,
                       'num_examples_per_epoch': 421,
                       'batch_size': 8,
                       'prefetch_threads': 4,
                       'test_dir': os.path.join(LOG_DIR, 'checkpoints', RUN_NAME+'test')},

  # Optimizer for training the model.
  'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD, RMSProp and MOMENTUM are supported
                       'momentum': 0.9,
                       'use_nesterov': False,
                       'decay': 0.9, },          # Discounting factor for history gradient(useful in RMSProp Mode)

  # Learning rate configs
  'lr_config': {'method': 'exponential',         # piecewise_constant, exponential, polynomial and cosine
                'initial_lr': 0.01,
                'power': 0.9,                   # Only useful in polynomial
                'num_epochs_per_decay': 20,
                'lr_decay_factor': 0.8685113737513527,
                'boundaries': [21050, 42100, 63150, 84200], # Only useful in piecewise_constant
                'learning_rates': [1e-2, 1e-3, 1e-4, (1e-4)/2, (1e-4)/4], # Only useful in piecewise_constant
                'staircase': True, },           # staircase 若为False则是标准的指数型衰减，True时则是阶梯式的衰减方法

  # If not None, clip gradients to this value.
  'clip_gradients': None,

  # Frequency at which loss and global step are logged
  'log_every_n_steps': 10,

  # Frequency to save model
  'save_model_every_n_step': 421 // 8,  # save model every epoch

  # How many model checkpoints to keep. No limit if None.
  'max_checkpoints_to_keep': 20,
}
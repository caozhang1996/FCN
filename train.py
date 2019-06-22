#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from datasets.CamVId_loader import CamVidLoader
from configuration import TRAIN_CONFIG, MODEL_CONFIG
from net.fcnnet import FCN_VGG


os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0

def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))


def get_loss(logits, labels):
    """Compute loss
    Args:
        logits: logits tensor, [batch_size, height, width, n_class]
        labels: one-hot labels, [batch_size, height, width, n_class]
    """
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar('total_loss', loss)

    return loss


def accuracy_and_IOU(logits, labels, n_class):
    """
    Compute accuracy and mean IOU
    :param logits: logits tensor, [batch_size, height, width, n_class]
    :param labels: one-hot labels, [batch_size, height, width, n_class]
    :return:
    """
    with tf.name_scope('accuracy_and_IOU') as scope:
        accuracy, accuracy_update_op = tf.metrics.accuracy(labels=tf.argmax(labels, -1),
                                                           predictions=tf.argmax(logits, -1))
        mean_IOU, mean_IOU_update_op = tf.metrics.mean_iou(labels=tf.argmax(labels, -1),
                                                           predictions=tf.argmax(logits, -1),
                                                           num_classes=n_class)
        with tf.control_dependencies([accuracy_update_op, mean_IOU_update_op]):
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('mean_IOU', mean_IOU)
            return accuracy, mean_IOU


def configure_learning_rate(train_config, global_step):
    lr_config = train_config['lr_config']

    num_batches_per_epoch = int(train_config['train_data_config']['num_examples_per_epoch'] /
                                train_config['train_data_config']['batch_size'])

    lr_method = lr_config['method']
    if lr_method == 'polynomial':        # 多项式衰减
        decay_steps = int(num_batches_per_epoch * train_config['train_data_config']['epoch']/2)  # epoch/2时衰减
        return tf.train.polynomial_decay(learning_rate=lr_config['initial_lr'],
                                         global_step=global_step,
                                         decay_steps=decay_steps,
                                         end_learning_rate=0.0001,
                                         power=lr_config['power'],
                                         cycle=False)

    elif lr_method == 'exponential':
        decay_steps = num_batches_per_epoch * lr_config['num_epochs_per_decay']
        return tf.train.exponential_decay(learning_rate=lr_config['initial_lr'],
                                          global_step=global_step,
                                          decay_steps=decay_steps,
                                          decay_rate=lr_config['lr_decay_factor'],
                                          staircase=lr_config['staircase'])

    elif lr_method == 'piecewise_constant':   # 分段常熟衰减
        boundaries = lr_config['boundaries']  # boundaries=[step_1, step_2, ..., step_n] 定义了在第几步进行lr衰减
        learning_rates = lr_config['learning_rates']
        return tf.train.piecewise_constant(global_step,
                                           boundaries=boundaries,
                                           values=learning_rates)


def configure_optimizer(train_config, learning_rate):
    optimizer_config = train_config['optimizer_config']
    optimizer_name = optimizer_config['optimizer']
    if optimizer_name == 'MOMENTUM':
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum=optimizer_config['momentum'],
                                               use_nesterov=optimizer_config['use_nesterov'],
                                               name='Momentum')
    elif optimizer_name == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer_name == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                              optimizer_config['decay'],
                                              optimizer_config['momentum'])
    else:
        raise TypeError('Optimizer [%s] was not recognized', optimizer_config['optimizer'])
    return optimizer


def train():

    train_config = TRAIN_CONFIG
    n_class = 32

    train_dir = train_config['train_dir']
    # if train_dir

    crop_h = train_config['train_data_config']['crop_h']
    crop_w = train_config['train_data_config']['crop_w']

    g = tf.Graph()
    with g.as_default():
        input_images = tf.placeholder(dtype=tf.float32, shape=[None, crop_h, crop_w, 3])
        input_labels = tf.placeholder(dtype=tf.int32, shape=[None, crop_h, crop_w, n_class])

        with tf.device('/cpu:0'):
            dataset = CamVidLoader(train_config['train_data_config'],
                                   train_config['DataSet'],
                                   train_config['class_dict'])

            images, labels = dataset.get_one_batch()
            labels = tf.one_hot(labels, n_class)


        fcn_vgg = FCN_VGG(train_config['train_data_config'])
        vgg_out = fcn_vgg.vgg_model(input_images, is_train=False)    # 使用vgg16.npy中的参数,这些参数不训练
        logits = fcn_vgg.fcn_model(vgg_out, n_class, is_training=True)   # if is_training=False, it's test model

        loss = get_loss(logits, input_labels)
        accuracy, mean_IOU = accuracy_and_IOU(logits, input_labels, n_class)


        global_step = tf.Variable(initial_value=0,
                                  name='global_step',
                                  trainable=False,
                                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        learning_rate = configure_learning_rate(train_config, global_step)
        optimizer = configure_optimizer(train_config, learning_rate)
        tf.summary.scalar('learning_rate', learning_rate)

        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            train_op = optimizer.minimize(loss, global_step=global_step)

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            print(tf.get_collection(tf.GraphKeys.SUMMARIES))
            # load the parameter file, assign the parameters, skip the specific layers
            load_with_skip(train_config['pre_trained_params'], sess, ['fc6', 'fc7', 'fc8'])

            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

            train_data_config = train_config['train_data_config']
            max_steps = train_data_config['epoch'] * int(train_data_config['num_examples_per_epoch'] / train_data_config['batch_size'])

            for step in range(max_steps):

                ## 训练代码
                train_images, train_labels = sess.run([images, labels])
                _, loss, accuracy, mean_IOU = sess.run([train_op, loss, accuracy, mean_IOU],
                                                       feed_dict={input_images: train_images,
                                                                  input_labels: train_labels})

                if step % 10 == 0:
                    print('Step: %d, loss: %.4f, accuracy: %.4f, mean_IOU: %.4f' % (step, loss, accuracy, mean_IOU))
                    summary_str = sess.run(summary_op, feed_dict={input_images: train_images,
                                                                  input_labels: train_labels})

                    summary_writer.add_summary(summary_str, global_step=step)

                if step % train_config['save_model_every_n_step'] == 0 or (step + 1) == max_steps:
                    checkpoints_file = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoints_file, global_step=step)


if __name__ == "__main__":
    train()

# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import logging
import pandas as pd

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def one_hot_encoding(label, label_values):
    """
    Args:
        label: the tensor of label
        label_values: the rgb value of label(list)
    Returns:
        the semactic map has n_class channels
    """
    semantic_map = []    # channels is the value of n_class
    for color in label_values:
        equality = tf.equal(label, color)   # label上对应的color变成True，其他的变成False
        class_map = tf.reduce_all(equality, axis=-1)    # 求逻辑和，将三通道压缩成单通道
        semantic_map.append(class_map)
    semantic_map = tf.stack(semantic_map, axis=-1)
    return semantic_map     # 返回n_class个通道的semantic_map， 每一个通道上的label，对应的color为True，其他为False
        

def get(config, key, default):
    """Get value in config by key, use default if key is not set
    
    This little function is useful for dynamical experimental settings.
    For example, we can add a new configuration without worrying compatibility with older versions.
    You can also achieve this by just calling config.get(key, default), but add a warning is even better : )
    """
    val = config.get(key)
    if val is None:
        logging.warning('{} is not explicitly specified, using default value: {}'.format(key, default))
        val = default
    return val


def get_label_info(csv_file):
    """
    Args:
        csv_file: the class_dict csv file
    Returns:
        Two lists: one for the class names and the other for the label values
    """
    if csv_file.split(".")[1] != "csv":
        raise TypeError("File is not a csv file!")
        
    class_names = []
    label_values = []
    info = pd.read_csv(csv_file)
    for i in range(len(info)):
        class_names.append(info.iloc[i, 0])
        label_values.append([info.iloc[i, 1], info.iloc[i, 2], info.iloc[i, 3]])
    return class_names, label_values
    
    
def _image_mirroring(img, label):
    """
    mirror the image randomly
    """
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)  # [False, True, False] or [False, False, False]
    mirror = tf.boolean_mask([0, 1, 2], mirror)   # return [1] or []
    img = tf.reverse(img, axis=mirror)            # when mirror is [1] reverse, when mirror is [] not reverse
    label = tf.reverse(label, axis=mirror)
    return img, label


def _image_scaling(img, label):
    """
    scaling th image randomly
    """
    scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=[1])       # 删除大小是1的维度
    img = tf.image.resize_images(img, new_shape)
    
    # need to expand dims because the shape of input is [batch, height, width, channels]
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)   
    label = tf.squeeze(label, axis=[0])
    
    return img, label


def _random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w):
    label = tf.to_float(label)
    image_shape = tf.shape(image)
    # TODO: only useful in camvid,  to fix
    pad_h = tf.maximum(crop_h, image_shape[0])-image_shape[0]
    pad_w = tf.maximum(crop_w, image_shape[1])-image_shape[0]
    image = tf.pad(image, [[0, pad_h], [0, pad_w], [0, 0]], constant_values=0)
    label = tf.pad(label, [[0, pad_h], [0, pad_w], [0, 0]], constant_values=30)

    combined = tf.concat(axis=2, values=[image, label])
    # combined = tf.image.pad_to_bounding_box(
    #                         combined,
    #                         0,
    #                         0,
    #                         tf.maximum(crop_h, image_shape[0]),
    #                         tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined, [crop_h, crop_w, last_image_dim+last_label_dim])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = tf.cast(label_crop, dtype=tf.int32)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    # label_crop = tf.image.resize_nearest_neighbor(tf.expand_dims(label_crop, 0), [crop_h//8, crop_w//8])
    # label_crop = tf.squeeze(label_crop, axis=0)

    return img_crop, label_crop


def _check_size(image, label, crop_h, crop_w):
    new_shape = tf.squeeze(tf.stack([[crop_h], [crop_w]]), axis=[1])
    image = tf.image.resize_images(image, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, axis=[0])
    # Set static shape so that tensorflow knows shape at compile time.
    image.set_shape((crop_h, crop_w, 3))
    label.set_shape((crop_h, crop_w, 1))
    label = tf.squeeze(label, axis=2)    # 删除维度为1
    return image, label


def _apply_with_random_selector(x, func, num_cases, label):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0], label


def _distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def _parse_function(image_filename, label_filename, img_mean, class_dict):
    """
    this function read the image and label
    
    Returns:
        the tensor of image and label
    """ 
    img_contents = tf.read_file(image_filename)
    label_contents = tf.read_file(label_filename)

    # Decode image & label
    img = tf.image.decode_png(img_contents, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    if img_mean:
        img = (img - img_mean)/tf.constant(255)

    label = tf.image.decode_png(label_contents, channels=3)
    _, label_values = get_label_info(class_dict)
    label = one_hot_encoding(label, label_values)
    label = tf.to_int32(label)   # 将True，False转换为1，0
    label = tf.argmax(label, axis=-1, output_type=tf.int32)  # get the index of the max value
    label = tf.expand_dims(label, axis=-1)

    return img, label


class CamVidLoader(object):

    def __init__(self, config, DataSet='CamVid', class_dict=None):
        self.config = config
        self.dataSet_dir = DataSet
        self.class_dict = class_dict
        self.dataset = None
        self.iterator = None
    
    def prepare_data(self):
        dataset_dir = self.dataSet_dir
        input_dir = self.config['input_dir']        # the directory of images
        output_dir = self.config['output_dir']      # the directory of labels
        crop_h = self.config['crop_h']
        crop_w = self.config['crop_w']
        threads = self.config['prefetch_threads']
        img_mean = get(self.config, 'img_mean', None)
        preprocess_name = get(self.config, 'preprocessing_name', None)
        random_scale = get(self.config, 'random_scale', False)
        random_mirror = get(self.config, 'random_mirror', True)
        batch_size = get(self.config, 'batch_size', 8)

        input_names = []
        output_names = []
        for file in os.listdir(os.path.join(dataset_dir, input_dir)):
            cwd = os.getcwd()
            input_names.append(cwd + "/" + os.path.join(dataset_dir, input_dir) + "/" + file)
        for file in os.listdir(os.path.join(dataset_dir, output_dir)):
            cwd = os.getcwd()
            output_names.append(cwd + "/" + os.path.join(dataset_dir, output_dir) + "/" + file)
        
        input_names.sort()
        output_names.sort()
        dataset = tf.data.Dataset.from_tensor_slices((input_names, output_names))
        dataset = dataset.map(lambda x, y: _parse_function(x, y, img_mean, self.class_dict), num_parallel_calls=threads) # map接收 一个函数 ，Dataset中的每个元素都会被当作这个函数的输入

        logging.info('preproces -- {}'.format(preprocess_name))
        if preprocess_name == 'augment':
            if random_mirror:
                dataset = dataset.map(_image_mirroring, num_parallel_calls=threads)
            if random_scale:
                dataset = dataset.map(_image_scaling, num_parallel_calls=threads)

            # dataset = dataset.map(lambda x, y: _random_crop_and_pad_image_and_labels(x, y, crop_h, crop_w),
            #                   num_parallel_calls=threads)
            dataset = dataset.map(lambda image, label: _apply_with_random_selector(image, lambda x, ordering: _distort_color
                                                                               (x, ordering, fast_mode=True),
                                                                               num_cases=4, label=label))
            
        dataset = dataset.map(lambda image, label: _check_size(image, label, crop_h, crop_w))
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        self.dataset = dataset
        
    def get_one_batch(self):
        self.prepare_data()
        self.iterator = self.dataset.make_one_shot_iterator()
        return self.iterator.get_next()


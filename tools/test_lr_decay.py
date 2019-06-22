#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf


def test_piecewise_constant():
    """
    分度常数衰减
    tf.train.piecewise_constant()参数为:
        1、x: 标量，指代训练次数
        2、boundaries： 学习率参数应用区间列表
        3、values： 学习率列表，values的长度比boundaries的长度多一个
        4、name： 操作的名称
    :return:
    """
    boundaries = [10, 20, 30]
    learing_rates = [0.1, 0.07, 0.025, 0.0125]

    y = []
    N = 40

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #num_epoch = tf.Variable(0, name='global_step', trainable=False)
        for num_epoch in range(N):
            learing_rate = tf.train.piecewise_constant(num_epoch, boundaries=boundaries, values=learing_rates)
            lr = sess.run([learing_rate])
            y.append(lr)

    x = range(N)
    plt.plot(x, y, 'r-', linewidth=2)
    plt.title('piecewise_constant')
    plt.show()


def test_exponential_decay():
    """
    指数衰减
    tf.train.exponential_decay()参数为:
        1、learning_rate: 初始学习率
        2、global_step: 当前训练轮次，epoch
        3、decay_step: 定义衰减周期，跟参数staircase配合，可以在decay_step个训练轮次内保持学习率不变
        4、decay_rate，衰减率系数
        5、staircase： 定义是否是阶梯型衰减，还是连续衰减，默认是False，即连续衰减（标准的指数型衰减）
        name： 操作名称
    计算公式为:
    decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
    :return:decayed_learning_rate
    """
    y = []
    z = []
    N = 200

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for num_epoch in range(N):
            # 阶梯型衰减
            learing_rate1 = tf.train.exponential_decay(
                learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=True)
            # 标准指数型衰减
            learing_rate2 = tf.train.exponential_decay(
                learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=False)
            lr1 = sess.run([learing_rate1])
            lr2 = sess.run([learing_rate2])
            y.append(lr1)
            z.append(lr2)

    x = range(N)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 0.55])

    plt.plot(x, y, 'r-', linewidth=2)
    plt.plot(x, z, 'g-', linewidth=2)
    plt.title('exponential_decay')
    ax.set_xlabel('step')
    ax.set_ylabel('learing rate')
    plt.show()


def test_natural_exp_decay():
    """
    自然指数衰减：自然指数衰减是指数衰减的一种特殊情况,
    自然指数衰减对学习率的衰减程度要远大于一般的指数衰减，一般用于可以较快收敛的网络，或者是训练数据集比较大的场合。
    tf.train.natural_exp_decay()参数为:
        1、learning_rate: 初始学习率
        2、global_step: 当前训练轮次，epoch
        3、decay_step: 定义衰减周期，跟参数staircase配合，可以在decay_step个训练轮次内保持学习率不变
        4、decay_rate，衰减率系数
        5、staircase： 定义是否是阶梯型衰减，还是连续衰减，默认是False，即连续衰减（标准的指数型衰减）
        6、name： 操作名称
    计算公式为:
    decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
    :return:decayed_learning_rate
    """
    y = []
    z = []
    w = []
    m = []
    N = 200

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for num_epoch in range(N):
            # 阶梯型衰减
            learing_rate1 = tf.train.natural_exp_decay(
                learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=True)

            # 标准指数型衰减
            learing_rate2 = tf.train.natural_exp_decay(
                learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=False)

            # 阶梯型指数衰减
            learing_rate3 = tf.train.exponential_decay(
                learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=True)

            # 标准指数衰减
            learing_rate4 = tf.train.exponential_decay(
                learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=False)

            lr1 = sess.run([learing_rate1])
            lr2 = sess.run([learing_rate2])
            lr3 = sess.run([learing_rate3])
            lr4 = sess.run([learing_rate4])

            y.append(lr1)
            z.append(lr2)
            w.append(lr3)
            m.append(lr4)

    x = range(N)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 0.55])

    plt.plot(x, y, 'r-', linewidth=2)
    plt.plot(x, z, 'g-', linewidth=2)
    plt.plot(x, w, 'r-', linewidth=2)
    plt.plot(x, m, 'g-', linewidth=2)

    plt.title('natural_exp_decay')
    ax.set_xlabel('step')
    ax.set_ylabel('learing rate')
    plt.show()


def test_polynomial_decay():
    """
    多项式衰减:
    多项式衰减是这样一种衰减机制：定义一个初始的学习率，一个最低的学习率，按照设置的衰减规则，学习率从初始学习率逐渐降低到最低的学习率，
    并且可以定义学习率降低到最低的学习率之后，是一直保持使用这个最低的学习率，还是到达最低的学习率之后再升高学习率到一定值，然后再降低到最低的学习率（反复这个过程）
    tf.train.polynomial_decay()参数为:
        1、learning_rate: 初始学习率
        2、global_step: 当前训练轮次，epoch
        3、decay_steps: 定义衰减周期
        4、end_learning_rate：最小的学习率，默认值是0.0001
        5、power： 多项式的幂，默认值是1,即线性的
        6、cycle： 定义学习率是否到达最低学习率后升高，然后再降低，默认False，保持最低学习率
        7、name： 操作名称
    计算公式为:
    global_step = min(global_step, decay_steps)
    decayed_learning_rate = (learning_rate - end_learning_rate) *
                            (1 - global_step / decay_steps) ^ (power) +
                            end_learning_rate
    如果定义 cycle为True，学习率在到达最低学习率后往复升高降低，此时学习率计算公式为：
    decay_steps = decay_steps * ceil(global_step / decay_steps)
    decayed_learning_rate = (learning_rate - end_learning_rate) *
                            (1 - global_step / decay_steps) ^ (power) +
                            end_learning_rate
    :return: decayed_learning_rate
    """
    y = []
    z = []
    N = 200

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for global_step in range(N):
            # cycle=False
            learing_rate1 = tf.train.polynomial_decay(
                learning_rate=0.1, global_step=global_step, decay_steps=50,
                end_learning_rate=0.01, power=0.5, cycle=False)
            # cycle=True
            learing_rate2 = tf.train.polynomial_decay(
                learning_rate=0.1, global_step=global_step, decay_steps=50,
                end_learning_rate=0.01, power=0.5, cycle=True)

            lr1 = sess.run([learing_rate1])
            lr2 = sess.run([learing_rate2])
            y.append(lr1)
            z.append(lr2)

    x = range(N)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, z, 'g-', linewidth=2)    # 绿色的学习率衰减曲线对应 cycle = True，下降后往复升降
    plt.plot(x, y, 'r--', linewidth=2)   # 红色的学习率衰减曲线对应 cycle = False，下降后不再上升，保持不变
    plt.title('polynomial_decay')
    ax.set_xlabel('step')
    ax.set_ylabel('learing rate')
    plt.show()


def test_cosine_decay():
    """
    余弦衰减:
    余弦衰减的衰减机制跟余弦函数相关，形状也大体上是余弦形状
    tf.train.cosine_decay():
        1、learning_rate：初始学习率
        2、global_step: 当前训练轮次，epoch
        3、decay_steps： 衰减步数，即从初始学习率衰减到最小学习率需要的训练轮次
        4、alpha=： 最小学习率
        5、name： 操作的名称
    计算公式:
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed
    改进的余弦衰减方法还有：
    线性余弦衰减，对应函数 tf.train.linear_cosine_decay()
    噪声线性余弦衰减，对应函数 tf.train.noisy_linear_cosine_decay()
    :return:
    """
    y = []
    z = []
    w = []
    N = 200
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for global_step in range(N):
            # 余弦衰减
            learing_rate1 = tf.train.cosine_decay(
                learning_rate=0.1, global_step=global_step, decay_steps=50)

            # 线性余弦衰减
            learing_rate2 = tf.train.linear_cosine_decay(
                learning_rate=0.1, global_step=global_step, decay_steps=50,
                num_periods=0.2, alpha=0.5, beta=0.2)

            # 噪声线性余弦衰减
            learing_rate3 = tf.train.noisy_linear_cosine_decay(
                learning_rate=0.1, global_step=global_step, decay_steps=50,
                initial_variance=0.01, variance_decay=0.1, num_periods=0.2, alpha=0.5, beta=0.2)

            lr1 = sess.run([learing_rate1])
            lr2 = sess.run([learing_rate2])
            lr3 = sess.run([learing_rate3])
            y.append(lr1)
            z.append(lr2)
            w.append(lr3)

    x = range(N)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, z, 'b-', linewidth=2)  # 蓝色：线性余弦衰减，学习率从初始线性过渡到最低学习率
    plt.plot(x, y, 'r-', linewidth=2)  # 红色：标准余弦衰减，学习率从初始曲线过渡到最低学习率
    plt.plot(x, w, 'g-', linewidth=2)  # 绿色：噪声线性余弦衰减，在线性余弦衰减基础上增加了随机噪声
    plt.title('cosine_decay')
    ax.set_xlabel('step')
    ax.set_ylabel('learing rate')
    plt.show()

if __name__ == '__main__':
    test_cosine_decay()
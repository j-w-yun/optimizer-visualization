import math
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def f1(x, y):
#     z = x * x + y * y
    z = (x * x + y * y) * (math.cos(x + y) + 1.2)
    return z


def f2(x, y):
    z = (tf.square(x) + tf.square(y)) * (tf.cos(x + y) + 1.2)
    return z


plt.ion()

for x in np.arange(-10.0, 10.0, 0.1):
    for y in np.arange(-10.0, 10.0, 0.1):
        z = f1(x, y)

        if 1 < z < 1.3:
            plt.scatter(x, y, color='g', s=1)

        if 2 < z < 2.5:
            plt.scatter(x, y, color='g', s=1)

        if 5 < z < 6:
            plt.scatter(x, y, color='g', s=1)

plt.draw()

x = tf.Variable(-2, [1], dtype=tf.float32)
y = tf.Variable(4, [1], dtype=tf.float32)

with tf.Session() as sess:
    cost = f2(x, y)

    # change optimizers here
    op = tf.train.AdamOptimizer(0.01).minimize(cost)

    sess.run(tf.global_variables_initializer())

    for _ in range(10000):
        _, x_val, y_val = sess.run([op, x, y])
        plt.scatter(x_val, y_val, color='r', s=1)
        plt.pause(.001)


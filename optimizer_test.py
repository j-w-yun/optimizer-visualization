import math
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def contour_f(x, y):
    z = (x ** 2 + y ** 2) * (math.cos(x + y) + 1.2)
    return z


def tf_f(x, y):
    z = (tf.square(x) + tf.square(y)) * (tf.cos(x + y) + 1.2)
    return z


plt.ion()

for x in np.arange(-10.0, 10.0, 0.1):
    for y in np.arange(-10.0, 10.0, 0.1):
        z = contour_f(x, y)

        if 1 < z < 1.3:
            plt.scatter(x, y, color='g', s=1)

        if 2 < z < 2.5:
            plt.scatter(x, y, color='g', s=1)

        if 5.5 < z < 6.5:
            plt.scatter(x, y, color='g', s=1)

plt.draw()

x_i = -1.8
y_i = 3.8

x = tf.Variable(x_i, [1], dtype=tf.float32)
y = tf.Variable(y_i, [1], dtype=tf.float32)

with tf.Session() as sess:
    cost = tf_f(x, y)

    # change optimizers here
    op = tf.train.AdamOptimizer(0.5).minimize(cost)

    sess.run(tf.global_variables_initializer())

    last_x = x_i
    last_y = y_i
    last_point = None

    for _ in range(50):
        _, x_val, y_val = sess.run([op, x, y])

        if last_point:
            last_point.remove()
        last_point = plt.scatter(x_val, y_val, color='r', s=20)

        if last_x and last_y:
            plt.plot([last_x, x_val], [last_y, y_val], color='r')
        last_x = x_val
        last_y = y_val
        plt.pause(.001)

print("done")
plt.pause(100)
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def contour_f(x, y):
#     z = (x ** 2 + y ** 2) * (math.cos(x + y) + 1.2)
    z = -1 * math.sin(x * x) * math.cos(3 * y * y) * math.exp(-(x * y) ** 2) - math.exp(-(x + y) ** 2)
    return z


def tf_f(x, y):
#     z = (tf.square(x) + tf.square(y)) * (tf.cos(x + y) + 1.2)
    z = -1 * tf.sin(x * x) * tf.cos(3 * y * y) * tf.exp(-(x * y) * (x * y)) - tf.exp(-(x + y) * (x + y))
    return z


plt.ion()

for x in np.arange(-1.5, 1.5, 0.02):  # (-1.5, 0.6, 0.02)
    for y in np.arange(-0.75, 1.0, 0.02):  # (-0.5, 1.0, 0.02)
        z = contour_f(x, y)

#         if 1 < z < 1.3:
#             plt.scatter(x, y, color='g', s=1)
#         if 2 < z < 2.5:
#             plt.scatter(x, y, color='g', s=1)
#         if 5.5 < z < 6.5:
#             plt.scatter(x, y, color='g', s=1)

        if -1.2 < z < -1.1:
            plt.scatter(x, y, color=(0.1, 0.1, 0.5, 1.0), s=1)

        elif -0.9 < z < -0.8:
            plt.scatter(x, y, color=(0.1, 0.1, 0.5, 0.8), s=1)

        elif -0.6 < z < -0.5:
            plt.scatter(x, y, color=(0.1, 0.1, 0.5, 0.6), s=1)

        elif -0.3 < z < -0.2:
            plt.scatter(x, y, color=(0.1, 0.1, 0.5, 0.4), s=1)

        elif 0.0 < z < 0.1:
            plt.scatter(x, y, color=(0.1, 0.1, 0.5, 0.2), s=1)

plt.draw()

x_i = 0.5
y_i = 0.75

x = tf.Variable(x_i, [1], dtype=tf.float32)
y = tf.Variable(y_i, [1], dtype=tf.float32)

with tf.Session() as sess:
    cost = tf_f(x, y)

    # change optimizers here
    op = tf.train.AdamOptimizer(0.05).minimize(cost)

    sess.run(tf.global_variables_initializer())

    last_x = x_i
    last_y = y_i
    last_point = None
    last_color = 1.0

    for _ in range(100):
        _, x_val, y_val = sess.run([op, x, y])

        if last_point:
            last_point.remove()
            last_color *= 0.97
        last_point = plt.scatter(x_val, y_val, color=(1., 0., last_color), s=20)

        if last_x and last_y:
            plt.plot([last_x, x_val], [last_y, y_val], color=(1., 0., last_color))
        last_x = x_val
        last_y = y_val
        plt.pause(.001)

print("done")
plt.pause(10000)

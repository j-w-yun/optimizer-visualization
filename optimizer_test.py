import math
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def f(x, y):
    z = -1 * tf.sin(x * x) * tf.cos(3 * y * y) * tf.exp(-(x * y) * (x * y)) - tf.exp(-(x + y) * (x + y))
    return z


plt.ion()
# plt.figure(figsize=(3, 2), dpi=300)
# params = {'legend.fontsize': 4,
#           'legend.handlelength': 4}
# plt.rcParams.update(params)
plt.axis('off')
plt.tight_layout()

x = np.arange(-1.5, 1.5, 0.03)
y = np.arange(-1.5, 1.5, 0.03)
x_in = np.array([elem_x for elem_x in x for elem_y in y])
y_in = np.array([elem_y for elem_x in x for elem_y in y])
z = f(x_in, y_in)

with tf.Session() as sess:
    output = sess.run(z)

    index = 0
    for x_val in x:
        for y_val in y:
            if -1.2 < output[index] < -1.1:
                plt.scatter(x_val, y_val, color=(0.1, 0.1, 0.5, 1.0), s=0.7)
            elif -0.9 < output[index] < -0.8:
                plt.scatter(x_val, y_val, color=(0.1, 0.1, 0.5, 0.8), s=0.7)
            elif -0.6 < output[index] < -0.5:
                plt.scatter(x_val, y_val, color=(0.1, 0.1, 0.5, 0.6), s=0.7)
            elif -0.3 < output[index] < -0.2:
                plt.scatter(x_val, y_val, color=(0.1, 0.1, 0.5, 0.4), s=0.7)
            elif 0.0 < output[index] < 0.1:
                plt.scatter(x_val, y_val, color=(0.1, 0.1, 0.5, 0.2), s=0.7)
            index += 1

plt.draw()

x_i = 0.5
y_i = 0.75

x_var = []
y_var = []
for i in range(7):
    x_var.append(tf.Variable(x_i, [1], dtype=tf.float32))
    y_var.append(tf.Variable(y_i, [1], dtype=tf.float32))

cost = []
for i in range(7):
    cost.append(f(x_var[i], y_var[i]))

ops = []
ops.append(tf.train.AdadeltaOptimizer(20).minimize(cost[0]))
ops.append(tf.train.AdagradOptimizer(0.03).minimize(cost[1]))
ops.append(tf.train.AdamOptimizer(0.01).minimize(cost[2]))
ops.append(tf.train.FtrlOptimizer(0.01).minimize(cost[3]))
ops.append(tf.train.GradientDescentOptimizer(0.01).minimize(cost[4]))
ops.append(tf.train.MomentumOptimizer(0.01, 0.9).minimize(cost[5]))
ops.append(tf.train.RMSPropOptimizer(0.01).minimize(cost[6]))

ops_label = ["Adadelta",
             "Adagrad",
             "Adam",
             "Ftrl",
             "GD",
             "Momentum",
             "RMSProp"]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    last_x = []
    last_y = []
    last_point = []
    for i in range(7):
        last_x.append(x_i)
        last_y.append(y_i)
        last_point.append(None)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for iter in range(300):
        for i, op in enumerate(ops):
            _, x_val, y_val = sess.run([op, x_var[i], y_var[i]])

            if last_point[i]:
                last_point[i].remove()
            last_point[i] = plt.scatter(x_val, y_val, color=colors[i], s=20, label=ops_label[i])

            if last_x[i] and last_y[i]:
                plt.plot([last_x[i], x_val], [last_y[i], y_val], color=colors[i])
            last_x[i] = x_val
            last_y[i] = y_val

        plt.legend(last_point, ops_label, prop={'size': 7})
        plt.savefig(str(iter) + '.png')

print("done")
